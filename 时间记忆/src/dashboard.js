#!/usr/bin/env node
'use strict';

const { ensureDataDirs } = require('./utils/paths');
const { flowLog, flowWarn, flowError, currentLogFile } = require('./utils/log');
const { loadUserConfig } = require('./config');

const { createInferenceService } = require('./inference');
const { createEmbeddingService } = require('./embeddings');
const { createFaceLibrary } = require('./faces/library');
const { createFaceService } = require('./faces');
const { createMemoryStore } = require('./memory');
const { createVectorIndex } = require('./vector');

const { createQuickFilter } = require('./capture/quickFilter');
const { createFfmpegAvfoundationSource } = require('./capture/sources/ffmpegAvfoundation');
const { createFfmpegSnapshotSource } = require('./capture/sources/ffmpegSnapshot');
const { createCaptureController } = require('./capture/controller');

const { createKeyframeSelector } = require('./pipeline/keyframe');
const { createBatcher } = require('./pipeline/batcher');
const { createFramePipeline } = require('./pipeline/framePipeline');

const { createQueryService } = require('./query');
const { createLifecycleService } = require('./lifecycle');

const { createApp } = require('./server/app');
const { listFrames: _listFrames } = require('./server/routes/memory');

const PORT = process.argv.includes('--port')
  ? parseInt(process.argv[process.argv.indexOf('--port') + 1], 10)
  : 8080;

async function main() {
  ensureDataDirs();
  flowLog('启动', '识境时空记忆启动', { pid: process.pid, logFile: currentLogFile(), nodeVersion: process.version });
  let config = loadUserConfig();
  flowLog('启动', '配置加载完成', {
    capture: config.capture.source,
    pipeline: { quickFilter: config.pipeline.quickFilter, keyframe: config.pipeline.keyframe },
    memory: config.memory.store,
    vector: config.vector.store,
  });

  if (config.pipeline.keyframe !== 'passthrough' && !config.pipeline.batch.enabled) {
    flowWarn('配置', 'keyframe 选择器为 "' + config.pipeline.keyframe + '"，但 batch.enabled=false，选择器仅对单帧操作（无实际过滤效果）。请同时设置 pipeline.batch.enabled=true 以启用批量聚合。');
  }
  const getConfig = () => config;
  const setConfig = (next) => { config = next; };

  const inferenceService = createInferenceService(getConfig);
  const embeddingService = createEmbeddingService(getConfig);

  const faceLibrary = createFaceLibrary();

  const memoryStore = createMemoryStore(config);
  await memoryStore.init();

  // 人脸识别到后同步写入 SQLite faces 表（fire-and-forget，不阻塞检测流程）
  const faceService = createFaceService(faceLibrary, {
    onFaceUpdated: (personId, faceData) => {
      if (typeof memoryStore.upsertFace === 'function') {
        try { memoryStore.upsertFace(personId, faceData); }
        catch (e) { flowWarn('人脸', 'upsertFace 失败', { personId, error: e.message }); }
      }
    },
  });

  const vectorIndex = createVectorIndex(config);
  await vectorIndex.init();
  vectorIndex.bootstrap(memoryStore, (text) => embeddingService.embed(text)).catch((e) => {
    flowLog('索引', '构建失败', { error: e.message });
  });

  const quickFilter = createQuickFilter(config);
  // 默认使用 snapshot 模式（每次独立 ffmpeg 进程 = 每帧都是当前真实画面）
  // 流式模式（ffmpeg-stream）在 macOS AVFoundation 下存在首帧后冻结的已知问题
  const ffmpegMode = (config.capture.ffmpeg && config.capture.ffmpeg.mode) || 'snapshot';
  const captureSource = ffmpegMode === 'stream'
    ? createFfmpegAvfoundationSource(config.capture.ffmpeg)
    : createFfmpegSnapshotSource(config.capture.ffmpeg);

  const keyframeSelector = createKeyframeSelector(config);

  const pipeline = createFramePipeline({
    getConfig,
    inferenceService,
    embeddingService,
    faceService,
    memoryStore,
    vectorIndex,
    keyframeSelector,
    batcher: { push: () => {} },
  });
  const batcher = createBatcher(config, async (batch) => {
    try { await pipeline.processBatch(batch); }
    catch (e) { flowError('管道', 'processBatch 异常', { error: e.message }); }
  });
  pipeline.onLanded = (landed) => batcher.push(landed);

  const captureController = createCaptureController({
    source: captureSource,
    quickFilter,
    onLanded: (landed) => pipeline.onLanded(landed),
    writePreview: true,
  });

  const queryService = createQueryService({
    inferenceService,
    embeddingService,
    vectorIndex,
    memoryStore,
    faceLibrary,
    framesProvider: { list: () => _listFrames(memoryStore) },
    getConfig,
  });

  const lifecycle = createLifecycleService({ memoryStore, vectorIndex });
  lifecycle.startScheduler(60 * 60 * 1000);

  faceService.init().catch((e) => flowLog('人脸', '初始化异常', { error: e.message }));

  // 服务启动后立即开启预览采帧（仅更新 latest.jpg，不落地不送 VLM）
  // 保证左上角实时帧在未开始录制时也能持续刷新
  captureController.startPreview().catch((e) => {
    flowWarn('控制', '预览采帧启动失败（摄像头可能未就绪）', { error: e.message });
  });

  const ctx = {
    getConfig,
    setConfig,
    captureController,
    queryService,
    memoryStore,
    vectorIndex,
    faceService,
    faceLibrary,
  };
  const app = createApp(ctx);
  const { server, httpsServer } = await app.start(PORT);

  process.on('SIGINT', async () => {
    console.log('\n⏹️  正在关闭...');
    try { await captureController.destroy(); } catch (_) {}
    if (httpsServer) httpsServer.close();
    server.close(() => process.exit(0));
  });
}

main().catch((err) => {
  flowError('启动', '启动失败', { error: err.message, stack: err.stack });
  console.error('启动失败：', err);
  process.exit(1);
});
