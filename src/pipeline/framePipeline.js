'use strict';

const { flowLog, flowWarn, flowError } = require('../utils/log');
const { getServerLabel } = require('../config');

// 批量推理的帧数范围：[BATCH_MIN, BATCH_MAX]
const BATCH_MIN = 2;
const BATCH_MAX = 6;

function createFramePipeline({
  getConfig,
  inferenceService,
  embeddingService,
  faceService,
  memoryStore,
  vectorIndex,
  keyframeSelector,
  batcher,
}) {
  /** 单帧完整处理：推理 → 嵌入 → 人脸 → 存储 */
  async function processOne(landed) {
    const cfg = getConfig();
    const started = Date.now();
    flowLog('管道', '开始处理帧', { id: landed.id, source: landed.source });

    let recog;
    try {
      recog = await inferenceService.recognizeFrame({
        framePath: landed.framePath,
        prompt: cfg.prompt,
        ollamaBase: cfg.ollamaBase,
      });
    } catch (e) {
      flowError('管道', '视觉推理失败', { id: landed.id, error: e.message });
      return null;
    }

    return _buildAndSave(landed, recog, Date.now() - started);
  }

  /**
   * 公共保存逻辑：嵌入 → 人脸 → SQLite → 向量索引
   * 抽出来供 processOne 和 processBatchInference 共用
   */
  async function _buildAndSave(landed, recog, inferElapsed) {
    const cfg = getConfig();
    const memory = {
      id: landed.id,
      framePath: landed.framePath,
      description: recog.description,
      model: recog.model,
      serverLabel: getServerLabel(cfg, cfg.ollamaBase),
      timestamp: recog.timestamp,
      captureTime: landed.captureTime || recog.timestamp,
      inferenceTime: recog.inferenceTime,
    };

    try {
      const embedding = await embeddingService.embed(memory.description);
      if (embedding) { memory.embedding = embedding; }
      else { flowWarn('管道', '嵌入返回空值', { id: landed.id }); }
    } catch (e) {
      flowWarn('管道', '嵌入失败 (不影响保存)', { id: landed.id, error: e.message });
    }

    try {
      if (faceService.isReady()) {
        const faces = await faceService.detect(landed.framePath);
        if (faces && faces.length > 0) {
          memory.faces = faces;
          flowLog('管道', '人脸已附加', { id: landed.id, count: faces.length });
        }
      }
    } catch (e) {
      flowWarn('管道', '人脸检测失败 (不影响保存)', { id: landed.id, error: e.message });
    }

    try {
      await memoryStore.save(memory);
    } catch (e) {
      // busy_timeout 下极少触发；若仍失败则记录 WARN 并继续（帧不丢，只是记忆未存）
      flowWarn('管道', 'SQLite 保存失败', { id: landed.id, error: e.message });
      return null;
    }

    if (memory.embedding) {
      try {
        await vectorIndex.upsert({
          id: memory.id,
          vector: memory.embedding,
          captureTime: memory.captureTime,
          timestamp: memory.timestamp,
          description: memory.description,
        });
      } catch (e) {
        flowWarn('管道', '向量索引更新失败', { id: landed.id, error: e.message });
      }
    }

    flowLog('管道', '帧处理完成', {
      id: landed.id,
      inferenceMs: recog.inferenceTime,
      hasEmbedding: !!memory.embedding,
      hasFaces: !!(memory.faces && memory.faces.length > 0),
    });
    return memory;
  }

  /**
   * 批量推理：2-6 帧一次 VLM API 调用（主要性能收益）。
   * 推理完成后串行执行嵌入/人脸/SQLite/LanceDB，避免并发写导致 database is locked。
   */
  async function processBatchInference(frames) {
    flowLog('管道', '批量推理', { count: frames.length });
    let recogResults;
    try {
      recogResults = await inferenceService.recognizeFrameBatch(frames);
    } catch (e) {
      flowError('管道', '批量视觉推理失败，逐帧回退', { error: e.message });
      const results = [];
      for (const f of frames) { const r = await processOne(f); if (r) results.push(r); }
      return results;
    }

    // 串行保存：SQLite / LanceDB 不支持同一进程内并发写，串行无性能损失
    // （批量收益已体现在上方一次 VLM 调用中）
    const results = [];
    for (let i = 0; i < frames.length; i++) {
      try {
        const r = await _buildAndSave(frames[i], recogResults[i], 0);
        if (r) results.push(r);
      } catch (e) {
        flowError('管道', '批量保存异常', { frameId: frames[i] && frames[i].id, error: e.message });
      }
    }
    return results;
  }

  async function processBatch(batch) {
    const selected = keyframeSelector.select(batch);
    if (!selected || selected.length === 0) {
      flowWarn('管道', '关键帧筛选后为空，跳过', { batchSize: batch.length });
      return [];
    }

    flowLog('管道', '处理批次', { batchSize: batch.length, selectedSize: selected.length });

    const results = [];
    let i = 0;
    while (i < selected.length) {
      const remaining = selected.length - i;
      if (remaining === 1) {
        // 单帧：普通单图推理
        const r = await processOne(selected[i]);
        if (r) results.push(r);
        i += 1;
      } else if (remaining <= BATCH_MAX) {
        // 2-6 帧：一次批量推理
        const chunk = selected.slice(i, i + remaining);
        const rs = await processBatchInference(chunk);
        results.push(...rs);
        i += remaining;
      } else {
        // >6 帧：取前 BATCH_MAX 帧批量处理，剩余留给下一轮循环
        const chunk = selected.slice(i, i + BATCH_MAX);
        const rs = await processBatchInference(chunk);
        results.push(...rs);
        i += BATCH_MAX;
      }
    }
    return results;
  }

  function onLanded(landed) {
    batcher.push(landed);
  }

  return {
    onLanded,
    processOne,
    processBatch,
  };
}

module.exports = { createFramePipeline };
