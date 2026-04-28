'use strict';

const fs = require('fs');
const path = require('path');
const { CAPTURE_DIR, FRAMES_DIR, ensureDataDirs } = require('../utils/paths');
const { flowLog, flowWarn, flowError, flowDebug } = require('../utils/log');

function nextFrameNum() {
  if (!fs.existsSync(FRAMES_DIR)) return 1;
  const files = fs.readdirSync(FRAMES_DIR).filter((f) => /^frame_\d+\.(png|jpg)$/.test(f));
  if (files.length === 0) return 1;
  const nums = files.map((f) => parseInt(f.match(/frame_(\d+)/)[1], 10));
  return Math.max.apply(null, nums) + 1;
}

function writeLatestPreview(jpeg) {
  try {
    ensureDataDirs();
    fs.writeFileSync(path.join(CAPTURE_DIR, 'latest.jpg'), jpeg);
  } catch (e) {
    flowWarn('采帧', '写入 latest.jpg 失败', { error: e.message });
  }
}

function persistFrame(jpeg) {
  ensureDataDirs();
  const id = nextFrameNum();
  const filename = 'frame_' + String(id).padStart(3, '0') + '.jpg';
  const framePath = path.join(FRAMES_DIR, filename);
  fs.writeFileSync(framePath, jpeg);
  return { id: String(id), framePath, filename };
}

function createCaptureController({ source, quickFilter, onLanded, writePreview }) {
  let isCapturing = false;
  let landedCount = 0;
  let droppedCount = 0;
  // 'server'  = 只处理 ffmpeg 帧（服务器摄像头模式）
  // 'local'   = 只处理上传帧（浏览器本地摄像头模式）
  // null      = 预览模式，不落地
  let captureMode = null;

  // 并发保护：quickFilter.shouldKeep 可能是 async（如 pixelDiff 需要解码 JPEG）
  // 同时只处理一帧，来不及处理的帧直接丢弃（不会丢失内容，只是减少重复帧）
  let filterBusy = false;

  function handleRawFrame(rawFrame) {
    // 预览：所有 ffmpeg 帧都更新 latest.jpg，与录制状态无关
    if (writePreview !== false && !rawFrame.uploaded) writeLatestPreview(rawFrame.jpeg);
    if (!isCapturing || !captureMode) return;

    // 按来源过滤：本地模式只处理上传帧，服务器模式只处理 ffmpeg 帧
    if (captureMode === 'local' && !rawFrame.uploaded) return;
    if (captureMode === 'server' && rawFrame.uploaded) return;

    if (filterBusy) {
      droppedCount++;
      flowDebug('采帧', '初筛忙碌，帧跳过', { dropped: droppedCount });
      return;
    }

    // 立即拷贝 jpeg buffer，防止 splitter 的 buf.slice 共享内存在 async shouldKeep
    // 期间被后续 pipe 数据覆盖，导致比较和落地的是错误帧内容
    const frame = { jpeg: Buffer.from(rawFrame.jpeg), ts: rawFrame.ts, source: rawFrame.source, uploaded: rawFrame.uploaded };

    filterBusy = true;
    Promise.resolve()
      .then(() => quickFilter.shouldKeep(frame))
      .then((keep) => {
        if (!keep) {
          droppedCount++;
          flowDebug('采帧', '帧被 QuickFilter 过滤', { dropped: droppedCount });
          return;
        }
        const captureTime = new Date(frame.ts || Date.now()).toISOString();
        let landed;
        try {
          landed = persistFrame(frame.jpeg);
        } catch (e) {
          flowError('采帧', '帧落地失败', { error: e.message });
          return;
        }
        landedCount++;
        flowDebug('采帧', '帧落地', { id: landed.id, captureTime, jpegSize: frame.jpeg.length, landed: landedCount });
        if (typeof onLanded === 'function') {
          onLanded({ ...landed, captureTime, jpegSize: frame.jpeg.length, source: source.name });
        }
      })
      .catch((e) => {
        flowWarn('采帧', 'quickFilter 异常，帧丢弃', { error: e.message });
      })
      .finally(() => {
        filterBusy = false;
      });
  }

  /**
   * 仅启动采帧源用于实时预览（不落地、不送 VLM）。
   * 服务启动时调用，保证左上角实时帧始终有画面。
   * 调用 start() 后会在此基础上开启记忆录制；stop() 只停录制不停预览。
   */
  async function startPreview() {
    if (source.isRunning()) return;
    flowLog('控制', '启动预览采帧', { source: source.name });
    if (typeof source.start === 'function') {
      await source.start(handleRawFrame);
    }
  }

  async function start() {
    if (isCapturing) return;
    captureMode = 'server';
    isCapturing = true;
    landedCount = 0;
    droppedCount = 0;
    quickFilter.reset && quickFilter.reset();
    flowLog('控制', '开始记忆', { source: source.name, quickFilter: quickFilter.name });
    // 若 source 尚未在预览模式中运行，则在此启动
    if (typeof source.start === 'function' && !source.isRunning()) {
      await source.start(handleRawFrame);
    }
  }

  /**
   * 本地（浏览器上传）模式：仅激活 isCapturing，不启动 ffmpeg。
   * 帧通过 ingestUploaded() 进入流水线。ffmpeg 预览循环继续运行但不落地。
   */
  async function startLocal() {
    if (isCapturing) return;
    captureMode = 'local';
    isCapturing = true;
    landedCount = 0;
    droppedCount = 0;
    quickFilter.reset && quickFilter.reset();
    flowLog('控制', '开始记忆 (本地上传模式)', { quickFilter: quickFilter.name });
  }

  async function stop() {
    if (!isCapturing) return;
    captureMode = null;
    isCapturing = false;
    flowLog('控制', '停止记忆', { landedFrames: landedCount, droppedFrames: droppedCount });
    // source 继续运行以保持实时预览；调用 destroy() 才完全停止
  }

  /** 完全停止，包括预览循环（服务关闭时调用）。 */
  async function destroy() {
    captureMode = null;
    isCapturing = false;
    if (typeof source.stop === 'function') {
      await source.stop();
    }
    flowLog('控制', '采帧源已关闭');
  }

  function ingestUploaded({ filePath, ts }) {
    const jpeg = fs.readFileSync(filePath);
    // 若当前处于本地模式，直接走正常流水线
    // 若未在录制（如 /api/capture-native 直接调用），临时切换为本地模式处理后恢复
    const wasCapturing = isCapturing;
    const prevMode = captureMode;
    if (!wasCapturing) {
      captureMode = 'local';
      isCapturing = true;
    }
    handleRawFrame({ jpeg, ts: ts || Date.now(), uploaded: true });
    if (!wasCapturing) {
      const restore = () => {
        if (!filterBusy) { isCapturing = false; captureMode = prevMode; return; }
        setTimeout(restore, 50);
      };
      setTimeout(restore, 50);
    }
  }

  return {
    startPreview,
    start,
    startLocal,
    stop,
    destroy,
    ingestUploaded,
    isCapturing: () => isCapturing,
    sourceName: () => source.name,
    stats: () => ({ landedCount, droppedCount }),
  };
}

module.exports = {
  createCaptureController,
  nextFrameNum,
  writeLatestPreview,
};
