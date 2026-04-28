'use strict';

/**
 * 快照采帧源（Snapshot Source）
 *
 * 背景：macOS AVFoundation + ffmpeg 流式模式（-f image2pipe）存在"首帧后冻结"问题：
 * ffmpeg 只正确输出第一帧，此后所有帧 bit-identical，即使画面已改变。
 *
 * 解法：每次独立启动 `ffmpeg -vframes 1` 拍一张快照，每次都是新的 AVFoundation session，
 * 保证拿到当前真实画面。代价是每次有 ~1-2s 的 ffmpeg 启动延迟，因此帧率远低于流式，
 * 但对"记忆系统"场景下 5-10s 级别的采样间隔完全够用。
 */

const { spawn } = require('child_process');
const { flowLog, flowWarn, flowError, flowDebug } = require('../../utils/log');

const STEP = '采帧';

function createFfmpegSnapshotSource(opts) {
  const o = opts || {};
  const deviceIndex = Number.isFinite(o.deviceIndex) ? o.deviceIndex : 0;
  const width = Number(o.width) > 0 ? Number(o.width) : 1280;
  const height = Number(o.height) > 0 ? Number(o.height) : 720;
  const quality = Number(o.quality) > 0 ? Number(o.quality) : 5;
  // 每次快照之间的最小间隔（毫秒）；ffmpeg 启动+采帧约需 1-2s，总采帧周期 = 启动时间 + intervalMs
  const intervalMs = Number(o.snapshotIntervalMs) > 0 ? Number(o.snapshotIntervalMs) : 1000;
  const timeoutMs = Number(o.snapshotTimeoutMs) > 0 ? Number(o.snapshotTimeoutMs) : 8000;
  const ffmpegBinary = o.ffmpegBinary || 'ffmpeg';

  let running = false;
  let frameCount = 0;
  let currentOnFrame = null;
  let lastFp = null;
  let fpChanges = 0;

  /**
   * 调用 `ffmpeg -vframes 1` 捕获单帧 JPEG
   * 每次都是独立进程 = 独立 AVFoundation session = 保证当前画面
   */
  function captureOneFrame() {
    return new Promise((resolve, reject) => {
      const args = [
        '-hide_banner', '-loglevel', 'error',
        '-f', 'avfoundation',
        '-framerate', String(o.framerate > 0 ? o.framerate : 30),
        '-pixel_format', 'uyvy422',
        '-video_size', `${width}x${height}`,
        '-i', String(deviceIndex) + ':none',
        '-vframes', '1',
        '-q:v', String(quality),
        '-f', 'mjpeg',
        '-',
      ];

      const proc = spawn(ffmpegBinary, args, { stdio: ['ignore', 'pipe', 'pipe'] });
      const chunks = [];
      let timedOut = false;

      const timer = setTimeout(() => {
        timedOut = true;
        try { proc.kill('SIGKILL'); } catch (_) {}
        reject(new Error('ffmpeg snapshot timeout'));
      }, timeoutMs);

      proc.stdout.on('data', (d) => chunks.push(d));

      proc.on('error', (e) => {
        clearTimeout(timer);
        reject(e);
      });

      proc.on('exit', (code) => {
        clearTimeout(timer);
        if (timedOut) return;
        const jpeg = Buffer.concat(chunks);
        if (jpeg.length < 100) {
          reject(new Error(`empty frame (exit ${code})`));
          return;
        }
        resolve(jpeg);
      });
    });
  }

  async function snapshotLoop() {
    while (running) {
      const t0 = Date.now();
      try {
        const jpeg = await captureOneFrame();
        if (!running) break;
        frameCount++;

        // 内容指纹：JPEG 中段 4 字节，用于确认每次快照内容不同
        const mid = jpeg.length >> 1;
        const fp = jpeg.length > mid + 4
          ? jpeg[mid].toString(16) + jpeg[mid + 1].toString(16)
            + jpeg[mid + 2].toString(16) + jpeg[mid + 3].toString(16)
          : 'short';

        if (fp !== lastFp) {
          fpChanges++;
          flowLog(STEP, '快照内容变化', { seq: frameCount, sizeBytes: jpeg.length, fp, totalChanges: fpChanges });
          lastFp = fp;
        } else {
          flowDebug(STEP, '快照内容与上次相同', { seq: frameCount, sizeBytes: jpeg.length });
        }

        currentOnFrame({ jpeg, ts: Date.now() });
      } catch (e) {
        if (running) {
          flowWarn(STEP, '快照失败，跳过', { error: e.message });
        }
      }

      // 等待剩余 interval，保证每次采帧之间至少间隔 intervalMs
      if (running) {
        const wait = Math.max(0, intervalMs - (Date.now() - t0));
        if (wait > 0) {
          await new Promise((r) => setTimeout(r, wait));
        }
      }
    }
  }

  async function start(onFrame) {
    if (running) return;
    running = true;
    frameCount = 0;
    currentOnFrame = onFrame;
    flowLog(STEP, '启动快照采帧模式', { deviceIndex, width, height, quality, intervalMs, timeoutMs });
    snapshotLoop().catch((e) => flowError(STEP, '快照循环异常', { error: e.message }));
  }

  async function stop() {
    if (!running) return;
    running = false;
    flowLog(STEP, '停止快照采帧', { totalFrames: frameCount, contentChanges: fpChanges });
  }

  return {
    name: 'ffmpeg-snapshot',
    start,
    stop,
    isRunning: () => running,
  };
}

module.exports = { createFfmpegSnapshotSource };
