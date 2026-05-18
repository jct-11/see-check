'use strict';

/**
 * 基于像素差异的快速初筛 Filter
 *
 * 逻辑：
 *   1. 当前帧与「上次落地帧」的缩略图做像素均值差（MAD，0-255）
 *   2. MAD >= diffThreshold  → 明显变化 → 落地
 *   3. MAD <  diffThreshold  → 变化不大 → 丢弃
 *   4. 距上次落地已超过 forceIntervalMs → 无论如何落地一帧（保证静止场景也有记录）
 *
 * 缩略图为 THUMB_W × THUMB_H（128×72）的灰度图，对比粗略但速度快。
 */

const { createCanvas, loadImage } = require('canvas');
const { flowDebug, flowLog } = require('../../utils/log');

const THUMB_W = 128;
const THUMB_H = 72; // 16:9，比原来 64×36 面积大4倍，对小物体变化更敏感

const STEP = '初筛';

/**
 * 将 JPEG buffer 解码并缩放为 64×36 灰度像素数组
 * @returns {Promise<Uint8ClampedArray>} RGBA 像素数据
 */
async function toThumb(jpegBuf) {
  const img = await loadImage(jpegBuf);
  const canvas = createCanvas(THUMB_W, THUMB_H);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, THUMB_W, THUMB_H);
  return ctx.getImageData(0, 0, THUMB_W, THUMB_H).data;
}

/**
 * 两幅缩略图的均值绝对差（0–255）
 */
function meanAbsDiff(a, b) {
  let sum = 0;
  const pixels = THUMB_W * THUMB_H;
  for (let i = 0; i < a.length; i += 4) {
    // 只比较 RGB，忽略 alpha
    sum += Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]);
  }
  return sum / (pixels * 3); // 归一化到 0–255
}

/**
 * @param {object} opts
 * @param {number} [opts.diffThreshold=50]     均值像素差阈值（0-255），超过则落地
 * @param {number} [opts.forceIntervalMs=10000] 超时强制落地（毫秒）
 */
function createPixelDiffFilter(opts = {}) {
  const diffThreshold = opts.diffThreshold != null ? opts.diffThreshold : 50;
  const forceIntervalMs = opts.forceIntervalMs != null ? opts.forceIntervalMs : 10000;

  let lastThumb = null;    // 上次落地帧的缩略图像素
  let lastLandedTs = 0;    // 上次落地的时间戳
  let lastJpeg = null;     // 上次落地帧的完整 JPEG buffer（全量字节比较，比采样更可靠）
  let totalChecked = 0;
  let totalLanded = 0;

  /**
   * 判断当前帧是否应该落地（async，需在 controller 中 await）
   * @param {{ jpeg: Buffer, ts: number }} frame
   * @returns {Promise<boolean>}
   */
  async function shouldKeep(frame) {
    totalChecked++;
    const now = frame.ts || Date.now();
    const elapsed = now - lastLandedTs;
    const forced = elapsed >= forceIntervalMs;

    // 全量字节比较（Buffer.equals 原生实现，48KB ≈ 5 微秒，远快于 canvas 100ms）
    // 若完全相同 → 摄像头 codec 输出 bit-identical 帧，无需解码
    const byteSame = lastJpeg !== null
      && frame.jpeg.length === lastJpeg.length
      && frame.jpeg.equals(lastJpeg);

    if (!byteSame) {
      // 每次发现字节变化都记录 INFO，让用户能确认摄像头输出了新帧
      flowLog(STEP, '字节变化帧', {
        prevLen: lastJpeg ? lastJpeg.length : 0,
        currLen: frame.jpeg.length,
        forced,
        elapsedMs: elapsed,
      });
    } else {
      flowDebug(STEP, '字节相同', { forced, elapsedMs: elapsed });
    }

    if (byteSame && !forced) {
      return false;
    }

    // forced + byteSame：摄像头仍输出相同字节（静止场景），仍需落地一帧保证时间线完整
    if (byteSame && forced) {
      totalLanded++;
      flowLog(STEP, '超时强制落地（场景静止）', {
        elapsedMs: elapsed,
        diff: '0.0',
        reason: 'force-static',
        landRate: ((totalLanded / totalChecked) * 100).toFixed(1) + '%',
      });
      lastLandedTs = now;
      return true;
    }

    let thumb;
    try {
      thumb = await toThumb(frame.jpeg);
    } catch (e) {
      flowDebug(STEP, '缩略图解码失败，丢弃', { error: e.message });
      return false;
    }

    // 第一帧 → 直接落地，初始化基准
    if (!lastThumb) {
      lastThumb = thumb;
      lastJpeg = Buffer.from(frame.jpeg);
      lastLandedTs = now;
      totalLanded++;
      flowLog(STEP, '首帧落地', { jpegLen: frame.jpeg.length });
      return true;
    }

    const diff = meanAbsDiff(thumb, lastThumb);
    const changed = diff >= diffThreshold;

    // 字节变化但 diff 很小：说明场景变化量在缩略图层面低于阈值（小物体/细微移动）
    flowLog(STEP, changed || forced ? (forced && !changed ? '超时强制落地' : '变化落地') : '字节变化但diff不足', {
      diff: diff.toFixed(1),
      diffThreshold,
      elapsedMs: elapsed,
      reason: forced && !changed ? 'force' : (changed ? 'pixel-diff' : 'below-threshold'),
      landRate: ((totalLanded / totalChecked) * 100).toFixed(1) + '%',
    });

    if (changed || forced) {
      totalLanded++;
      lastThumb = thumb;
      lastJpeg = Buffer.from(frame.jpeg);
      lastLandedTs = now;
      return true;
    }
    return false;
  }

  function reset() {
    lastThumb = null;
    lastJpeg = null;
    lastLandedTs = 0;
    totalChecked = 0;
    totalLanded = 0;
  }

  return { name: 'pixelDiff', shouldKeep, reset };
}

module.exports = { createPixelDiffFilter };
