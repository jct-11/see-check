'use strict';

const { flowLog, flowWarn } = require('../utils/log');

// 最大待处理队列深度，超出时丢弃最旧帧（防止极端情况下内存无限增长）
const MAX_QUEUE = 50;

function createBatcher(cfg, onFlush) {
  const batchCfg = (cfg && cfg.pipeline && cfg.pipeline.batch) || { enabled: false };
  const enabled = batchCfg.enabled === true;
  const maxSize = Number(batchCfg.maxSize) > 0 ? Number(batchCfg.maxSize) : 4;
  const windowMs = Number(batchCfg.windowMs) > 0 ? Number(batchCfg.windowMs) : 2000;

  let queue = [];
  let timer = null;
  let processing = false; // 串行化：同一时刻只允许一个 onFlush 在跑

  function clearTimer() {
    if (timer) { clearTimeout(timer); timer = null; }
  }

  function tryFlush() {
    if (processing || queue.length === 0) return;
    clearTimer();
    // 取出本次要处理的帧
    // batch 模式：按 maxSize 切片
    // 非 batch 模式：最多取 6 帧（匹配 framePipeline 的 BATCH_MAX），剩余留给下次
    const chunkSize = enabled ? Math.min(maxSize, queue.length) : Math.min(6, queue.length);
    const batch = queue.splice(0, chunkSize);
    processing = true;
    flowLog('批量', '开始处理', { batchSize: batch.length, queueRemaining: queue.length });
    Promise.resolve()
      .then(() => onFlush(batch))
      .catch((e) => { flowLog('批量', 'flush 异常', { error: e.message }); })
      .finally(() => {
        processing = false;
        if (queue.length > 0) {
          flowLog('批量', '积压队列继续处理', { pending: queue.length });
          tryFlush(); // 串行处理剩余积压
        }
      });
  }

  function push(landed) {
    if (queue.length >= MAX_QUEUE) {
      // 丢弃最旧的帧，给新帧腾位（最新画面优先于陈旧帧）
      const dropped = queue.shift();
      flowWarn('批量', '队列满，丢弃最旧帧', { droppedId: dropped && dropped.id, queueMax: MAX_QUEUE });
    }
    queue.push(landed);
    if (enabled) {
      if (queue.length >= maxSize) { tryFlush(); return; }
      if (!timer) timer = setTimeout(tryFlush, windowMs);
    } else {
      tryFlush(); // 非 batch 模式：有空就立即处理，忙则等待
    }
  }

  return { push, flushNow: tryFlush, settings: { enabled, maxSize, windowMs } };
}

module.exports = { createBatcher };
