'use strict';

const { flowLog } = require('../../utils/log');

/**
 * 窗口代表帧选择器
 *
 * 将批次帧按时间窗口分组，每个窗口取 1 个代表帧，降低送往 VLM 的帧数。
 *
 * strategy（可通过 config.pipeline.windowRepresentative.strategy 配置）：
 *   'first'  - 每个窗口的第一帧（时间最早）
 *   'middle' - 每个窗口的中间帧（默认）
 *   'last'   - 每个窗口的最后一帧（时间最晚，信息最新）
 */
function createWindowRepresentativeSelector(opts) {
  const windowMs = opts && Number(opts.windowMs) > 0 ? Number(opts.windowMs) : 5000;
  const strategy = (opts && opts.strategy) || 'middle';

  function select(batch) {
    if (!batch || batch.length === 0) return [];
    if (batch.length === 1) return batch;

    // 按 captureTime 升序排列
    const sorted = batch.slice().sort(function (a, b) {
      return new Date(a.captureTime || 0).getTime() - new Date(b.captureTime || 0).getTime();
    });

    const windowStart = new Date(sorted[0].captureTime || 0).getTime();

    // 按 windowMs 分桶
    var groups = {};
    for (var i = 0; i < sorted.length; i++) {
      var frame = sorted[i];
      var t = new Date(frame.captureTime || 0).getTime();
      var bucketKey = Math.floor((t - windowStart) / windowMs);
      if (!groups[bucketKey]) groups[bucketKey] = [];
      groups[bucketKey].push(frame);
    }

    var selected = [];
    var keys = Object.keys(groups).sort(function (a, b) { return Number(a) - Number(b); });
    for (var k = 0; k < keys.length; k++) {
      var group = groups[keys[k]];
      var pick;
      if (strategy === 'first') {
        pick = group[0];
      } else if (strategy === 'last') {
        pick = group[group.length - 1];
      } else {
        // middle（默认）
        pick = group[Math.floor(group.length / 2)];
      }
      selected.push(pick);
    }

    flowLog('关键帧', 'windowRepresentative 筛选', {
      batchSize: batch.length,
      selectedSize: selected.length,
      windowMs: windowMs,
      strategy: strategy,
    });

    return selected;
  }

  function reset() {}

  return { name: 'windowRepresentative', select: select, reset: reset };
}

module.exports = { createWindowRepresentativeSelector };
