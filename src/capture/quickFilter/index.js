'use strict';

const { createPassthroughQuickFilter } = require('./passthrough');
const { createPixelDiffFilter } = require('./pixelDiff');

function createQuickFilter(cfg) {
  const name = (cfg && cfg.pipeline && cfg.pipeline.quickFilter) || 'pixelDiff';
  const filterCfg = (cfg && cfg.pipeline && cfg.pipeline.pixelDiff) || {};
  switch (name) {
    case 'passthrough':
      return createPassthroughQuickFilter();
    case 'pixelDiff':
      return createPixelDiffFilter({
        diffThreshold: filterCfg.diffThreshold,
        forceIntervalMs: filterCfg.forceIntervalMs,
      });
    default:
      throw new Error('未知的 quickFilter: ' + name);
  }
}

module.exports = { createQuickFilter };
