'use strict';

const { createPassthroughKeyframeSelector } = require('./passthrough');
const { createWindowRepresentativeSelector } = require('./windowRepresentative');

function createKeyframeSelector(cfg) {
  const name = (cfg && cfg.pipeline && cfg.pipeline.keyframe) || 'passthrough';
  switch (name) {
    case 'passthrough':
      return createPassthroughKeyframeSelector();
    case 'windowRepresentative':
      return createWindowRepresentativeSelector(
        (cfg && cfg.pipeline && cfg.pipeline.windowRepresentative) || {}
      );
    default:
      throw new Error('未知的 keyframe selector: ' + name);
  }
}

module.exports = { createKeyframeSelector };
