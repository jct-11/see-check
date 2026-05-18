'use strict';

function createPassthroughKeyframeSelector() {
  return {
    name: 'passthrough',
    select(landed) { return landed; },
    reset() {},
  };
}

module.exports = { createPassthroughKeyframeSelector };
