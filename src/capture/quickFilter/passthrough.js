'use strict';

function createPassthroughQuickFilter() {
  return {
    name: 'passthrough',
    shouldKeep() { return true; },
    reset() {},
  };
}

module.exports = { createPassthroughQuickFilter };
