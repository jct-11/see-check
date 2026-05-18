'use strict';

const { createMemoryVectorIndex } = require('./adapters/memory');
const { flowLog, flowWarn } = require('../utils/log');

function createVectorIndex(cfg) {
  const store = (cfg && cfg.vector && cfg.vector.store) || 'memory';
  if (store === 'lancedb') {
    return createLanceDbWithFallback(cfg);
  }
  return createMemoryVectorIndex();
}

// Creates a LanceDB index that falls back to in-memory if LanceDB is unavailable.
function createLanceDbWithFallback(cfg) {
  let impl = null;

  async function init() {
    try {
      const { createLanceDbVectorIndex } = require('./adapters/lancedb');
      const candidate = createLanceDbVectorIndex(cfg);
      await candidate.init();
      impl = candidate;
      flowLog('向量', '使用 LanceDB 持久向量索引');
    } catch (e) {
      flowWarn('向量', 'LanceDB 不可用，降级到内存向量索引', { error: e.message });
      const mem = createMemoryVectorIndex();
      await mem.init();
      impl = mem;
    }
  }

  function proxy(method) {
    return function () {
      if (!impl) throw new Error('VectorIndex 尚未初始化，请先调用 init()');
      return impl[method].apply(impl, arguments);
    };
  }

  return {
    init,
    upsert: proxy('upsert'),
    deleteMany: proxy('deleteMany'),
    filterByTime: proxy('filterByTime'),
    search: proxy('search'),
    size: proxy('size'),
    bootstrap: proxy('bootstrap'),
  };
}

module.exports = { createVectorIndex };
