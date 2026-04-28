'use strict';

const { createJsonFsMemoryStore } = require('./adapters/jsonFs');

function createMemoryStore(cfg) {
  const store = (cfg && cfg.memory && cfg.memory.store) || 'jsonFs';
  if (store === 'sqlite') {
    const { createSqliteMemoryStore } = require('./adapters/sqlite');
    return createSqliteMemoryStore();
  }
  return createJsonFsMemoryStore();
}

module.exports = { createMemoryStore };
