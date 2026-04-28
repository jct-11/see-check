'use strict';

const { flowLog } = require('../../utils/log');
const { EMBED_DIM } = require('../../config/defaults');

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

function createMemoryVectorIndex() {
  let rows = [];

  async function init() { rows = []; }

  async function upsert(row) {
    if (!row || !row.id || !row.vector) return;
    rows = rows.filter((r) => r.id !== row.id);
    rows.push({
      id: row.id,
      vector: row.vector,
      captureTime: row.captureTime,
      timestamp: row.timestamp,
      description: row.description,
    });
  }

  async function deleteMany(ids) {
    if (!ids || ids.length === 0) return;
    const set = new Set(ids);
    rows = rows.filter((r) => !set.has(r.id));
  }

  async function filterByTime({ start, end } = {}) {
    if (!start && !end) return rows.slice();
    return rows.filter((r) => {
      const t = new Date(r.captureTime || r.timestamp);
      if (start && t < start) return false;
      if (end && t > end) return false;
      return true;
    });
  }

  async function search({ queryVector, topK, timeRange }) {
    let candidates = rows;
    if (timeRange) candidates = await filterByTime(timeRange);
    if (!queryVector) {
      return candidates
        .slice()
        .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
        .slice(0, topK || 20);
    }
    return candidates
      .map((r) => ({ ...r, score: cosine(queryVector, r.vector) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK || 20);
  }

  function size() { return rows.length; }

  async function bootstrap(memoryStore, embedFn) {
    const memories = await memoryStore.list();
    let backfilled = 0;
    let migrated = 0;
    for (const mem of memories) {
      const captureTime = mem.captureTime || mem.timestamp;
      const needsRegen = mem.embedding && mem.embedding.length !== EMBED_DIM;
      if (mem.embedding && !needsRegen) {
        await upsert({
          id: mem.id,
          vector: mem.embedding,
          captureTime,
          timestamp: mem.timestamp,
          description: mem.description,
        });
      } else if (mem.description && embedFn) {
        if (needsRegen) migrated++;
        try {
          const v = await embedFn(mem.description);
          if (v) {
            mem.embedding = v;
            if (!mem.captureTime) mem.captureTime = captureTime;
            await memoryStore.save(mem);
            await upsert({
              id: mem.id,
              vector: v,
              captureTime,
              timestamp: mem.timestamp,
              description: mem.description,
            });
            backfilled++;
          }
        } catch (e) {
          flowLog('索引', '回填失败', { id: mem.id, error: e.message });
        }
      }
    }
    flowLog('索引', '向量索引已构建', {
      total: rows.length, backfilled, migrated, dims: EMBED_DIM,
    });
  }

  return { init, upsert, deleteMany, filterByTime, search, size, bootstrap };
}

module.exports = { createMemoryVectorIndex };
