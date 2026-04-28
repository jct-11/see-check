'use strict';

const path = require('path');
const fs = require('fs');
const { VECTORS_DIR } = require('../../utils/paths');
const { flowLog, flowError } = require('../../utils/log');
const { EMBED_DIM } = require('../../config/defaults');

// Convert LanceDB Arrow Vector to plain JS array
function toArray(v) {
  if (!v) return [];
  if (Array.isArray(v)) return v;
  if (typeof v.toArray === 'function') return Array.from(v.toArray());
  return Array.from(v);
}

function rowToJs(row) {
  return {
    id: String(row.id),
    vector: toArray(row.vector),
    captureTime: row.capture_time || '',
    timestamp: row.timestamp || '',
    description: row.description || '',
  };
}

// Dummy row used to bootstrap an empty table with the correct schema
function makeDummyRow() {
  return {
    id: '__schema_init__',
    vector: Array(EMBED_DIM).fill(0.0),
    capture_time: '',
    timestamp: '',
    description: '',
  };
}

function createLanceDbVectorIndex(cfg) {
  const dir = (cfg && cfg.vector && cfg.vector.dir)
    ? path.resolve(cfg.vector.dir)
    : VECTORS_DIR;

  let db = null;
  let table = null;

  async function init() {
    const lancedb = require('@lancedb/lancedb');
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    db = await lancedb.connect(dir);
    const names = await db.tableNames();
    if (names.includes('memory_vectors')) {
      table = await db.openTable('memory_vectors');
    } else {
      // Create with dummy row then remove it
      table = await db.createTable('memory_vectors', [makeDummyRow()]);
      await table.delete("id = '__schema_init__'");
    }
    const cnt = await table.countRows();
    flowLog('向量', 'LanceDB 初始化完成', { dir, rows: cnt });
  }

  async function upsert(row) {
    if (!row || !row.id || !row.vector) return;
    await table.mergeInsert('id')
      .whenMatchedUpdateAll()
      .whenNotMatchedInsertAll()
      .execute([{
        id: String(row.id),
        vector: row.vector,
        capture_time: row.captureTime || '',
        timestamp: row.timestamp || '',
        description: row.description || '',
      }]);
  }

  async function deleteMany(ids) {
    if (!ids || ids.length === 0) return;
    const quoted = ids.map((id) => "'" + String(id).replace(/'/g, "''") + "'").join(', ');
    await table.delete('id IN (' + quoted + ')');
  }

  async function filterByTime({ start, end } = {}) {
    const conditions = [];
    if (start) conditions.push("capture_time >= '" + (start instanceof Date ? start.toISOString() : start) + "'");
    if (end) conditions.push("capture_time <= '" + (end instanceof Date ? end.toISOString() : end) + "'");
    let q = table.query();
    if (conditions.length > 0) q = q.where(conditions.join(' AND '));
    const rows = await q.toArray();
    return rows.map(rowToJs);
  }

  async function search({ queryVector, topK, timeRange } = {}) {
    const k = topK || 20;
    if (!queryVector) {
      const rows = await filterByTime(timeRange || {});
      return rows
        .sort((a, b) => new Date(b.timestamp || 0) - new Date(a.timestamp || 0))
        .slice(0, k);
    }
    let q = table.vectorSearch(queryVector).limit(k);
    if (timeRange) {
      const conditions = [];
      if (timeRange.start) conditions.push("capture_time >= '" + (timeRange.start instanceof Date ? timeRange.start.toISOString() : timeRange.start) + "'");
      if (timeRange.end) conditions.push("capture_time <= '" + (timeRange.end instanceof Date ? timeRange.end.toISOString() : timeRange.end) + "'");
      if (conditions.length > 0) q = q.where(conditions.join(' AND '));
    }
    const rows = await q.toArray();
    return rows.map((row) => ({
      ...rowToJs(row),
      score: row._distance != null ? Math.max(0, 1 - row._distance) : 0,
    }));
  }

  async function size() {
    return table.countRows();
  }

  async function bootstrap(memoryStore, embedFn) {
    const cnt = await table.countRows();
    flowLog('向量', 'LanceDB 向量索引就绪', { rows: cnt });
    if (cnt > 0) return;
    // Empty table: try to backfill from memoryStore (e.g. if jsonFs memories exist)
    let memories;
    try { memories = await memoryStore.list(); } catch (_) { return; }
    if (!memories || memories.length === 0) return;
    let backfilled = 0;
    for (const mem of memories) {
      if (!mem.description || !embedFn) continue;
      try {
        const v = await embedFn(mem.description);
        if (v) {
          await upsert({
            id: mem.id,
            vector: v,
            captureTime: mem.captureTime || mem.timestamp,
            timestamp: mem.timestamp,
            description: mem.description,
          });
          backfilled++;
        }
      } catch (e) {
        flowError('向量', 'LanceDB 回填失败', { id: mem.id, error: e.message });
      }
    }
    if (backfilled > 0) flowLog('向量', 'LanceDB 回填完成', { backfilled });
  }

  return { init, upsert, deleteMany, filterByTime, search, size, bootstrap };
}

module.exports = { createLanceDbVectorIndex };
