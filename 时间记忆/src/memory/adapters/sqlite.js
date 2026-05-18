'use strict';

const fs = require('fs');
const { SQLITE_FILE, DATA_DIR, ARCHIVE_DIR } = require('../../utils/paths');
const { flowLog, flowError } = require('../../utils/log');

const SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS memories (
  id              INTEGER PRIMARY KEY,
  frame_path      TEXT NOT NULL DEFAULT '',
  description     TEXT NOT NULL DEFAULT '',
  model           TEXT,
  server_label    TEXT,
  capture_time    TEXT NOT NULL,
  timestamp       TEXT NOT NULL,
  inference_time  INTEGER,
  faces_json      TEXT,
  created_at      TEXT DEFAULT (datetime('now')),
  updated_at      TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_memories_capture_time ON memories(capture_time);

CREATE TABLE IF NOT EXISTS faces (
  person_id   TEXT PRIMARY KEY,
  name        TEXT,
  gender      TEXT,
  age         INTEGER,
  descriptor  BLOB,
  first_seen  TEXT,
  last_seen   TEXT,
  count       INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS archived_memories (
  id              INTEGER PRIMARY KEY,
  frame_path      TEXT,
  description     TEXT,
  model           TEXT,
  server_label    TEXT,
  capture_time    TEXT,
  timestamp       TEXT,
  inference_time  INTEGER,
  faces_json      TEXT,
  archived_at     TEXT DEFAULT (datetime('now'))
);
`;

function rowToMem(row) {
  if (!row) return null;
  return {
    id: row.id,
    framePath: row.frame_path || '',
    description: row.description || '',
    model: row.model || '',
    serverLabel: row.server_label || '',
    captureTime: row.capture_time || '',
    timestamp: row.timestamp || '',
    inferenceTime: row.inference_time || 0,
    faces: row.faces_json ? tryParse(row.faces_json, []) : [],
  };
}

function tryParse(str, fallback) {
  try { return JSON.parse(str); } catch (_) { return fallback; }
}

function createSqliteMemoryStore() {
  let db = null;

  async function init() {
    if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
    const Database = require('better-sqlite3');
    db = new Database(SQLITE_FILE);
    db.pragma('journal_mode = WAL');
    db.pragma('busy_timeout = 10000'); // 遇到写锁时最多等 10 秒自动重试，而不是立即报 SQLITE_LOCKED
    db.exec(SCHEMA_SQL);
    flowLog('存储', 'SQLite 初始化完成', { file: SQLITE_FILE });
  }

  async function save(mem) {
    const stmt = db.prepare(`
      INSERT INTO memories
        (id, frame_path, description, model, server_label, capture_time, timestamp, inference_time, faces_json, updated_at)
      VALUES
        (@id, @frame_path, @description, @model, @server_label, @capture_time, @timestamp, @inference_time, @faces_json, datetime('now'))
      ON CONFLICT(id) DO UPDATE SET
        frame_path     = excluded.frame_path,
        description    = excluded.description,
        model          = excluded.model,
        server_label   = excluded.server_label,
        capture_time   = excluded.capture_time,
        timestamp      = excluded.timestamp,
        inference_time = excluded.inference_time,
        faces_json     = excluded.faces_json,
        updated_at     = excluded.updated_at
    `);
    stmt.run({
      id: mem.id,
      frame_path: mem.framePath || '',
      description: mem.description || '',
      model: mem.model || null,
      server_label: mem.serverLabel || null,
      capture_time: mem.captureTime || mem.timestamp || new Date().toISOString(),
      timestamp: mem.timestamp || new Date().toISOString(),
      inference_time: mem.inferenceTime || null,
      faces_json: mem.faces && mem.faces.length > 0 ? JSON.stringify(mem.faces) : null,
    });
    flowLog('存储', '记忆已保存(SQLite)', { id: mem.id, faces: mem.faces ? mem.faces.length : 0 });
    return mem;
  }

  async function get(id) {
    const row = db.prepare('SELECT * FROM memories WHERE id = ?').get(id);
    return rowToMem(row);
  }

  async function list({ since, until, ids } = {}) {
    let sql = 'SELECT * FROM memories';
    const conditions = [];
    const params = [];
    if (since) {
      conditions.push('capture_time >= ?');
      params.push(since instanceof Date ? since.toISOString() : String(since));
    }
    if (until) {
      conditions.push('capture_time <= ?');
      params.push(until instanceof Date ? until.toISOString() : String(until));
    }
    if (ids && ids.length > 0) {
      conditions.push('id IN (' + ids.map(() => '?').join(',') + ')');
      params.push(...ids);
    }
    if (conditions.length > 0) sql += ' WHERE ' + conditions.join(' AND ');
    sql += ' ORDER BY capture_time ASC';
    return db.prepare(sql).all(...params).map(rowToMem);
  }

  async function deleteMany(ids) {
    if (!ids || ids.length === 0) return;
    const ph = ids.map(() => '?').join(',');
    db.prepare('DELETE FROM memories WHERE id IN (' + ph + ')').run(...ids);
  }

  async function archive(ids) {
    if (!ids || ids.length === 0) return;
    if (!fs.existsSync(ARCHIVE_DIR)) fs.mkdirSync(ARCHIVE_DIR, { recursive: true });
    const ph = ids.map(() => '?').join(',');
    const rows = db.prepare('SELECT * FROM memories WHERE id IN (' + ph + ')').all(...ids);
    const ins = db.prepare(`
      INSERT OR REPLACE INTO archived_memories
        (id, frame_path, description, model, server_label, capture_time, timestamp, inference_time, faces_json)
      VALUES
        (@id, @frame_path, @description, @model, @server_label, @capture_time, @timestamp, @inference_time, @faces_json)
    `);
    const tx = db.transaction((rows) => {
      for (const row of rows) ins.run(row);
    });
    tx(rows);
    db.prepare('DELETE FROM memories WHERE id IN (' + ph + ')').run(...ids);
    flowLog('存储', '归档完成(SQLite)', { count: rows.length });
  }

  function upsertFace(personId, faceData) {
    const stmt = db.prepare(`
      INSERT INTO faces (person_id, name, gender, age, descriptor, first_seen, last_seen, count)
      VALUES (@person_id, @name, @gender, @age, @descriptor, @first_seen, @last_seen, @count)
      ON CONFLICT(person_id) DO UPDATE SET
        name       = excluded.name,
        gender     = excluded.gender,
        age        = excluded.age,
        descriptor = excluded.descriptor,
        last_seen  = excluded.last_seen,
        count      = excluded.count
    `);
    stmt.run({
      person_id:  personId,
      name:       faceData.name || '未命名',
      gender:     faceData.gender || null,
      age:        faceData.age || null,
      descriptor: faceData.descriptor ? JSON.stringify(faceData.descriptor) : null,
      first_seen: faceData.firstSeen || new Date().toISOString(),
      last_seen:  faceData.lastSeen  || new Date().toISOString(),
      count:      faceData.count || 1,
    });
  }

  function listFaces() {
    return db.prepare('SELECT * FROM faces ORDER BY count DESC').all().map((row) => ({
      personId:   row.person_id,
      name:       row.name,
      gender:     row.gender,
      age:        row.age,
      descriptor: row.descriptor ? tryParse(row.descriptor, null) : null,
      firstSeen:  row.first_seen,
      lastSeen:   row.last_seen,
      count:      row.count,
    }));
  }

  return { init, save, get, list, deleteMany, archive, upsertFace, listFaces };
}

module.exports = { createSqliteMemoryStore };
