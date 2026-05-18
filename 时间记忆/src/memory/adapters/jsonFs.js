'use strict';

const fs = require('fs');
const path = require('path');
const { MEMORY_DIR, FRAMES_DIR, ARCHIVE_DIR } = require('../../utils/paths');
const { flowLog } = require('../../utils/log');

function ensureDir(dir) { if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true }); }

function memoryFilePath(id) {
  return path.join(MEMORY_DIR, 'frame_' + String(id).padStart(3, '0') + '.json');
}

function frameImagePathById(id) {
  const padded = String(id).padStart(3, '0');
  const jpg = path.join(FRAMES_DIR, 'frame_' + padded + '.jpg');
  const png = path.join(FRAMES_DIR, 'frame_' + padded + '.png');
  if (fs.existsSync(jpg)) return jpg;
  if (fs.existsSync(png)) return png;
  return null;
}

function createJsonFsMemoryStore() {
  async function init() { ensureDir(MEMORY_DIR); }

  async function save(mem) {
    ensureDir(MEMORY_DIR);
    fs.writeFileSync(memoryFilePath(mem.id), JSON.stringify(mem, null, 2));
    flowLog('存储', '记忆已保存', { id: mem.id, faces: mem.faces ? mem.faces.length : 0 });
    return mem;
  }

  async function get(id) {
    const fp = memoryFilePath(id);
    if (!fs.existsSync(fp)) return null;
    try { return JSON.parse(fs.readFileSync(fp, 'utf8')); }
    catch (_) { return null; }
  }

  async function list({ since, until, ids } = {}) {
    if (!fs.existsSync(MEMORY_DIR)) return [];
    let files = fs.readdirSync(MEMORY_DIR).filter((f) => f.endsWith('.json')).sort();
    const out = [];
    for (const f of files) {
      try {
        const mem = JSON.parse(fs.readFileSync(path.join(MEMORY_DIR, f), 'utf8'));
        if (ids && Array.isArray(ids) && !ids.includes(mem.id)) continue;
        if (since || until) {
          const t = new Date(mem.captureTime || mem.timestamp);
          if (since && t < since) continue;
          if (until && t > until) continue;
        }
        out.push(mem);
      } catch (_) {}
    }
    return out;
  }

  async function deleteMany(ids) {
    for (const id of ids) {
      const fp = memoryFilePath(id);
      try { if (fs.existsSync(fp)) fs.unlinkSync(fp); } catch (_) {}
      const img = frameImagePathById(id);
      if (img) { try { fs.unlinkSync(img); } catch (_) {} }
    }
  }

  async function archive(ids) {
    ensureDir(ARCHIVE_DIR);
    for (const id of ids) {
      const fp = memoryFilePath(id);
      if (!fs.existsSync(fp)) continue;
      try {
        const mem = JSON.parse(fs.readFileSync(fp, 'utf8'));
        delete mem.embedding;
        const af = path.join(ARCHIVE_DIR, 'frame_' + String(id).padStart(3, '0') + '.json');
        fs.writeFileSync(af, JSON.stringify(mem, null, 2));
        fs.unlinkSync(fp);
      } catch (_) {}
    }
  }

  return { init, save, get, list, deleteMany, archive };
}

module.exports = { createJsonFsMemoryStore };
