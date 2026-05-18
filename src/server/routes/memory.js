'use strict';

const fs = require('fs');
const path = require('path');
const { sendJson } = require('../router');
const { FRAMES_DIR, MEMORY_DIR } = require('../../utils/paths');

// List frame image files on disk; no memory data included.
function listFrameFiles() {
  if (!fs.existsSync(FRAMES_DIR)) return [];
  return fs.readdirSync(FRAMES_DIR)
    .filter((f) => f.endsWith('.png') || f.endsWith('.jpg'))
    .sort()
    .map((filename) => {
      const framePath = path.join(FRAMES_DIR, filename);
      const stats = fs.statSync(framePath);
      const ext = path.extname(filename);
      const id = filename.replace('frame_', '').replace(ext, '');
      return { id, filename, path: framePath, size: stats.size, createdAt: stats.birthtime.toISOString(), thumbnail: '/frames/' + filename };
    });
}

// Read memory from JSON file (legacy fallback for jsonFs adapter).
function readLegacyMemory(filename) {
  const ext = path.extname(filename);
  const memFile = path.join(MEMORY_DIR, filename.replace(ext, '.json'));
  if (!fs.existsSync(memFile)) return null;
  try { return JSON.parse(fs.readFileSync(memFile, 'utf8')); } catch (_) { return null; }
}

/**
 * Build the frames list that is served to the front-end and used by the query pipeline.
 * - For SQLite store: memory data comes from the store (async).
 * - For jsonFs store (legacy): memory data comes from the JSON files on disk.
 * Always returns a Promise<Frame[]>.
 */
async function listFrames(memoryStore) {
  const files = listFrameFiles();
  if (files.length === 0) return [];

  if (memoryStore) {
    // Fetch all memories from store in one call
    const allMems = await memoryStore.list({}).catch(() => []);
    const memById = {};
    for (const m of allMems) memById[String(m.id)] = m;

    return files.map((f) => {
      const mem = memById[String(parseInt(f.id, 10))] || null;
      // Fall back to JSON file if store returned nothing (transition period)
      const legacyMem = mem ? null : readLegacyMemory(f.filename);
      const resolved = mem || legacyMem;
      return {
        id: f.id,
        filename: f.filename,
        path: f.path,
        size: f.size,
        createdAt: resolved ? (resolved.captureTime || f.createdAt) : f.createdAt,
        hasMemory: !!resolved,
        memory: resolved,
        thumbnail: f.thumbnail,
      };
    });
  }

  // No store provided (legacy path): read JSON files directly
  return files.map((f) => {
    const mem = readLegacyMemory(f.filename);
    return {
      id: f.id,
      filename: f.filename,
      path: f.path,
      size: f.size,
      createdAt: f.createdAt,
      hasMemory: !!mem,
      memory: mem,
      thumbnail: f.thumbnail,
    };
  });
}

function registerMemoryRoutes(router, ctx) {

  router.get('/api/status', (req, res) => {
    sendJson(res, 200, {
      config: ctx.getConfig(),
      isCapturing: ctx.captureController.isCapturing(),
      faceApiReady: ctx.faceService.isReady(),
      sourceName: ctx.captureController.sourceName(),
    });
  });

  router.get('/api/frames', async (req, res) => {
    try {
      const frames = await listFrames(ctx.memoryStore);
      sendJson(res, 200, frames);
    } catch (err) {
      sendJson(res, 500, { error: 'Failed to list frames', message: err.message });
    }
  });

  router.get('/api/faces', (req, res) => {
    const summary = ctx.faceLibrary.summary();
    sendJson(res, 200, { success: true, faces: summary, total: summary.length });
  });

  router.any((p) => p.startsWith('/api/memory/'), async (req, res, info) => {
    const id = info.url.pathname.split('/').pop().replace('.json', '');
    const memory = await ctx.memoryStore.get(id);
    if (memory) sendJson(res, 200, memory);
    else sendJson(res, 404, { error: 'Not found' });
  });

  router.any((p) => p.startsWith('/frames/'), (req, res, info) => {
    const fname = info.url.pathname.split('/').pop();
    const framePath = path.join(FRAMES_DIR, fname);
    if (fs.existsSync(framePath)) {
      const ct = fname.endsWith('.jpg') ? 'image/jpeg' : 'image/png';
      res.writeHead(200, {
        'Content-Type': ct,
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
      });
      fs.createReadStream(framePath).pipe(res);
      return;
    }
    res.writeHead(404); res.end('Not found');
  });
}

module.exports = { registerMemoryRoutes, listFrames };
