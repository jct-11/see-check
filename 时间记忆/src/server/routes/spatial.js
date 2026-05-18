'use strict';

const fs = require('fs');
const path = require('path');
const { sendJson } = require('../router');
const { DATA_DIR } = require('../../utils/paths');

function registerSpatialRoutes(router) {
  router.get('/api/scene-graph', (req, res) => {
    const sgFile = path.join(DATA_DIR, 'scene_graph', 'scene_graph.json');
    if (fs.existsSync(sgFile)) {
      res.writeHead(200, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
      fs.createReadStream(sgFile).pipe(res);
      return;
    }
    sendJson(res, 404, { error: 'scene_graph.json not found' });
  });

  router.any((p) => p === '/api/ply/default' || p === '/api/ply/point_cloud.ply', (req, res) => {
    const plyFile = path.join(DATA_DIR, 'ply', 'point_cloud.ply');
    if (fs.existsSync(plyFile)) {
      const stat = fs.statSync(plyFile);
      res.writeHead(200, {
        'Content-Type': 'application/octet-stream',
        'Content-Length': stat.size,
        'Cache-Control': 'public, max-age=3600'
      });
      fs.createReadStream(plyFile).pipe(res);
      return;
    }
    res.writeHead(404); res.end('PLY not found');
  });

  router.any((p) => p.startsWith('/data/object/'), (req, res, ctx) => {
    const fname = path.basename(ctx.url.pathname);
    if (/^\d+\.(jpg|jpeg|png)$/i.test(fname)) {
      const imgFile = path.join(DATA_DIR, 'object', fname);
      if (fs.existsSync(imgFile)) {
        res.writeHead(200, { 'Content-Type': 'image/jpeg', 'Cache-Control': 'public, max-age=86400' });
        fs.createReadStream(imgFile).pipe(res);
        return;
      }
    }
    res.writeHead(404); res.end('Not found');
  });
}

module.exports = { registerSpatialRoutes };
