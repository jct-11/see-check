'use strict';

const fs = require('fs');
const path = require('path');

const WEB_DIR = path.join(__dirname, '../../web');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.js':   'application/javascript; charset=utf-8',
  '.svg':  'image/svg+xml',
  '.png':  'image/png',
  '.ico':  'image/x-icon',
};

function serveFile(res, filePath, cacheControl) {
  if (!fs.existsSync(filePath)) {
    res.writeHead(404); res.end('Not found'); return;
  }
  const ext = path.extname(filePath).toLowerCase();
  const mime = MIME[ext] || 'application/octet-stream';
  res.writeHead(200, {
    'Content-Type': mime,
    'Cache-Control': cacheControl || 'no-store',
  });
  fs.createReadStream(filePath).pipe(res);
}

function registerWebRoutes(router) {
  router.get('/', (req, res) => {
    serveFile(res, path.join(WEB_DIR, 'index.html'), 'no-store, no-cache, must-revalidate');
  });

  router.get('/index.html', (req, res) => {
    serveFile(res, path.join(WEB_DIR, 'index.html'), 'no-store');
  });

  router.get('/favicon.svg', (req, res) => {
    serveFile(res, path.join(WEB_DIR, 'favicon.svg'), 'public, max-age=3600');
  });

  router.get('/favicon.ico', (req, res) => {
    serveFile(res, path.join(WEB_DIR, 'favicon.ico'), 'public, max-age=3600');
  });

  router.get(
    (p) => p.startsWith('/web/assets/') || p.startsWith('/assets/'),
    (req, res, ctx) => {
      const rel = ctx.url.pathname.replace(/^\/(?:web\/)?assets\//, '');
      const filePath = path.join(WEB_DIR, 'assets', rel);
      serveFile(res, filePath, 'public, max-age=60');
    }
  );
}

module.exports = { registerWebRoutes };
