'use strict';

const http = require('http');

const BATCH_SERVICE_HOST = process.env.BATCH_SERVICE_HOST || '127.0.0.1';
const BATCH_SERVICE_PORT = parseInt(process.env.BATCH_SERVICE_PORT, 10) || 8000;

function registerBatchRoutes(router) {
  router.any((p) => p.startsWith('/batch/'), (req, res, ctx) => {
    const urlPath = ctx.url.pathname;
    const queryParams = ctx.url.search || '';
    const method = req.method || 'GET';

    const options = {
      hostname: BATCH_SERVICE_HOST,
      port: BATCH_SERVICE_PORT,
      path: urlPath + queryParams,
      method: method,
      headers: { ...req.headers },
      timeout: 300000,
    };

    const proxyReq = http.request(options, (proxyRes) => {
      const resHeaders = {
        'Content-Type': proxyRes.headers['content-type'] || 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, PATCH, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
      };
      if (proxyRes.headers['x-inference-time']) {
        resHeaders['X-Inference-Time'] = proxyRes.headers['x-inference-time'];
      }
      res.writeHead(proxyRes.statusCode, resHeaders);
      proxyRes.pipe(res);
    });

    proxyReq.on('error', (e) => {
      console.error('[Batch Proxy] Error:', e.message);
      if (!res.headersSent) {
        res.writeHead(502, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
      }
      res.end(JSON.stringify({ success: false, error: 'Batch service unavailable: ' + e.message }));
    });

    proxyReq.on('timeout', () => {
      proxyReq.destroy();
      if (!res.headersSent) {
        res.writeHead(504, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
      }
      res.end(JSON.stringify({ success: false, error: 'Batch service timeout' }));
    });

    // 原始数据流式透传（适用于任何 Content-Type）
    req.pipe(proxyReq);
  });
}

module.exports = { registerBatchRoutes };
