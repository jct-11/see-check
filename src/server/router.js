'use strict';

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk) => { body += chunk; });
    req.on('end', () => resolve(body));
    req.on('error', reject);
  });
}

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(payload));
}

function sendError(res, statusCode, error) {
  if (!res.headersSent) res.writeHead(statusCode, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ success: false, error: error.message || String(error) }));
}

function createRouter() {
  const handlers = [];

  function add(method, matcher, handler) {
    handlers.push({ method, matcher, handler });
  }

  function get(matcher, handler) { add('GET', matcher, handler); }
  function post(matcher, handler) { add('POST', matcher, handler); }
  function any(matcher, handler) { add(null, matcher, handler); }

  async function handle(req, res, ctx) {
    const url = new URL(req.url, 'http://localhost');
    res.setHeader('Access-Control-Allow-Origin', '*');
    for (const h of handlers) {
      if (h.method && h.method !== req.method) continue;
      let match = false;
      let params = {};
      if (typeof h.matcher === 'string') {
        match = h.matcher === url.pathname;
      } else if (h.matcher instanceof RegExp) {
        const m = url.pathname.match(h.matcher);
        if (m) { match = true; params = m.groups || {}; }
      } else if (typeof h.matcher === 'function') {
        const r = h.matcher(url.pathname, req);
        if (r) { match = true; params = r === true ? {} : r; }
      }
      if (!match) continue;
      try {
        await h.handler(req, res, { ...ctx, url, params });
      } catch (err) {
        sendError(res, 500, err);
      }
      return;
    }
    sendError(res, 404, new Error('Not Found'));
  }

  return { add, get, post, any, handle };
}

module.exports = { createRouter, readBody, sendJson, sendError };
