'use strict';

const { networkInterfaces } = require('os');

function getLocalIP() {
  const nets = networkInterfaces();
  for (const name of Object.keys(nets)) {
    for (const net of nets[name]) {
      if (net.family === 'IPv4' && !net.internal) return net.address;
    }
  }
  return 'localhost';
}

function normalizeBaseUrl(base) {
  return String(base || '').trim().replace(/\/$/, '');
}

function normalizeBasePath(basePath) {
  const p = String(basePath || '').trim();
  if (!p) return '';
  const withSlash = p.startsWith('/') ? p : '/' + p;
  return withSlash.replace(/\/$/, '');
}

function buildInferenceBase(server) {
  if (!server) return '';
  const protocol = String(server.protocol || 'http').toLowerCase() === 'https' ? 'https' : 'http';
  const host = String(server.host || '').trim();
  const port = Number(server.port) > 0 ? Number(server.port) : 80;
  const basePath = normalizeBasePath(server.basePath);
  if (!host) return '';
  return normalizeBaseUrl(protocol + '://' + host + ':' + port + basePath);
}

module.exports = {
  getLocalIP,
  normalizeBaseUrl,
  normalizeBasePath,
  buildInferenceBase,
};
