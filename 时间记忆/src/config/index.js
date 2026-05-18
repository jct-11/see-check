'use strict';

const fs = require('fs');
const {
  DEFAULT_USER_CONFIG,
  DEFAULT_INFERENCE_SERVERS,
  DEFAULT_EMBED_BASE,
} = require('./defaults');
const { CONFIG_FILE } = require('../utils/paths');
const {
  normalizeBaseUrl,
  normalizeBasePath,
  buildInferenceBase,
} = require('../utils/net');
const { flowLog, flowWarn, flowError } = require('../utils/log');

function normalizeInferenceServer(server, fallback) {
  const input = server && typeof server === 'object' ? server : {};
  const fb = fallback && typeof fallback === 'object' ? fallback : {};
  const id = String(input.id || fb.id || '').trim() || 'server';
  const name = String(input.name || fb.name || id).trim() || id;
  const protocol = String(input.protocol || fb.protocol || 'http').toLowerCase() === 'https' ? 'https' : 'http';
  const host = String(input.host || fb.host || 'localhost').trim() || 'localhost';
  const parsedPort = parseInt(input.port, 10);
  const fbPort = parseInt(fb.port, 10);
  const port = Number.isFinite(parsedPort) && parsedPort > 0
    ? parsedPort
    : (Number.isFinite(fbPort) && fbPort > 0 ? fbPort : 80);
  const basePath = normalizeBasePath(input.basePath !== undefined ? input.basePath : fb.basePath);
  const apiStyle = String(input.apiStyle || fb.apiStyle || 'ollama').toLowerCase() === 'openai' ? 'openai' : 'ollama';
  const model = String(input.model || fb.model || '').trim();
  return { id, name, protocol, host, port, basePath, apiStyle, model };
}

function inferServersFromLegacyBases(inferenceBases) {
  if (!inferenceBases || typeof inferenceBases !== 'object') return [];
  const legacy = [];
  for (const key of Object.keys(inferenceBases)) {
    const raw = normalizeBaseUrl(inferenceBases[key]);
    if (!raw) continue;
    try {
      const u = new URL(raw);
      legacy.push({
        id: key,
        name: key === '3090' ? 'RTX 3090' : (key === 'orin' ? 'Orin' : (key === 'local' ? 'Mac Mini' : key)),
        protocol: u.protocol.replace(':', ''),
        host: u.hostname,
        port: u.port ? parseInt(u.port, 10) : (u.protocol === 'https:' ? 443 : 80),
        basePath: u.pathname === '/' ? '' : u.pathname,
        apiStyle: key === 'orin' ? 'openai' : 'ollama'
      });
    } catch (_) {}
  }
  return legacy;
}

function normalizeInferenceServers(inputServers) {
  const defaultsById = {};
  DEFAULT_INFERENCE_SERVERS.forEach((s) => { defaultsById[s.id] = s; });
  const source = Array.isArray(inputServers) && inputServers.length > 0 ? inputServers : DEFAULT_INFERENCE_SERVERS;
  const seen = new Set();
  const normalized = [];
  for (const raw of source) {
    const fallback = defaultsById[String((raw && raw.id) || '').trim()] || {};
    const server = normalizeInferenceServer(raw, fallback);
    if (!server.id || seen.has(server.id)) continue;
    seen.add(server.id);
    normalized.push(server);
  }
  if (normalized.length === 0) return DEFAULT_INFERENCE_SERVERS.map((s) => ({ ...s }));
  return normalized;
}

function normalizeCaptureConfig(raw) {
  const d = DEFAULT_USER_CONFIG.capture;
  const input = raw && typeof raw === 'object' ? raw : {};
  const ffRaw = input.ffmpeg && typeof input.ffmpeg === 'object' ? input.ffmpeg : {};
  const source = String(input.source || d.source).toLowerCase();
  const framerate = Number(ffRaw.framerate) > 0 ? Number(ffRaw.framerate) : d.ffmpeg.framerate;
  return {
    source: source === 'upload' ? 'upload' : 'ffmpeg',
    ffmpeg: {
      deviceIndex: Number.isFinite(parseInt(ffRaw.deviceIndex, 10)) ? parseInt(ffRaw.deviceIndex, 10) : d.ffmpeg.deviceIndex,
      framerate,
      width: Number(ffRaw.width) > 0 ? Number(ffRaw.width) : d.ffmpeg.width,
      height: Number(ffRaw.height) > 0 ? Number(ffRaw.height) : d.ffmpeg.height,
      quality: Number(ffRaw.quality) > 0 ? Number(ffRaw.quality) : d.ffmpeg.quality
    }
  };
}

function normalizePipelineConfig(raw) {
  const d = DEFAULT_USER_CONFIG.pipeline;
  const input = raw && typeof raw === 'object' ? raw : {};
  const batchRaw = input.batch && typeof input.batch === 'object' ? input.batch : {};
  const pdRaw = input.pixelDiff && typeof input.pixelDiff === 'object' ? input.pixelDiff : {};
  const wrRaw = input.windowRepresentative && typeof input.windowRepresentative === 'object'
    ? input.windowRepresentative : {};
  return {
    quickFilter: String(input.quickFilter || d.quickFilter),
    pixelDiff: {
      diffThreshold: Number(pdRaw.diffThreshold) >= 0
        ? Number(pdRaw.diffThreshold) : d.pixelDiff.diffThreshold,
      forceIntervalMs: Number(pdRaw.forceIntervalMs) > 0
        ? Number(pdRaw.forceIntervalMs) : d.pixelDiff.forceIntervalMs,
    },
    keyframe: String(input.keyframe || d.keyframe),
    windowRepresentative: {
      windowMs: Number(wrRaw.windowMs) > 0 ? Number(wrRaw.windowMs) : d.windowRepresentative.windowMs,
      strategy: ['first', 'middle', 'last'].indexOf(wrRaw.strategy) !== -1
        ? wrRaw.strategy : d.windowRepresentative.strategy,
    },
    batch: {
      enabled: batchRaw.enabled === true,
      maxSize: Number(batchRaw.maxSize) > 0 ? Number(batchRaw.maxSize) : d.batch.maxSize,
      windowMs: Number(batchRaw.windowMs) > 0 ? Number(batchRaw.windowMs) : d.batch.windowMs,
      concurrency: Number(batchRaw.concurrency) >= 1
        ? Math.floor(Number(batchRaw.concurrency)) : d.batch.concurrency,
    },
  };
}

function normalizeMemoryConfig(raw) {
  const input = raw && typeof raw === 'object' ? raw : {};
  const store = String(input.store || DEFAULT_USER_CONFIG.memory.store).toLowerCase();
  return { store: store === 'sqlite' ? 'sqlite' : 'jsonFs' };
}

function normalizeVectorConfig(raw) {
  const input = raw && typeof raw === 'object' ? raw : {};
  const store = String(input.store || DEFAULT_USER_CONFIG.vector.store).toLowerCase();
  return {
    store: store === 'lancedb' ? 'lancedb' : 'memory',
    dir: String(input.dir || DEFAULT_USER_CONFIG.vector.dir),
  };
}

function normalizeUserConfig(raw) {
  const input = raw && typeof raw === 'object' ? raw : {};
  const { inferenceBases: _legacyInferenceBases, dedupEnabled: _legacyDedup, ...rest } = input;
  const legacyServers = inferServersFromLegacyBases(input.inferenceBases);
  const servers = normalizeInferenceServers(Array.isArray(rest.inferenceServers) ? rest.inferenceServers : legacyServers);

  let selectedId = String(rest.selectedInferenceServerId || '').trim();
  let selected = selectedId ? servers.find((s) => s.id === selectedId) : null;
  const normalizedOllamaBase = normalizeBaseUrl(rest.ollamaBase);
  if (!selected && normalizedOllamaBase) {
    selected = servers.find((s) => buildInferenceBase(s) === normalizedOllamaBase);
  }
  if (!selected) selected = servers[0];
  selectedId = selected ? selected.id : servers[0].id;

  const timeoutSecParsed = parseInt(rest.inferenceRequestTimeoutSec, 10);
  const legacyTimeoutMsParsed = parseInt(rest.inferenceRequestTimeoutMs, 10);
  const timeoutSec = Number.isFinite(timeoutSecParsed) && timeoutSecParsed >= 5
    ? timeoutSecParsed
    : (Number.isFinite(legacyTimeoutMsParsed) && legacyTimeoutMsParsed >= 5000
      ? Math.round(legacyTimeoutMsParsed / 1000)
      : DEFAULT_USER_CONFIG.inferenceRequestTimeoutSec);

  return {
    ...DEFAULT_USER_CONFIG,
    ...rest,
    inferenceRequestTimeoutSec: timeoutSec,
    inferenceServers: servers,
    selectedInferenceServerId: selectedId,
    ollamaBase: normalizedOllamaBase || buildInferenceBase(selected),
    embedBase: normalizeBaseUrl(rest.embedBase) || DEFAULT_EMBED_BASE,
    capture: normalizeCaptureConfig(rest.capture),
    pipeline: normalizePipelineConfig(rest.pipeline),
    memory: normalizeMemoryConfig(rest.memory),
    vector: normalizeVectorConfig(rest.vector),
  };
}

function loadUserConfig() {
  let cfg = normalizeUserConfig(DEFAULT_USER_CONFIG);
  try {
    if (fs.existsSync(CONFIG_FILE)) {
      cfg = normalizeUserConfig(JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf8')));
    }
  } catch (e) {
    flowWarn('配置', '读取 config.json 失败，使用默认值', { error: e.message });
  }
  return cfg;
}

function saveUserConfig(cfg) {
  const normalized = normalizeUserConfig(cfg);
  fs.writeFileSync(CONFIG_FILE, JSON.stringify(normalized, null, 2));
  return normalized;
}

function getInferenceServerById(cfg, id) {
  const serverId = String(id || '').trim();
  if (!serverId) return null;
  return (cfg.inferenceServers || []).find((s) => s.id === serverId) || null;
}

function getInferenceServerByBase(cfg, base) {
  const normalizedBase = normalizeBaseUrl(base);
  if (!normalizedBase) return null;
  return (cfg.inferenceServers || []).find((s) => buildInferenceBase(s) === normalizedBase) || null;
}

function getActiveInferenceServer(cfg, base) {
  return (
    getInferenceServerByBase(cfg, base) ||
    getInferenceServerById(cfg, cfg.selectedInferenceServerId) ||
    (cfg.inferenceServers && cfg.inferenceServers[0]) ||
    null
  );
}

function getServerLabel(cfg, base) {
  const server = getInferenceServerByBase(cfg, base);
  return server ? server.name : 'Mac mini';
}

module.exports = {
  loadUserConfig,
  saveUserConfig,
  normalizeUserConfig,
  getInferenceServerById,
  getInferenceServerByBase,
  getActiveInferenceServer,
  getServerLabel,
};
