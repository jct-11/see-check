'use strict';

const { buildInferenceBase } = require('../utils/net');

const DEFAULT_PROMPT = '请描述这张图片的具体内容，重点关注：人物（人数、外貌特征、穿着、具体动作）、物品（型号、颜色、位置）、场景变化（与空旷状态的不同之处）。忽略不变的背景描述，只描述画面中有特征的元素。如果画面为空场景，仅回复"空场景，无人无特殊物品"。';

const DEFAULT_INFERENCE_SERVERS = [
  { id: '3090', name: 'RTX 3090', protocol: 'http', host: '192.168.0.200', port: 11434, basePath: '', apiStyle: 'ollama' },
  { id: 'orin', name: 'Orin', protocol: 'http', host: '192.168.1.123', port: 8080, basePath: '/v1', apiStyle: 'openai', model: 'gemma-4-E2B-it-UD-Q8_K_XL.gguf' },
  { id: 'local', name: 'Mac Mini', protocol: 'http', host: 'localhost', port: 11434, basePath: '', apiStyle: 'ollama' }
];

const EMBED_MODEL = 'nomic-embed-text';
const EMBED_DIM = 768;
const DEFAULT_EMBED_BASE = 'http://192.168.0.200:11434';

const DEFAULT_USER_CONFIG = {
  fps: 30,
  prompt: DEFAULT_PROMPT,
  model: 'gemma4:e2b',
  inferenceRequestTimeoutSec: 120,
  inferenceServers: DEFAULT_INFERENCE_SERVERS.map((s) => ({ ...s })),
  selectedInferenceServerId: '3090',
  ollamaBase: buildInferenceBase(DEFAULT_INFERENCE_SERVERS[0]),
  embedBase: DEFAULT_EMBED_BASE,
  capture: {
    source: 'ffmpeg',
    // avfoundation 摄像头常见支持：30 / 25 / 15 / 10 fps，不支持 20fps
    ffmpeg: { deviceIndex: 0, framerate: 30, width: 1280, height: 720, quality: 5 }
  },
  pipeline: {
    quickFilter: 'pixelDiff',
    pixelDiff: {
      diffThreshold: 50,      // 均值像素差阈值（0-255），超过则落地
      forceIntervalMs: 10000, // 超时强制落地（毫秒）
    },
    keyframe: 'passthrough',
    windowRepresentative: {
      windowMs: 5000,         // 每个时间窗口的长度（毫秒）
      strategy: 'middle',     // 代表帧策略：first | middle | last
    },
    batch: { enabled: false, maxSize: 4, windowMs: 2000, concurrency: 1 }
  },
  memory: { store: 'sqlite' },
  vector: { store: 'lancedb', dir: 'data/vectors' }
};

module.exports = {
  DEFAULT_PROMPT,
  DEFAULT_INFERENCE_SERVERS,
  DEFAULT_USER_CONFIG,
  DEFAULT_EMBED_BASE,
  EMBED_MODEL,
  EMBED_DIM,
};
