'use strict';

const { embedOnce } = require('./providers/nomicOllama');
const { flowLog, flowWarn, flowError } = require('../utils/log');

function createEmbeddingService(getConfig) {
  async function embed(text, retries) {
    const cfg = getConfig();
    const tries = Number.isFinite(retries) ? retries : 3;
    for (let i = 0; i < tries; i++) {
      try {
        const vec = await embedOnce({ base: cfg.embedBase, text, timeoutMs: 30000 });
        return vec;
      } catch (e) {
        if (i < tries - 1) {
          flowWarn('嵌入', `第 ${i + 1}/${tries} 次失败，将重试`, { error: e.message, base: cfg.embedBase });
          await new Promise((r) => setTimeout(r, (i + 1) * 2000));
        } else {
          flowError('嵌入', `全部 ${tries} 次重试均失败`, { error: e.message, base: cfg.embedBase });
        }
      }
    }
    return null;
  }

  return { embed };
}

module.exports = { createEmbeddingService };
