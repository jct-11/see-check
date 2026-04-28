'use strict';

const { postJson } = require('../../inference/providers/ollama');
const { EMBED_MODEL } = require('../../config/defaults');

async function embedOnce({ base, text, timeoutMs }) {
  const url = base + '/api/embed';
  const { body } = await postJson(url, { model: EMBED_MODEL, input: text }, timeoutMs);
  if (body.embeddings && body.embeddings[0]) return body.embeddings[0];
  throw new Error('Ollama 返回空 embedding');
}

module.exports = { embedOnce };
