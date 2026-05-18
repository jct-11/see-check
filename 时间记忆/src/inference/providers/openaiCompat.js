'use strict';

const { postJson, parseBatchResponse } = require('./ollama');

const DEFAULT_MODEL = 'gemma-4-E2B-it-UD-Q8_K_XL.gguf';

async function visionInfer({ base, model, prompt, imageBase64, timeoutMs }) {
  const started = Date.now();
  const url = base + '/chat/completions';
  const useModel = model || DEFAULT_MODEL;
  const payload = {
    model: useModel,
    messages: [{
      role: 'user',
      content: [
        { type: 'text', text: prompt },
        { type: 'image_url', image_url: { url: 'data:image/jpeg;base64,' + imageBase64 } }
      ]
    }],
    max_tokens: 1024,
    stream: false,
  };
  const { body } = await postJson(url, payload, timeoutMs);
  const elapsedMs = Date.now() - started;
  const description = (body.choices && body.choices[0] && body.choices[0].message)
    ? body.choices[0].message.content
    : '无结果';
  return { description, elapsedMs, model: useModel, raw: body };
}

async function visionBatchInfer({ base, model, prompt, imagesBase64, timeoutMs }) {
  const started = Date.now();
  const url = base + '/chat/completions';
  const useModel = model || DEFAULT_MODEL;
  const n = imagesBase64.length;
  const markers = Array.from({ length: n }, (_, i) => '[图' + (i + 1) + '] (第' + (i + 1) + '张图的描述)').join('\n');
  const batchPrompt = '你将看到 ' + n + ' 张按时间顺序排列的图片。请依次描述每张图，格式严格如下（不要省略标记）：\n' +
    markers + '\n\n每张图只描述其独特内容。' + prompt;
  const content = [
    { type: 'text', text: batchPrompt },
    ...imagesBase64.map((b64) => ({
      type: 'image_url',
      image_url: { url: 'data:image/jpeg;base64,' + b64 },
    })),
  ];
  const payload = {
    model: useModel,
    messages: [{ role: 'user', content }],
    max_tokens: 512 * n, // 每帧约 512 token
    stream: false,
  };
  const { body } = await postJson(url, payload, timeoutMs);
  const elapsedMs = Date.now() - started;
  const rawText = (body.choices && body.choices[0] && body.choices[0].message)
    ? body.choices[0].message.content : '';
  const descriptions = parseBatchResponse(rawText, n);
  return { descriptions, elapsedMs, model: useModel, raw: body };
}

async function textInfer({ base, model, prompt, timeoutMs }) {
  const started = Date.now();
  const url = base + '/chat/completions';
  const useModel = model || DEFAULT_MODEL;
  const payload = {
    model: useModel,
    messages: [{ role: 'user', content: prompt }],
    max_tokens: 1024,
    stream: false,
  };
  const { body } = await postJson(url, payload, timeoutMs);
  const elapsedMs = Date.now() - started;
  const answer = (body.choices && body.choices[0] && body.choices[0].message)
    ? body.choices[0].message.content
    : '无结果';
  return { answer, elapsedMs, raw: body };
}

module.exports = { visionInfer, visionBatchInfer, textInfer };
