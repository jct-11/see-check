'use strict';

const http = require('http');
const https = require('https');
const { URL } = require('url');

function chooseHttp(url) {
  return url.startsWith('https://') ? https : http;
}

function postJson(url, payload, timeoutMs) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(payload);
    const u = new URL(url);
    const lib = chooseHttp(url);
    const req = lib.request({
      protocol: u.protocol,
      hostname: u.hostname,
      port: u.port || (u.protocol === 'https:' ? 443 : 80),
      path: u.pathname + (u.search || ''),
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
      timeout: timeoutMs,
    }, (res) => {
      let data = '';
      res.on('data', (c) => { data += c; });
      res.on('end', () => {
        try { resolve({ statusCode: res.statusCode, body: JSON.parse(data) }); }
        catch (e) { reject(new Error('解析失败：' + e.message + ' - ' + data.slice(0, 200))); }
      });
    });
    req.on('error', reject);
    req.on('timeout', () => { req.destroy(); reject(new Error('超时')); });
    req.write(body);
    req.end();
  });
}

/**
 * 将多图批量响应按 [图N] 标记拆分为每帧独立描述。
 * 支持多种容错格式：[图1] / 图1： / **图1** / 1. 等
 */
function parseBatchResponse(text, count) {
  const results = [];
  for (let i = 1; i <= count; i++) {
    const next = i + 1;
    // 主格式 [图N]，向后匹配到 [图N+1] 或字符串末尾
    const primary = new RegExp(
      '\\[图' + i + '\\]\\s*([\\s\\S]*?)(?=\\[图' + next + '\\]|$)'
    );
    let m = text.match(primary);
    if (m && m[1].trim()) { results.push(m[1].trim()); continue; }

    // 备用格式：图N：/ 图N: / **图N**
    const fallback = new RegExp(
      '(?:图' + i + '[：:]|\\*\\*图' + i + '\\*\\*)\\s*([\\s\\S]*?)(?=(?:图' + next + '[：:]|\\*\\*图' + next + '\\*\\*)|$)'
    );
    m = text.match(fallback);
    if (m && m[1].trim()) { results.push(m[1].trim()); continue; }

    results.push(null); // 标记为待兜底
  }

  // 兜底：若全部解析失败，按空行分段分配
  const allNull = results.every((r) => r === null);
  if (allNull) {
    const paras = text.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
    for (let i = 0; i < count; i++) {
      results[i] = paras[i] || text; // 无分段则全文兜底
    }
  } else {
    // 个别 null：用完整文本兜底（宁可重复也不丢帧）
    for (let i = 0; i < count; i++) {
      if (!results[i]) results[i] = text;
    }
  }
  return results;
}

async function visionInfer({ base, model, prompt, imageBase64, timeoutMs }) {
  const started = Date.now();
  const url = base + '/api/generate';
  const payload = { model, prompt, images: [imageBase64], stream: false };
  const { body } = await postJson(url, payload, timeoutMs);
  const elapsedMs = Date.now() - started;
  const description = body.response || '无结果';
  return { description, elapsedMs, model, raw: body };
}

async function visionBatchInfer({ base, model, prompt, imagesBase64, timeoutMs }) {
  const started = Date.now();
  const n = imagesBase64.length;
  const url = base + '/api/generate';
  // 构造带编号格式要求的 prompt
  const markers = Array.from({ length: n }, (_, i) => '[图' + (i + 1) + '] (第' + (i + 1) + '张图的描述)').join('\n');
  const batchPrompt = '你将看到 ' + n + ' 张按时间顺序排列的图片。请依次描述每张图，格式严格如下（不要省略标记）：\n' +
    markers + '\n\n每张图只描述其独特内容。' + prompt;
  const payload = { model, prompt: batchPrompt, images: imagesBase64, stream: false };
  const { body } = await postJson(url, payload, timeoutMs);
  const elapsedMs = Date.now() - started;
  const rawText = body.response || '';
  const descriptions = parseBatchResponse(rawText, n);
  return { descriptions, elapsedMs, model, raw: body };
}

async function textInfer({ base, model, prompt, timeoutMs }) {
  const started = Date.now();
  const url = base + '/api/generate';
  const payload = { model, prompt, stream: false };
  const { body } = await postJson(url, payload, timeoutMs);
  const elapsedMs = Date.now() - started;
  const answer = body.response || '无结果';
  return { answer, elapsedMs, raw: body };
}

module.exports = { visionInfer, visionBatchInfer, textInfer, postJson, parseBatchResponse };
