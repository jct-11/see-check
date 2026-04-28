'use strict';

const fs = require('fs');
const ollama = require('./providers/ollama');
const openaiCompat = require('./providers/openaiCompat');
const { flowLog, flowError } = require('../utils/log');
const { normalizeBaseUrl } = require('../utils/net');
const { getActiveInferenceServer } = require('../config');

function pickProvider(server) {
  return server && server.apiStyle === 'openai' ? openaiCompat : ollama;
}

function createInferenceService(getConfig) {
  function activeContext(baseOverride) {
    const cfg = getConfig();
    const base = normalizeBaseUrl(baseOverride || cfg.ollamaBase || '');
    const server = getActiveInferenceServer(cfg, base);
    const provider = pickProvider(server);
    const model = (server && server.model) || cfg.model || 'gemma4:e2b';
    const timeoutMs = (cfg.inferenceRequestTimeoutSec || 120) * 1000;
    return { base, server, provider, model, timeoutMs };
  }

  async function recognizeFrame({ framePath, prompt, ollamaBase }) {
    const cfg = getConfig();
    const ctx = activeContext(ollamaBase);
    const imageBase64 = fs.readFileSync(framePath).toString('base64');
    const usePrompt = prompt || cfg.prompt;
    const endpoint = ctx.base + (ctx.server && ctx.server.apiStyle === 'openai' ? '/chat/completions' : '/api/generate');
    flowLog('模型', '请求视觉推理', { endpoint, model: ctx.model, framePath });
    try {
      const result = await ctx.provider.visionInfer({
        base: ctx.base,
        model: ctx.model,
        prompt: usePrompt,
        imageBase64,
        timeoutMs: ctx.timeoutMs,
      });
      flowLog('模型', '视觉推理完成', {
        model: result.model,
        elapsedMs: result.elapsedMs,
        descLen: result.description.length,
        preview: result.description.slice(0, 120),
      });
      return {
        success: true,
        description: result.description,
        model: result.model,
        timestamp: new Date().toISOString(),
        inferenceTime: result.elapsedMs,
      };
    } catch (e) {
      flowError('模型', '视觉推理请求失败', { endpoint, model: ctx.model, error: e.message });
      throw e;
    }
  }

  /**
   * 批量视觉推理：1 帧走单图接口，2-6 帧在一次 API 调用中处理所有图片。
   * frames: Array<{ framePath: string, ... }>
   * 返回与 frames 等长的 recognition 结果数组
   */
  async function recognizeFrameBatch(frames) {
    if (!frames || frames.length === 0) return [];
    const cfg = getConfig();
    const ctx = activeContext(cfg.ollamaBase);

    // 单帧：走普通接口
    if (frames.length === 1) {
      const r = await recognizeFrame({ framePath: frames[0].framePath, prompt: cfg.prompt, ollamaBase: cfg.ollamaBase });
      return [r];
    }

    // 多帧：provider 必须支持 visionBatchInfer
    if (!ctx.provider.visionBatchInfer) {
      flowLog('模型', 'provider 不支持批量推理，回退到逐帧模式', { count: frames.length });
      return Promise.all(frames.map((f) => recognizeFrame({ framePath: f.framePath, prompt: cfg.prompt, ollamaBase: cfg.ollamaBase })));
    }

    const imagesBase64 = frames.map((f) => fs.readFileSync(f.framePath).toString('base64'));
    flowLog('模型', '批量视觉推理开始', { count: frames.length, model: ctx.model });
    const result = await ctx.provider.visionBatchInfer({
      base: ctx.base,
      model: ctx.model,
      prompt: cfg.prompt,
      imagesBase64,
      timeoutMs: ctx.timeoutMs,
    });
    flowLog('模型', '批量视觉推理完成', {
      count: frames.length,
      elapsedMs: result.elapsedMs,
      msPerFrame: Math.round(result.elapsedMs / frames.length),
      results: result.descriptions.map((desc, i) => ({
        index: i + 1,
        frameId: frames[i] && frames[i].id,
        descLen: (desc || '').length,
        preview: (desc || '无结果').slice(0, 80),
      })),
    });
    return result.descriptions.map((desc, i) => ({
      success: true,
      description: desc || '无结果',
      model: result.model,
      timestamp: new Date().toISOString(),
      inferenceTime: Math.round(result.elapsedMs / frames.length),
    }));
  }

  async function batchRecognize(items, opts) {
    return Promise.all(items.map((it) => recognizeFrame({
      framePath: it.framePath,
      prompt: opts && opts.prompt,
      ollamaBase: opts && opts.ollamaBase,
    })));
  }

  async function textInfer({ prompt, ollamaBase }) {
    const ctx = activeContext(ollamaBase);
    flowLog('模型', '请求文本推理', { endpoint: ctx.base, model: ctx.model, promptLen: prompt.length });
    try {
      const r = await ctx.provider.textInfer({
        base: ctx.base,
        model: ctx.model,
        prompt,
        timeoutMs: ctx.timeoutMs,
      });
      flowLog('模型', '文本推理完成', { elapsedMs: r.elapsedMs, answerLen: r.answer.length });
      return { answer: r.answer, inferenceTime: r.elapsedMs };
    } catch (e) {
      flowError('模型', '文本推理请求失败', { error: e.message });
      throw e;
    }
  }

  return { recognizeFrame, recognizeFrameBatch, batchRecognize, textInfer, activeContext };
}

module.exports = { createInferenceService };
