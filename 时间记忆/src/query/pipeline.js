'use strict';

const { flowLog } = require('../utils/log');
const { parseTimeRange } = require('./timeParse');
const { EMBED_DIM } = require('../config/defaults');

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

function createQueryService({
  inferenceService,
  embeddingService,
  vectorIndex,
  memoryStore,
  faceLibrary,
  framesProvider,
  getConfig,
}) {
  async function query({ question, ollamaBase }) {
    if (!question) throw new Error('请输入问题');
    flowLog('查询', '收到语义查询', { question });
    const started = Date.now();
    const timeRange = parseTimeRange(question);

    const allRows = await vectorIndex.filterByTime({});
    let candidates = allRows;
    let timeFilterEmpty = false;
    if (timeRange) {
      const filtered = allRows.filter((c) => {
        const t = new Date(c.captureTime || c.timestamp);
        return t >= timeRange.start && t <= timeRange.end;
      });
      flowLog('查询', '时间过滤', { before: allRows.length, after: filtered.length });
      if (filtered.length > 0) candidates = filtered;
      else { timeFilterEmpty = true; flowLog('查询', '时间范围内无记忆，回退到全量搜索'); }
    }

    let queryEmb = null;
    try { queryEmb = await embeddingService.embed(question); } catch (_) {}

    let topMemories;
    if (candidates.length === 0) topMemories = [];
    else if (candidates.length <= 20) topMemories = candidates;
    else if (queryEmb) {
      topMemories = candidates
        .map((c) => ({ ...c, score: cosine(queryEmb, c.vector) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 20);
    } else {
      topMemories = candidates.slice().sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp)).slice(0, 20);
    }

    if (topMemories.length === 0) {
      let totalMemories = allRows.length;
      try { if (vectorIndex.size) totalMemories = await Promise.resolve(vectorIndex.size()); } catch (_) {}
      return {
        success: true,
        answer: '没有找到任何记忆数据。',
        memoriesCount: 0,
        totalMemories,
        inferenceTime: Date.now() - started,
        matchedFrames: [],
      };
    }

    const allMemories = await memoryStore.list();
    const memoryById = {};
    for (const mem of allMemories) memoryById[mem.id] = mem;

    const lib = faceLibrary.getAll();
    const contextLines = topMemories
      .slice()
      .sort((a, b) => new Date(a.captureTime || a.timestamp) - new Date(b.captureTime || b.timestamp))
      .map((m) => {
        let line = '[' + new Date(m.captureTime || m.timestamp).toLocaleString('zh-CN') + '] ' + m.description;
        const fullMem = memoryById[m.id];
        if (fullMem && fullMem.faces && fullMem.faces.length > 0) {
          const faceInfo = fullMem.faces.map((f) => {
            const name = lib[f.personId] ? lib[f.personId].name : '未命名';
            return f.personId + '(' + name + ',性别:' + f.gender + ',年龄:~' + f.age + ')';
          }).join(', ');
          line += ' [识别到人物: ' + faceInfo + ']';
        }
        return line;
      })
      .join('\n');

    const uniquePersons = new Set();
    for (const m of topMemories) {
      const fullMem = memoryById[m.id];
      if (fullMem && fullMem.faces) fullMem.faces.forEach((f) => uniquePersons.add(f.personId));
    }

    const nowStr = new Date().toLocaleString('zh-CN');
    let timeHint = '';
    if (timeFilterEmpty) timeHint = '\n\n注意：用户询问的时间范围内没有捕获到任何画面，以上记忆来自其它时间段，请据实告知用户该时间段没有记忆，但可以参考其它时间的记忆来回答。';
    let faceHint = '';
    if (uniquePersons.size > 0) {
      faceHint = '\n\n人物身份信息：系统通过人脸识别追踪了 ' + uniquePersons.size + ' 个不同的人。相同 personId 表示同一个人，请据此准确统计人数，避免重复计数。';
    }
    const fullPrompt = '你是一个视觉记忆助手。当前时间是 ' + nowStr + '。以下是摄像头在不同时间捕获的画面描述（记忆）：\n\n' + contextLines + timeHint + faceHint + '\n\n请根据以上记忆回答用户的问题。如果记忆中没有相关信息，请如实说明。回答要简洁准确。\n\n用户问题：' + question;
    flowLog('查询', '发送至 LLM', { memoriesCount: topMemories.length, promptLength: fullPrompt.length });
    const cfg = getConfig();
    const base = ollamaBase || cfg.ollamaBase;
    const { answer, inferenceTime } = await inferenceService.textInfer({ prompt: fullPrompt, ollamaBase: base });
    flowLog('查询', '查询完成', { inferenceTime, answerLength: answer.length });

    const citedDates = answer.match(/\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}\s+\d{1,2}:\d{2}:\d{2}/g);
    let relevant = [];
    if (citedDates && citedDates.length > 0) {
      const seen = new Set();
      for (const ds of citedDates) {
        const ref = new Date(ds.replace(/\//g, '-'));
        if (isNaN(ref.getTime())) continue;
        for (const m of topMemories) {
          if (seen.has(m.id)) continue;
          const mt = new Date(m.captureTime || m.timestamp);
          if (Math.abs(ref.getTime() - mt.getTime()) < 5000) {
            seen.add(m.id);
            relevant.push({ ...m, score: 1 });
          }
        }
      }
      if (relevant.length > 0) flowLog('查询', '从回答中提取引用帧', { cited: citedDates.length, matched: relevant.length });
    }
    if (relevant.length === 0 && queryEmb) {
      const scored = topMemories.map((m) => ({ ...m, score: cosine(queryEmb, m.vector) }));
      scored.sort((a, b) => b.score - a.score);
      const threshold = scored[0] ? scored[0].score * 0.75 : 0;
      relevant = scored.filter((m) => m.score >= threshold).slice(0, 8);
      flowLog('查询', '回退到相似度筛选', { relevant: relevant.length });
    }
    if (relevant.length === 0) {
      relevant = topMemories.slice().sort((a, b) => new Date(b.captureTime || b.timestamp) - new Date(a.captureTime || a.timestamp)).slice(0, 5);
    }

    const allFrameFiles = await Promise.resolve(framesProvider.list());
    const matchedFrames = relevant
      .slice()
      .sort((a, b) => new Date(b.captureTime || b.timestamp) - new Date(a.captureTime || a.timestamp))
      .map((m) => {
        const mId = parseInt(m.id, 10);
        const f = allFrameFiles.find((ff) => parseInt(ff.id, 10) === mId);
        if (!f) return null;
        return {
          id: f.id,
          thumbnail: f.thumbnail,
          hasMemory: f.hasMemory,
          createdAt: f.createdAt,
          score: m.score ? Math.round(m.score * 100) : null,
          memory: f.memory ? {
            description: f.memory.description,
            timestamp: f.memory.timestamp,
            captureTime: f.memory.captureTime,
            model: f.memory.model,
            serverLabel: f.memory.serverLabel,
            inferenceTime: f.memory.inferenceTime,
            id: f.memory.id,
            framePath: f.memory.framePath,
            faces: f.memory.faces || [],
          } : null,
        };
      })
      .filter(Boolean);

    let totalMemories = allRows.length;
    try { if (vectorIndex.size) totalMemories = await Promise.resolve(vectorIndex.size()); } catch (_) {}
    return {
      success: true,
      answer,
      memoriesCount: topMemories.length,
      totalMemories,
      inferenceTime: Date.now() - started,
      matchedFrames,
    };
  }

  return { query };
}

module.exports = { createQueryService };
