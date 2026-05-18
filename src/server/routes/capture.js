'use strict';

const fs = require('fs');
const path = require('path');
const formidable = require('formidable').formidable;
const { readBody, sendJson, sendError } = require('../router');
const { CAPTURE_DIR } = require('../../utils/paths');
const { flowLog, flowWarn, flowError } = require('../../utils/log');

function registerCaptureRoutes(router, ctx) {
  router.post('/api/capture', (req, res) => {
    flowLog('捕获', '收到摄像头帧 (上传)');
    const form = formidable({ uploadDir: CAPTURE_DIR, keepExtensions: true });
    form.parse(req, async (err, fields, files) => {
      if (err) {
        flowWarn('捕获', '表单解析失败', { error: err.message });
        return sendError(res, 400, err);
      }
      try {
        const frameFiles = files.frame;
        if (!frameFiles || !frameFiles[0]) {
          return sendJson(res, 400, { success: false, error: '没有收到文件' });
        }
        const tmpPath = frameFiles[0].filepath;
        ctx.captureController.ingestUploaded({ filePath: tmpPath, ts: Date.now() });
        try { fs.unlinkSync(tmpPath); } catch (_) {}
        const cfg = ctx.getConfig();
        sendJson(res, 200, {
          success: true,
          accepted: true,
          source: 'upload',
          modelInfo: { url: cfg.ollamaBase, model: cfg.model },
        });
      } catch (err2) {
        flowError('捕获', '上传帧处理失败', { error: err2.message });
        sendError(res, 500, err2);
      }
    });
  });

  router.post('/api/capture-native', (req, res) => {
    const latest = path.join(CAPTURE_DIR, 'latest.jpg');
    if (!fs.existsSync(latest)) {
      return sendJson(res, 503, { success: false, error: '摄像头未就绪' });
    }
    try {
      ctx.captureController.ingestUploaded({ filePath: latest, ts: Date.now() });
      const cfg = ctx.getConfig();
      sendJson(res, 200, {
        success: true,
        accepted: true,
        source: 'native-snapshot',
        modelInfo: { url: cfg.ollamaBase, model: cfg.model },
      });
    } catch (err) {
      flowError('捕获', 'native 取帧失败', { error: err.message });
      sendError(res, 500, err);
    }
  });

  router.post('/api/start', async (req, res) => {
    try {
      let mode = 'server';
      try {
        const raw = await readBody(req);
        if (raw) {
          const body = JSON.parse(raw);
          if (body.mode) mode = body.mode;
        }
      } catch (_) { /* body 为空或非 JSON，使用默认 server 模式 */ }

      if (mode === 'local') {
        // 浏览器本地摄像头模式：不启动 ffmpeg，帧由 /api/capture 上传进入流水线
        await ctx.captureController.startLocal();
      } else {
        // 服务端 ffmpeg 模式
        await ctx.captureController.start();
      }
      sendJson(res, 200, {
        success: true,
        mode,
        sourceName: ctx.captureController.sourceName(),
        isCapturing: ctx.captureController.isCapturing(),
      });
    } catch (err) {
      sendError(res, 500, err);
    }
  });

  router.post('/api/stop', async (req, res) => {
    try {
      await ctx.captureController.stop();
      sendJson(res, 200, { success: true });
    } catch (err) {
      sendError(res, 500, err);
    }
  });

  // 添加状态查询端点
  router.get('/api/status', async (req, res) => {
    try {
      sendJson(res, 200, {
        success: true,
        isCapturing: ctx.captureController ? ctx.captureController.isCapturing() : false,
        sourceName: ctx.captureController ? ctx.captureController.sourceName() : null,
        timestamp: Date.now(),
      });
    } catch (err) {
      sendError(res, 500, err);
    }
  });
}

module.exports = { registerCaptureRoutes };
