'use strict';

const { readBody, sendJson, sendError } = require('../router');
const { saveUserConfig } = require('../../config');
const { flowLog } = require('../../utils/log');

function registerConfigRoutes(router, ctx) {
  router.get('/api/config', (req, res) => {
    sendJson(res, 200, { success: true, config: ctx.getConfig() });
  });

  router.post('/api/config', async (req, res) => {
    try {
      const body = await readBody(req);
      const incoming = JSON.parse(body || '{}');
      const current = ctx.getConfig();
      const merged = {
        ...current,
        ...incoming,
        inferenceServers: Array.isArray(incoming.inferenceServers)
          ? incoming.inferenceServers
          : current.inferenceServers,
      };
      const next = saveUserConfig(merged);
      ctx.setConfig(next);
      flowLog('配置', '已保存', {
        ollamaBase: next.ollamaBase,
        embedBase: next.embedBase,
        model: next.model,
        fps: next.fps,
        inferenceRequestTimeoutSec: next.inferenceRequestTimeoutSec,
        selectedInferenceServerId: next.selectedInferenceServerId,
        servers: (next.inferenceServers || []).map((s) => ({ id: s.id, name: s.name, host: s.host, port: s.port })),
      });
      sendJson(res, 200, { success: true, config: next });
    } catch (err) {
      sendError(res, 400, err);
    }
  });
}

module.exports = { registerConfigRoutes };
