'use strict';

const { readBody, sendJson, sendError } = require('../router');
const { flowLog, flowError } = require('../../utils/log');

function registerQueryRoutes(router, ctx) {
  router.post('/api/query', async (req, res) => {
    try {
      const body = await readBody(req);
      const { question, ollamaBase } = JSON.parse(body || '{}');
      if (!question) return sendJson(res, 400, { success: false, error: '请输入问题' });
      const result = await ctx.queryService.query({ question, ollamaBase });
      sendJson(res, 200, result);
    } catch (err) {
      flowError('查询', '查询失败', { error: err.message });
      sendError(res, 500, err);
    }
  });
}

module.exports = { registerQueryRoutes };
