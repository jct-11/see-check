'use strict';

const fs = require('fs');
const path = require('path');
const { LOGS_DIR } = require('../../utils/paths');
const { currentLogFile } = require('../../utils/log');
const { sendJson } = require('../router');

/**
 * GET /api/logs
 *   ?date=YYYY-MM-DD   (不传则今天)
 *   ?level=ERROR,WARN  (逗号分隔，不传则全量)
 *   ?step=管道         (按 step 过滤，不传则全量)
 *   ?n=200             (最多返回最近 N 条，默认 200)
 *
 * GET /api/logs/files  列出所有日志文件
 */
function registerLogsRoutes(router) {
  router.get('/api/logs/files', (req, res) => {
    try {
      if (!fs.existsSync(LOGS_DIR)) return sendJson(res, 200, { files: [] });
      const files = fs.readdirSync(LOGS_DIR)
        .filter((f) => f.startsWith('app-') && f.endsWith('.log'))
        .sort()
        .reverse()
        .map((f) => {
          const stat = fs.statSync(path.join(LOGS_DIR, f));
          return { name: f, sizeBytes: stat.size, mtime: stat.mtime.toISOString() };
        });
      sendJson(res, 200, { files });
    } catch (e) {
      sendJson(res, 500, { error: e.message });
    }
  });

  router.get('/api/logs', (req, res, info) => {
    try {
      const qs = info.url.searchParams;
      const dateStr = qs.get('date') || '';
      const levelFilter = qs.get('level') ? qs.get('level').toUpperCase().split(',').map((s) => s.trim()) : null;
      const stepFilter = qs.get('step') || null;
      const maxN = Math.min(parseInt(qs.get('n') || '200', 10) || 200, 2000);

      const logFile = dateStr
        ? path.join(LOGS_DIR, 'app-' + dateStr + '.log')
        : currentLogFile();

      if (!fs.existsSync(logFile)) {
        return sendJson(res, 200, { file: path.basename(logFile), entries: [], total: 0 });
      }

      const raw = fs.readFileSync(logFile, 'utf8').trim();
      if (!raw) return sendJson(res, 200, { file: path.basename(logFile), entries: [], total: 0 });

      const allLines = raw.split('\n');
      const entries = [];
      for (const line of allLines) {
        if (!line.trim()) continue;
        try {
          const entry = JSON.parse(line);
          if (levelFilter && !levelFilter.includes(entry.level)) continue;
          if (stepFilter && entry.step !== stepFilter) continue;
          entries.push(entry);
        } catch (_) {}
      }

      const total = entries.length;
      const recent = entries.slice(-maxN);

      sendJson(res, 200, {
        file: path.basename(logFile),
        total,
        returned: recent.length,
        entries: recent,
      });
    } catch (e) {
      sendJson(res, 500, { error: e.message });
    }
  });
}

module.exports = { registerLogsRoutes };
