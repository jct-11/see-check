'use strict';

const fs = require('fs');
const path = require('path');

// LOGS_DIR 通过懒加载避免循环依赖（paths 不依赖 log）
let _logsDir = null;
function getLogsDir() {
  if (_logsDir) return _logsDir;
  try {
    _logsDir = require('./paths').LOGS_DIR;
  } catch (_) {
    _logsDir = path.join(__dirname, '..', '..', 'logs');
  }
  try {
    if (!fs.existsSync(_logsDir)) fs.mkdirSync(_logsDir, { recursive: true });
  } catch (_) {}
  return _logsDir;
}

// 当前日志文件路径（按日期轮转）
let _currentDateStr = '';
let _currentLogPath = '';

/**
 * 格式化时间戳为 "YYYY-MM-DD HH:MM:SS"（本地时间）
 */
function fmtTs(date) {
  const d = date || new Date();
  const y = d.getFullYear();
  const mo = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  const h = String(d.getHours()).padStart(2, '0');
  const mi = String(d.getMinutes()).padStart(2, '0');
  const s = String(d.getSeconds()).padStart(2, '0');
  return y + '-' + mo + '-' + day + ' ' + h + ':' + mi + ':' + s;
}

function todayStr() {
  const d = new Date();
  return d.getFullYear() + '-' +
    String(d.getMonth() + 1).padStart(2, '0') + '-' +
    String(d.getDate()).padStart(2, '0');
}

function getLogPath() {
  const today = todayStr();
  if (today !== _currentDateStr) {
    _currentDateStr = today;
    _currentLogPath = path.join(getLogsDir(), 'app-' + today + '.log');
  }
  return _currentLogPath;
}

// ── 调用位置解析 ──────────────────────────────────────────────────────────────

// 用完整路径匹配，避免同名文件误跳
const THIS_FILE_FULL = __filename;
// 项目根目录（用于提取相对路径）
const PROJECT_ROOT = path.join(__dirname, '..', '..');

/**
 * 从当前调用栈中找到第一个不属于 log.js 的帧，
 * 返回 { file, func, line }
 *   file: 相对于项目根目录的路径，如 "src/inference/index.js"
 *   func: 调用函数名，如 "recognizeFrame"
 *   line: 行号
 */
function getCaller() {
  const err = new Error();
  const lines = (err.stack || '').split('\n');

  for (let i = 1; i < lines.length; i++) {
    const raw = lines[i];
    // 跳过 log.js 自身的所有帧（用全路径匹配）
    if (raw.includes(THIS_FILE_FULL)) continue;
    // 跳过 Node.js 内部模块帧（无本地 .js 路径）
    if (/at node:|node:internal|node:events/.test(raw)) continue;

    // 匹配两种格式：
    //   at funcName (filepath:line:col)
    //   at filepath:line:col
    const m = raw.match(/^\s+at\s+(?:(.+?)\s+\()?(.+?\.(?:js|ts|mjs|cjs)):(\d+):\d+\)?/);
    if (!m) continue;

    let func = (m[1] || '').trim();
    const fullPath = m[2] || '';
    const lineNum = parseInt(m[3], 10);

    // 清理函数名前缀
    func = func
      .replace(/^async\s+/, '')            // async func → func
      .replace(/^Object\./, '')             // Object.method → method
      .replace(/^Module\./, '')             // Module.method → method
      .replace(/^Promise\..+$/, 'Promise')
      .trim() || '<anonymous>';

    // 计算相对于项目根的路径，便于阅读
    let file = fullPath;
    try {
      const rel = path.relative(PROJECT_ROOT, fullPath);
      // 仅当是项目内文件时才用相对路径
      if (!rel.startsWith('..')) file = rel;
    } catch (_) {}

    return { file, func, line: lineNum };
  }

  return { file: '', func: '', line: 0 };
}

// ── 格式化工具 ────────────────────────────────────────────────────────────────

function formatDetailForConsole(detail) {
  if (detail === undefined || detail === null) return '';
  if (typeof detail === 'string') return '\n' + detail;
  try { return '\n' + JSON.stringify(detail, null, 2); } catch (_) { return ''; }
}

function safeStringify(val) {
  if (val === undefined || val === null) return null;
  if (typeof val === 'string') return val;
  try { return JSON.parse(JSON.stringify(val)); } catch (_) { return String(val); }
}

// ── 核心写入 ──────────────────────────────────────────────────────────────────

function emit(level, step, title, detail) {
  const now = new Date();
  const ts = fmtTs(now);
  const caller = getCaller();

  // ── 控制台输出 ──
  // 格式：2026-04-24 14:15:15 [LEVEL][step] title  (file:line func())
  const location = caller.file
    ? '  (' + caller.file + ':' + caller.line + ' ' + caller.func + '())'
    : '';
  const consoleLine = ts + ' [' + level + '][' + step + '] ' + title + location;

  if (level === 'WARN') {
    console.warn(consoleLine + formatDetailForConsole(detail));
  } else if (level === 'ERROR') {
    console.error(consoleLine + formatDetailForConsole(detail));
  } else if (level === 'DEBUG') {
    if (process.env.STMEM_DEBUG === '1') {
      console.log(consoleLine + formatDetailForConsole(detail));
    }
  } else {
    console.log(consoleLine + formatDetailForConsole(detail));
  }

  // ── 文件写入 ──
  try {
    const entry = { ts, level, step, title };
    if (caller.file) {
      entry.file = caller.file;
      entry.func = caller.func;
      entry.line = caller.line;
    }
    if (detail !== undefined && detail !== null) {
      entry.detail = safeStringify(detail);
    }
    fs.appendFileSync(getLogPath(), JSON.stringify(entry) + '\n', 'utf8');
  } catch (_) {
    // 日志写入失败不应影响主流程
  }
}

// ── 公共 API ──────────────────────────────────────────────────────────────────

/** INFO：常规流程日志 */
function flowLog(step, title, detail) { emit('INFO', step, title, detail); }

/** WARN：可恢复的异常（重试、配置回退等） */
function flowWarn(step, title, detail) { emit('WARN', step, title, detail); }

/** ERROR：需要关注的错误（推理失败、启动失败等） */
function flowError(step, title, detail) { emit('ERROR', step, title, detail); }

/**
 * DEBUG：高频事件（每帧、ffmpeg stderr 等）
 * 默认只写文件；设置环境变量 STMEM_DEBUG=1 同时打印到控制台
 */
function flowDebug(step, title, detail) { emit('DEBUG', step, title, detail); }

/** 返回今日日志文件路径 */
function currentLogFile() { return getLogPath(); }

module.exports = { flowLog, flowWarn, flowError, flowDebug, currentLogFile, fmtTs };
