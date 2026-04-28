'use strict';

function parseTimeRange(question) {
  const now = new Date();
  let m;
  m = question.match(/最近\s*(\d+)\s*分钟|(\d+)\s*分钟[内以]|(\d+)\s*分钟前/);
  if (m) {
    const mins = parseInt(m[1] || m[2] || m[3]);
    return { start: new Date(now.getTime() - mins * 60000), end: now };
  }
  m = question.match(/最近\s*(\d+)\s*小时|(\d+)\s*小时[内以]|(\d+)\s*小时前/);
  if (m) {
    const hrs = parseInt(m[1] || m[2] || m[3]);
    return { start: new Date(now.getTime() - hrs * 3600000), end: now };
  }
  m = question.match(/最近\s*半小时|半小时[内以]/);
  if (m) return { start: new Date(now.getTime() - 30 * 60000), end: now };
  if (question.includes('今天')) {
    return { start: new Date(now.getFullYear(), now.getMonth(), now.getDate()), end: now };
  }
  if (question.includes('昨天')) {
    return {
      start: new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1),
      end: new Date(now.getFullYear(), now.getMonth(), now.getDate()),
    };
  }
  if (question.includes('上午') || question.includes('早上')) {
    return {
      start: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 0),
      end: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 12),
    };
  }
  if (question.includes('下午')) {
    return {
      start: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 12),
      end: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 18),
    };
  }
  if (question.includes('晚上')) {
    return {
      start: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 18),
      end: new Date(now.getFullYear(), now.getMonth(), now.getDate(), 23, 59, 59),
    };
  }
  return null;
}

module.exports = { parseTimeRange };
