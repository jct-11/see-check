'use strict';

const { flowLog, flowWarn, flowError } = require('../utils/log');

const HOUR_24 = 24 * 60 * 60 * 1000;
const DAY_7 = 7 * 24 * 60 * 60 * 1000;
const COMPRESS_INTERVAL = 5 * 60 * 1000;

function pickBest(a, b) {
  const aScore = (a.faces ? a.faces.length * 10 : 0) + (a.description || '').length;
  const bScore = (b.faces ? b.faces.length * 10 : 0) + (b.description || '').length;
  return bScore > aScore ? b : a;
}

function createLifecycleService({ memoryStore, vectorIndex }) {
  async function run() {
    const memories = await memoryStore.list();
    if (memories.length === 0) return { recent: 0, compressed: 0, archived: 0 };

    const now = Date.now();
    const recent = [];
    const compressible = [];
    const archivable = [];
    for (const mem of memories) {
      const age = now - new Date(mem.timestamp).getTime();
      if (age <= HOUR_24) recent.push(mem);
      else if (age <= DAY_7) compressible.push(mem);
      else archivable.push(mem);
    }

    flowLog('生命周期', '开始清理', {
      total: memories.length,
      recent: recent.length,
      compressible: compressible.length,
      archivable: archivable.length,
    });

    let compressedCount = 0;
    if (compressible.length > 0) {
      const buckets = {};
      for (const mem of compressible) {
        const bucketKey = Math.floor(new Date(mem.timestamp).getTime() / COMPRESS_INTERVAL);
        if (!buckets[bucketKey]) buckets[bucketKey] = [];
        buckets[bucketKey].push(mem);
      }
      const toDelete = [];
      for (const bucket of Object.values(buckets)) {
        if (bucket.length <= 1) continue;
        const best = bucket.reduce(pickBest);
        for (const mem of bucket) {
          if (mem.id === best.id) continue;
          toDelete.push(mem.id);
          compressedCount++;
        }
      }
      if (toDelete.length > 0) {
        try {
          await memoryStore.deleteMany(toDelete);
          await vectorIndex.deleteMany(toDelete);
          flowLog('生命周期', '压缩完成', { deleted: toDelete.length });
        } catch (e) {
          flowError('生命周期', '压缩删除失败', { error: e.message });
        }
      }
    }

    let archivedCount = 0;
    if (archivable.length > 0) {
      const ids = archivable.map((m) => m.id);
      try {
        await memoryStore.archive(ids);
        await vectorIndex.deleteMany(ids);
        archivedCount = ids.length;
        flowLog('生命周期', '归档完成', { archived: archivedCount });
      } catch (e) {
        flowError('生命周期', '归档失败', { error: e.message, count: ids.length });
      }
    }

    if (compressedCount > 0 || archivedCount > 0) {
      let indexSize = null;
      try { indexSize = vectorIndex.size ? await Promise.resolve(vectorIndex.size()) : null; } catch (_) {}
      flowLog('生命周期', '记忆清理完成', {
        recent: recent.length,
        compressed: compressedCount,
        archived: archivedCount,
        indexSize,
      });
    } else {
      flowLog('生命周期', '无需清理');
    }
    return { recent: recent.length, compressed: compressedCount, archived: archivedCount };
  }

  function startScheduler(intervalMs) {
    const ms = Number(intervalMs) > 0 ? Number(intervalMs) : 60 * 60 * 1000;
    // 延迟 60 秒再做第一次清理，避免与启动初期的推理写操作并发冲突
    setTimeout(() => {
      run().catch((e) => flowError('生命周期', '初始清理失败', { error: e.message }));
    }, 60 * 1000);
    return setInterval(() => {
      run().catch((e) => flowError('生命周期', '定时清理失败', { error: e.message }));
    }, ms);
  }

  return { run, startScheduler };
}

module.exports = { createLifecycleService };
