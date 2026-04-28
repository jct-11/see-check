'use strict';

const { spawn } = require('child_process');
const { flowLog, flowWarn, flowError, flowDebug } = require('../../utils/log');

const SOI = Buffer.from([0xff, 0xd8]);
const EOI = Buffer.from([0xff, 0xd9]);

function createMjpegSplitter(onFrame) {
  let buf = Buffer.alloc(0);
  return function push(chunk) {
    buf = buf.length === 0 ? Buffer.from(chunk) : Buffer.concat([buf, chunk]);
    let cursor = 0;
    while (true) {
      const start = buf.indexOf(SOI, cursor);
      if (start < 0) {
        if (cursor > 0) buf = buf.slice(cursor);
        return;
      }
      const end = buf.indexOf(EOI, start + 2);
      if (end < 0) {
        if (start > 0) buf = buf.slice(start);
        return;
      }
      const jpeg = buf.slice(start, end + 2);
      onFrame(jpeg);
      cursor = end + 2;
    }
  };
}

/**
 * 从 ffmpeg stderr 中提取设备支持的分辨率@帧率列表
 * 示例行：[avfoundation @ 0x...] 1280x720@[30.000030 30.000030]fps
 */
function parseSupportedModes(stderrText) {
  const modes = [];
  const re = /(\d+x\d+)@\[[\d.]+\s+([\d.]+)\]fps/g;
  let m;
  while ((m = re.exec(stderrText)) !== null) {
    modes.push(m[1] + '@' + Math.round(parseFloat(m[2])) + 'fps');
  }
  return [...new Set(modes)];
}

function createFfmpegAvfoundationSource(opts) {
  const o = opts || {};
  const deviceIndex = Number.isFinite(o.deviceIndex) ? o.deviceIndex : 0;
  // 默认 30fps（avfoundation 摄像头普遍支持，20fps 不被支持会导致 I/O error）
  const framerate = Number(o.framerate) > 0 ? Number(o.framerate) : 30;
  const width = Number(o.width) > 0 ? Number(o.width) : 1280;
  const height = Number(o.height) > 0 ? Number(o.height) : 720;
  const quality = Number(o.quality) > 0 ? Number(o.quality) : 5;
  const ffmpegBinary = o.ffmpegBinary || 'ffmpeg';
  let proc = null;
  let running = false;
  let frameCount = 0;
  let stderrBuf = '';          // 缓存 stderr 用于事后解析
  let framerateWarned = false; // 避免重复打印帧率不支持的警告

  async function start(onFrame) {
    if (running) return;
    const args = [
      '-hide_banner', '-loglevel', 'warning',
      '-f', 'avfoundation',
      '-framerate', String(framerate),
      '-pixel_format', 'uyvy422',
      '-video_size', `${width}x${height}`,  // 明确指定分辨率，防止摄像头选最高档(如 2944x1656)
      '-probesize', '32',                   // 减少探帧时间，加速首帧
      '-i', String(deviceIndex) + ':none',
      '-c:v', 'mjpeg',
      '-q:v', String(quality),
      '-fflags', '+nobuffer',              // 输出侧减少缓冲
      '-f', 'image2pipe',
      '-',
    ];
    flowLog('采帧', '启动 ffmpeg', { binary: ffmpegBinary, deviceIndex, framerate, width, height, quality });
    proc = spawn(ffmpegBinary, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    running = true;
    frameCount = 0;
    stderrBuf = '';
    framerateWarned = false;

    // 诊断：追踪摄像头原始输出是否变化（4字节中段采样 + 尺寸）
    let _lastRawFp = null;
    let _lastRawLen = 0;
    let _fpChanges = 0;

    const splitter = createMjpegSplitter((jpeg) => {
      try {
        frameCount++;

        // 取 JPEG 中段 4 字节作为"内容指纹"——不同场景的 DCT 系数在此区域几乎必然不同
        const mid = jpeg.length >> 1;
        const fp = jpeg.length > mid + 4
          ? jpeg[mid].toString(16) + jpeg[mid + 1].toString(16)
            + jpeg[mid + 2].toString(16) + jpeg[mid + 3].toString(16)
          : 'short';

        if (fp !== _lastRawFp || jpeg.length !== _lastRawLen) {
          _fpChanges++;
          flowLog('采帧', '摄像头原始帧内容变化', {
            seq: frameCount,
            sizeBytes: jpeg.length,
            fp,
            totalChanges: _fpChanges,
          });
          _lastRawFp = fp;
          _lastRawLen = jpeg.length;
        }

        if (frameCount % 100 === 0) {
          flowLog('采帧', 'ffmpeg 帧计数', { total: frameCount, sizeBytes: jpeg.length });
        } else {
          flowDebug('采帧', 'ffmpeg 帧', { seq: frameCount, sizeBytes: jpeg.length });
        }
        onFrame({ jpeg, ts: Date.now() });
      } catch (e) {
        flowWarn('采帧', '帧回调失败', { seq: frameCount, error: e.message });
      }
    });

    proc.stdout.on('data', splitter);

    // 5 秒内没收到第一帧则打 WARN，帮助诊断摄像头被占用或权限问题
    const firstFrameTimer = setTimeout(() => {
      if (running && frameCount === 0) {
        flowWarn('采帧', 'ffmpeg 已运行 5s 但未收到任何帧——最可能原因：macOS 未授权当前终端/Node 访问摄像头', {
          deviceIndex,
          framerate,
          fix: '请打开「系统设置 → 隐私与安全性 → 摄像头」，找到运行本程序的终端（Terminal / iTerm2）并开启权限，然后重启服务',
          otherCauses: '摄像头被其他进程占用 | 设备索引有误（可改 config.json capture.ffmpeg.deviceIndex）',
        });
      }
    }, 5000);

    proc.on('exit', () => clearTimeout(firstFrameTimer));

    proc.stderr.on('data', (d) => {
      const text = d.toString();
      stderrBuf += text;
      const trimmed = text.trim();
      if (!trimmed) return;

      // 帧率不支持：提取并打印设备支持的模式，给出明确指引
      if (/not supported by the device/i.test(trimmed) && !framerateWarned) {
        framerateWarned = true;
        flowWarn('采帧', `ffmpeg：摄像头不支持 ${framerate}fps，请修改 config.json capture.ffmpeg.framerate`, {
          requested: framerate,
          hint: '可选值见下方 supportedModes（启动后从 stderr 解析）',
        });
        return;
      }

      // 列出支持模式时只写文件（信息量大但不需要打到控制台）
      if (/Supported modes|avfoundation.*\d+x\d+@/i.test(trimmed)) {
        flowDebug('采帧', 'ffmpeg stderr（支持模式）', trimmed);
        return;
      }

      // 真正的错误（I/O error、Error opening 等）
      if (/error|failed|invalid|no such/i.test(trimmed)) {
        flowWarn('采帧', 'ffmpeg stderr', trimmed);
        return;
      }

      // 其余信息只写文件
      flowDebug('采帧', 'ffmpeg stderr', trimmed);
    });

    proc.on('exit', (code, signal) => {
      running = false;

      if (code !== 0 && code !== null) {
        // 解析 stderr 中的支持模式，方便快速诊断
        const modes = parseSupportedModes(stderrBuf);
        flowError('采帧', 'ffmpeg 进程异常退出', {
          code,
          signal,
          totalFrames: frameCount,
          diagnosis: framerateWarned
            ? `摄像头不支持 ${framerate}fps，请从 supportedModes 中选一个值填入 config.json`
            : '未知原因，请查看 WARN/DEBUG 日志',
          supportedModes: modes.length > 0 ? modes : '未能解析（可用 STMEM_DEBUG=1 查看完整 stderr）',
        });
      } else {
        flowLog('采帧', 'ffmpeg 进程退出', { code, signal, totalFrames: frameCount });
      }
      proc = null;
    });

    proc.on('error', (e) => {
      running = false;
      if (e.code === 'ENOENT') {
        flowError('采帧', 'ffmpeg 未安装或不在 PATH 中', {
          binary: ffmpegBinary,
          hint: '请执行: brew install ffmpeg',
        });
      } else {
        flowError('采帧', 'ffmpeg 启动失败', { error: e.message, binary: ffmpegBinary });
      }
      proc = null;
    });
  }

  async function stop() {
    if (!proc) { running = false; return; }
    flowLog('采帧', '停止 ffmpeg', { totalFrames: frameCount });
    try { proc.kill('SIGTERM'); } catch (_) {}
    await new Promise((r) => setTimeout(r, 200));
    if (proc) {
      try { proc.kill('SIGKILL'); } catch (_) {}
    }
    proc = null;
    running = false;
  }

  return {
    name: 'ffmpeg-avfoundation',
    start,
    stop,
    isRunning: () => running,
  };
}

module.exports = {
  createFfmpegAvfoundationSource,
  createMjpegSplitter,
};
