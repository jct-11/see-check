'use strict';

const http = require('http');
const https = require('https');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { createRouter } = require('./router');
const { registerConfigRoutes } = require('./routes/config');
const { registerCaptureRoutes } = require('./routes/capture');
const { registerQueryRoutes } = require('./routes/query');
const { registerSpatialRoutes } = require('./routes/spatial_route');
const { registerMemoryRoutes } = require('./routes/memory');
const { registerWebRoutes } = require('./routes/web');
const { registerBatchRoutes } = require('./routes/batch');
const { registerLogsRoutes } = require('./routes/logs');
const { CERTS_DIR } = require('../utils/paths');
const { getLocalIP } = require('../utils/net');
const { getServerLabel } = require('../config');
const { flowLog, flowWarn, currentLogFile } = require('../utils/log');

const WS_MAGIC = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';

function parseWsFrame(buf) {
  if (buf.length < 2) return null;
  const first = buf[0];
  const second = buf[1];
  const fin = (first >> 7) & 1;
  const opcode = first & 0x0f;
  const masked = (second >> 7) & 1;
  let payloadLen = second & 0x7f;
  let offset = 2;
  if (payloadLen === 126) {
    if (buf.length < 4) return null;
    payloadLen = buf.readUInt16BE(2);
    offset = 4;
  } else if (payloadLen === 127) {
    if (buf.length < 10) return null;
    payloadLen = Number(buf.readBigUInt64BE(2));
    offset = 10;
  }
  let maskKey = null;
  if (masked) {
    if (buf.length < offset + 4) return null;
    maskKey = buf.slice(offset, offset + 4);
    offset += 4;
  }
  if (buf.length < offset + payloadLen) return null;
  let payload = buf.slice(offset, offset + payloadLen);
  if (masked && maskKey) {
    for (let i = 0; i < payload.length; i++) {
      payload[i] ^= maskKey[i % 4];
    }
  }
  return { fin, opcode, payload, totalLen: offset + payloadLen };
}

function buildWsFrame(payload, opcode, masked) {
  const maskBit = masked ? 0x80 : 0;
  const header = [];
  header.push(0x80 | opcode);
  if (payload.length < 126) {
    header.push(maskBit | payload.length);
  } else if (payload.length < 65536) {
    header.push(maskBit | 126);
    const lenBuf = Buffer.alloc(2);
    lenBuf.writeUInt16BE(payload.length, 0);
    header.push(...lenBuf);
  } else {
    header.push(maskBit | 127);
    const lenBuf = Buffer.alloc(8);
    lenBuf.writeBigUInt64BE(BigInt(payload.length), 0);
    header.push(...lenBuf);
  }
  if (masked) {
    const maskKey = crypto.randomBytes(4);
    const maskedPayload = Buffer.alloc(payload.length);
    for (let i = 0; i < payload.length; i++) {
      maskedPayload[i] = payload[i] ^ maskKey[i % 4];
    }
    return Buffer.concat([Buffer.from(header), maskKey, maskedPayload]);
  }
  return Buffer.concat([Buffer.from(header), payload]);
}

function handleWsUpgrade(req, socket, head, targetHost, targetPort) {
  const wsKey = req.headers['sec-websocket-key'];
  if (!wsKey) { socket.destroy(); return; }

  const acceptHash = crypto.createHash('sha1').update(wsKey + WS_MAGIC).digest('base64');

  const proxyReq = http.request({
    hostname: targetHost,
    port: targetPort,
    path: req.url,
    method: 'GET',
    headers: {
      'Upgrade': 'websocket',
      'Connection': 'Upgrade',
      'Sec-WebSocket-Key': wsKey,
      'Sec-WebSocket-Version': '13',
    },
  });

  proxyReq.on('upgrade', (proxyRes, proxySocket, proxyHead) => {
    socket.write('HTTP/1.1 101 Switching Protocols\r\n' +
      'Upgrade: websocket\r\n' +
      'Connection: Upgrade\r\n' +
      'Sec-WebSocket-Accept: ' + acceptHash + '\r\n' +
      '\r\n');

    if (proxyHead && proxyHead.length > 0) {
      socket.write(proxyHead);
    }

    proxySocket.on('data', (chunk) => {
      socket.write(chunk);
    });

    socket.on('data', (chunk) => {
      proxySocket.write(chunk);
    });

    proxySocket.on('close', () => { socket.destroy(); });
    proxySocket.on('error', () => { socket.destroy(); });
    socket.on('close', () => { proxySocket.destroy(); });
    socket.on('error', () => { proxySocket.destroy(); });
  });

  proxyReq.on('error', (e) => {
    flowWarn('WS代理', '连接RTX3090失败', { error: e.message });
    socket.destroy();
  });

  proxyReq.end();
}

function createApp(ctx) {
  const router = createRouter();
  registerConfigRoutes(router, ctx);
  registerCaptureRoutes(router, ctx);
  registerQueryRoutes(router, ctx);
  registerBatchRoutes(router, ctx);
  registerSpatialRoutes(router, ctx);
  registerMemoryRoutes(router, ctx);
  registerLogsRoutes(router);
  registerWebRoutes(router);

  const handler = (req, res) => router.handle(req, res, ctx);

  function start(port) {
    const server = http.createServer(handler);
    let httpsServer = null;
    const httpsPort = port + 1;
    const keyPath = path.join(CERTS_DIR, 'key.pem');
    const certPath = path.join(CERTS_DIR, 'cert.pem');
    if (fs.existsSync(keyPath) && fs.existsSync(certPath)) {
      try {
        httpsServer = https.createServer({
          key: fs.readFileSync(keyPath),
          cert: fs.readFileSync(certPath),
        }, handler);
      } catch (e) {
        flowWarn('服务', 'HTTPS 启动失败', { error: e.message });
      }
    }

    server.on('upgrade', (req, socket, head) => {
      if (req.url === '/ws/spatial') {
        const cfg = ctx.getConfig();
        const servers = cfg.inferenceServers || [];
        const rtx = servers.find(s => s.id === '3090');
        const targetHost = rtx ? rtx.host : '192.168.0.200';
        const targetPort = 8000;
        handleWsUpgrade(req, socket, head, targetHost, targetPort);
      } else {
        socket.destroy();
      }
    });

    if (httpsServer) {
      httpsServer.on('upgrade', (req, socket, head) => {
        if (req.url === '/ws/spatial') {
          const cfg = ctx.getConfig();
          const servers = cfg.inferenceServers || [];
          const rtx = servers.find(s => s.id === '3090');
          const targetHost = rtx ? rtx.host : '192.168.0.200';
          const targetPort = 8000;
          handleWsUpgrade(req, socket, head, targetHost, targetPort);
        } else {
          socket.destroy();
        }
      });
    }

    server.on('error', (err) => {
      if (err.code === 'EADDRINUSE') {
        console.error('❌ HTTP 端口 ' + port + ' 已被占用，请先释放端口：fuser -k ' + port + '/tcp');
        process.exit(1);
      } else {
        flowWarn('服务', 'HTTP 启动失败', { error: err.message });
      }
    });

    return new Promise((resolve) => {
      server.listen(port, '0.0.0.0', () => {
        const ip = getLocalIP();
        const cfg = ctx.getConfig();
        console.log('');
        console.log('🧠 识境时空记忆');
        console.log('='.repeat(50));
        console.log('📊 服务已启动');
        console.log('📍 本地地址：http://localhost:' + port);
        console.log('🌐 局域网 HTTP：http://' + ip + ':' + port);
        if (httpsServer) {
          httpsServer.on('error', (err) => {
            if (err.code === 'EADDRINUSE') {
              console.log('🔒 HTTPS 端口 ' + httpsPort + ' 被占用，跳过 HTTPS（仅 HTTP 可用）');
            } else {
              flowWarn('服务', 'HTTPS 启动失败', { error: err.message });
            }
          });
          httpsServer.listen(httpsPort, '0.0.0.0', () => {
            console.log('🔒 局域网 HTTPS：https://' + ip + ':' + httpsPort + '  ← 其它设备请用此地址');
          });
        }
        console.log('⚙️  推理设备：' + getServerLabel(cfg, cfg.ollamaBase));
        console.log('📝 日志文件：' + currentLogFile());
        console.log('');
        console.log('⚠️  按 Ctrl+C 停止服务');
        console.log('='.repeat(50));
        resolve({ server, httpsServer });
      });
    });
  }

  return { router, start, handler };
}

module.exports = { createApp };
