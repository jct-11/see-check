'use strict';

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const { createRouter } = require('./router');
const { registerConfigRoutes } = require('./routes/config');
const { registerCaptureRoutes } = require('./routes/capture');
const { registerQueryRoutes } = require('./routes/query');
const { registerSpatialRoutes } = require('./routes/spatial');
const { registerMemoryRoutes } = require('./routes/memory');
const { registerWebRoutes } = require('./routes/web');
const { registerLogsRoutes } = require('./routes/logs');
const { CERTS_DIR } = require('../utils/paths');
const { getLocalIP } = require('../utils/net');
const { getServerLabel } = require('../config');
const { flowLog, flowWarn, currentLogFile } = require('../utils/log');

function createApp(ctx) {
  const router = createRouter();
  registerConfigRoutes(router, ctx);
  registerCaptureRoutes(router, ctx);
  registerQueryRoutes(router, ctx);
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
