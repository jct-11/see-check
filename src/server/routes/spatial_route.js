'use strict';

const fs = require('fs');
const path = require('path');
const http = require('http');
const https = require('https');
const { readBody, sendJson } = require('../router');
const { DATA_DIR } = require('../../utils/paths');

let config = {};
try {
  const configPath = path.join(__dirname, '../../config.json');
  if (fs.existsSync(configPath)) {
    config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  }
} catch (e) {
  console.warn('[Spatial] Failed to load config:', e.message);
}

let isSpatialCapturing = false;
let spatialFrameCounter = 0;
let spatialMode = 'server';

function getRTX3090Config() {
  const servers = config.inferenceServers || [];
  const server = servers.find(s => s.id === '3090');
  return server || { protocol: 'http', host: '192.168.0.200', port: 8000 };
}

// 转发请求到RTX3090服务
async function forwardToRTX3090(endpoint, frameData) {
  return new Promise((resolve, reject) => {
    const server = getRTX3090Config();
    const protocol = server.protocol === 'https' ? https : http;
    
    const options = {
      hostname: server.host,
      port: server.port,
      path: endpoint,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    };

    const req = protocol.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => {
        data += chunk;
      });
      res.on('end', () => {
        try {
          const result = JSON.parse(data);
          resolve(result);
        } catch (err) {
          reject(new Error('Invalid response from RTX3090'));
        }
      });
    });

    req.on('error', (e) => {
      console.warn('[Spatial] Failed to connect to RTX3090:', e.message);
      reject(e);
    });

    req.write(JSON.stringify(frameData));
    req.end();
  });
}

// ============== 路由注册 ==============

function registerSpatialRoutes(router) {
  router.get('/api/spatial/proxy', (req, res, ctx) => {
    const url = new URL(req.url, 'http://localhost');
    const targetPath = url.searchParams.get('path') || '/test/info';
    const server = getRTX3090Config();
    const protocol = server.protocol === 'https' ? https : http;

    const proxyReq = protocol.request({
      hostname: server.host,
      port: server.port,
      path: targetPath,
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    }, (proxyRes) => {
      res.writeHead(proxyRes.statusCode, {
        'Content-Type': proxyRes.headers['content-type'] || 'application/json',
        'Access-Control-Allow-Origin': '*',
      });
      proxyRes.pipe(res);
    });

    proxyReq.on('error', (e) => {
      sendJson(res, 502, { success: false, error: 'RTX3090 不可达: ' + e.message });
    });

    proxyReq.end();
  });

  router.post('/api/spatial/proxy', (req, res, ctx) => {
    const url = new URL(req.url, 'http://localhost');
    const targetPath = url.searchParams.get('path') || '/api/lingbot-map/process-frame';
    const server = getRTX3090Config();
    const protocol = server.protocol === 'https' ? https : http;

    const bodyData = req.body ? JSON.stringify(req.body) : '';

    const proxyReq = protocol.request({
      hostname: server.host,
      port: server.port,
      path: targetPath,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Content-Length': Buffer.byteLength(bodyData),
      },
    }, (proxyRes) => {
      res.writeHead(proxyRes.statusCode, {
        'Content-Type': proxyRes.headers['content-type'] || 'application/json',
        'Access-Control-Allow-Origin': '*',
      });
      proxyRes.pipe(res);
    });

    proxyReq.on('error', (e) => {
      sendJson(res, 502, { success: false, error: 'RTX3090 不可达: ' + e.message });
    });

    if (bodyData) proxyReq.write(bodyData);
    proxyReq.end();
  });

  router.post('/api/spatial/upload-frame', async (req, res, ctx) => {
    try {
      const raw = await readBody(req);
      const body = JSON.parse(raw);
      const imageBase64 = body.imageBase64 || body.image;
      if (!imageBase64) {
        sendJson(res, 400, { success: false, error: '缺少图片数据' });
        return;
      }

      const imgBuf = Buffer.from(imageBase64, 'base64');
      const boundary = '----SpatialFrameBoundary' + Date.now();
      const header = Buffer.from(
        '--' + boundary + '\r\n' +
        'Content-Disposition: form-data; name="file"; filename="frame.jpg"\r\n' +
        'Content-Type: image/jpeg\r\n\r\n'
      );
      const footer = Buffer.from('\r\n--' + boundary + '--\r\n');
      const bodyBuf = Buffer.concat([header, imgBuf, footer]);

      const server = getRTX3090Config();
      const protocol = server.protocol === 'https' ? https : http;

      const proxyReq = protocol.request({
        hostname: server.host,
        port: server.port,
        path: '/upload',
        method: 'POST',
        headers: {
          'Content-Type': 'multipart/form-data; boundary=' + boundary,
          'Content-Length': bodyBuf.length,
        },
      }, (proxyRes) => {
        let data = '';
        proxyRes.on('data', (chunk) => { data += chunk; });
        proxyRes.on('end', () => {
          try {
            const result = JSON.parse(data);
            sendJson(res, 200, { success: true, ...result });
          } catch (e) {
            sendJson(res, 502, { success: false, error: 'RTX3090 返回解析失败', raw: data.substring(0, 200) });
          }
        });
      });

      proxyReq.on('error', (e) => {
        sendJson(res, 502, { success: false, error: 'RTX3090 不可达: ' + e.message });
      });

      proxyReq.write(bodyBuf);
      proxyReq.end();
    } catch (e) {
      sendJson(res, 400, { success: false, error: '请求解析失败: ' + e.message });
    }
  });

  router.post('/api/spatial/batch-inference', async (req, res, ctx) => {
    try {
      const raw = await readBody(req);
      const body = JSON.parse(raw);
      const frames = body.frames || [];
      const batchId = body.batch_id || 0;

      if (frames.length === 0) {
        sendJson(res, 400, { success: false, error: '没有帧数据' });
        return;
      }

      console.log(`[Spatial] 批量推理(HTTP): ${frames.length} 帧, batch=${batchId}`);

      let result;
      try {
        const server = getRTX3090Config();
        result = await new Promise((resolve, reject) => {
          const options = {
            hostname: server.host,
            port: server.port,
            path: '/batch-inference',
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json'
            },
            timeout: 600000
          };

          const protocol = server.protocol === 'https' ? https : http;
          const req = protocol.request(options, (proxyRes) => {
            let data = '';
            proxyRes.on('data', (chunk) => { data += chunk; });
            proxyRes.on('end', () => {
              try {
                const parsed = JSON.parse(data);
                if (parsed.code === 200 || parsed.success) {
                  resolve(parsed);
                } else {
                  reject(new Error(parsed.msg || '批量推理失败'));
                }
              } catch (e) {
                reject(new Error('RTX3090 返回解析失败: ' + data.substring(0, 200)));
              }
            });
          });

          req.on('error', (e) => {
            reject(new Error('RTX3090 HTTP请求失败: ' + e.message));
          });

          req.on('timeout', () => {
            req.destroy();
            reject(new Error('RTX3090 请求超时'));
          });

          req.write(JSON.stringify({ frames: frames, batch_id: batchId }));
          req.end();
        });

      } catch (e) {
        console.error('[Spatial] RTX3090 HTTP请求失败:', e.message);
        throw new Error('RTX3090 HTTP请求失败: ' + e.message);
      }

      console.log(`[Spatial] 批量推理完成: ${result.frames_processed} 帧, 点云=${result.point_count}, source=${result.source || 'rtx3090'}`);

      sendJson(res, 200, {
        success: true,
        points: result.points,
        point_count: result.point_count,
        colors: result.colors,
        cameraPoses: result.cameraPoses,
        sceneCenter: result.scene_center,
        frames_processed: result.frames_processed,
        batchId: batchId,
        source: 'rtx3090'
      });

    } catch (e) {
      console.error('[Spatial] 批量推理异常:', e.message);
      sendJson(res, 500, { success: false, error: '批量推理失败: ' + e.message });
    }
  });

  router.get('/api/scene-graph', (req, res) => {
    const sgFile = path.join(DATA_DIR, 'scene_graph', 'scene_graph.json');
    if (fs.existsSync(sgFile)) {
      res.writeHead(200, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
      fs.createReadStream(sgFile).pipe(res);
      return;
    }
    sendJson(res, 404, { error: 'scene_graph.json not found' });
  });

  router.any((p) => p === '/api/ply/default' || p === '/api/ply/point_cloud.ply', (req, res) => {
    const plyFile = path.join(DATA_DIR, 'ply', 'point_cloud.ply');
    if (fs.existsSync(plyFile)) {
      const stat = fs.statSync(plyFile);
      res.writeHead(200, {
        'Content-Type': 'application/octet-stream',
        'Content-Length': stat.size,
        'Cache-Control': 'public, max-age=3600'
      });
      fs.createReadStream(plyFile).pipe(res);
      return;
    }
    res.writeHead(404); res.end('PLY not found');
  });

  router.any((p) => p.startsWith('/data/object/'), (req, res, ctx) => {
    const fname = path.basename(ctx.url.pathname);
    if (/^\d+\.(jpg|jpeg|png)$/i.test(fname)) {
      const imgFile = path.join(DATA_DIR, 'object', fname);
      if (fs.existsSync(imgFile)) {
        res.writeHead(200, { 'Content-Type': 'image/jpeg', 'Cache-Control': 'public, max-age=86400' });
        fs.createReadStream(imgFile).pipe(res);
        return;
      }
    }
    res.writeHead(404); res.end('Not found');
  });

  router.get('/api/spatial/rt-server-status', (req, res) => {
    const server = getRTX3090Config();
    sendJson(res, 200, {
      success: true,
      server: server,
      model: 'lingbot-map (RTX3090)'
    });
  });

  router.post('/api/spatial/start', (req, res) => {
    try {
      const body = req.body || {};
      spatialMode = body.mode || 'server';
      isSpatialCapturing = true;
      spatialFrameCounter = 0;
      console.log(`[Spatial] 开始空间记忆采集, 模式: ${spatialMode}`);
      sendJson(res, 200, { success: true, message: '空间记忆采集已启动' });
    } catch (err) {
      sendJson(res, 500, { success: false, error: err.message });
    }
  });

  router.post('/api/spatial/stop', (req, res) => {
    try {
      isSpatialCapturing = false;
      console.log('[Spatial] 停止空间记忆采集');
      sendJson(res, 200, { success: true, message: '空间记忆采集已停止' });
    } catch (err) {
      sendJson(res, 500, { success: false, error: err.message });
    }
  });

  router.get('/api/spatial/status', (req, res) => {
    sendJson(res, 200, {
      success: true,
      isCapturing: isSpatialCapturing,
      frameCount: spatialFrameCounter,
      mode: spatialMode
    });
  });

  // 转发帧到RTX3090处理
  router.post('/api/spatial/process-frame', async (req, res) => {
    try {
      spatialFrameCounter++;
      const frameData = req.body || {};
      
      console.log(`[Spatial] 转发帧 #${spatialFrameCounter} 到 RTX3090`);

      let result;
      
      try {
        // 直接转发到RTX3090的lingbot-map服务
        result = await forwardToRTX3090('/api/lingbot-map/process-frame', {
          frameId: spatialFrameCounter,
          timestamp: Date.now(),
          ...frameData
        });
        console.log('[Spatial] RTX3090返回结果');
      } catch (err) {
        // RTX3090服务不可用时直接返回错误
        console.error('[Spatial] RTX3090服务不可用:', err.message);
        throw new Error('RTX3090服务不可用: ' + err.message);
      }

      sendJson(res, 200, {
        success: true,
        ...result
      });
    } catch (err) {
      sendJson(res, 500, { success: false, error: err.message });
    }
  });

  // 转发完整帧到RTX3090
  router.post('/api/spatial/send-frame', async (req, res) => {
    try {
      spatialFrameCounter++;
      const body = req.body || {};
      const { imageBase64, timestamp, frameNumber } = body;

      console.log(`[Spatial] 转发完整帧 #${frameNumber || spatialFrameCounter} 到 RTX3090`);

      let result;
      
      try {
        // 直接转发到RTX3090
        result = await forwardToRTX3090('/api/lingbot-map/process-frame', {
          frameId: spatialFrameCounter,
          timestamp: timestamp || Date.now(),
          frameNumber: frameNumber,
          imageBase64: imageBase64,
          type: 'full_frame',
          ...body
        });
      } catch (err) {
        // RTX3090服务不可用时直接返回错误
        console.error('[Spatial] RTX3090服务不可用:', err.message);
        throw new Error('RTX3090服务不可用: ' + err.message);
      }

      sendJson(res, 200, {
        success: true,
        ...result
      });
    } catch (err) {
      sendJson(res, 500, { success: false, error: err.message });
    }
  });

  // 获取点云数据
  router.get('/api/spatial/point-cloud', (req, res) => {
    sendJson(res, 503, { success: false, error: 'RTX3090服务不可用，无法获取点云数据' });
  });

  // 获取轨迹数据
  router.get('/api/spatial/trajectory', (req, res) => {
    sendJson(res, 503, { success: false, error: 'RTX3090服务不可用，无法获取轨迹数据' });
  });
}

module.exports = { registerSpatialRoutes };
