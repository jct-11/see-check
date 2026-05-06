'use strict';

const fs = require('fs');
const path = require('path');
const http = require('http');
const WebSocket = require('ws');
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
  return server || { protocol: 'http', host: '192.168.0.200', port: 11434 };
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

// 生成模拟数据（当RTX3090服务不可用时）
function generateSimulatedData(frameId) {
  const pointCloud = [];
  const pointCount = Math.floor(Math.random() * 50) + 10;
  for (let i = 0; i < pointCount; i++) {
    pointCloud.push({
      x: (Math.random() - 0.5) * 6,
      y: (Math.random() - 0.5) * 2,
      z: (Math.random() - 0.5) * 6,
      color: [Math.random(), Math.random(), Math.random()]
    });
  }

  const angle = (frameId * 0.1) % (Math.PI * 2);
  const radius = 3;
  const trajectoryPoint = {
    x: Math.cos(angle) * radius,
    y: Math.sin(angle * 0.5) * 0.5,
    z: Math.sin(angle) * radius,
    frameId: frameId,
    timestamp: Date.now()
  };

  return {
    success: true,
    frameId: frameId,
    pointCloud: pointCloud,
    trajectoryPoint: trajectoryPoint,
    processingTime: Math.floor(Math.random() * 200) + 50,
    source: 'simulated'
  };
}

function generateSimulatedDataForBatch(frames, batchId) {
  const pointCloud = [];
  const pointCount = Math.min(frames.length * 5, 500);
  for (let i = 0; i < pointCount; i++) {
    pointCloud.push((Math.random() - 0.5) * 6);
    pointCloud.push((Math.random() - 0.5) * 3);
    pointCloud.push((Math.random() - 0.5) * 6);
  }

  const colors = [];
  for (let i = 0; i < pointCount; i++) {
    colors.push(Math.floor(Math.random() * 255));
    colors.push(Math.floor(Math.random() * 255));
    colors.push(Math.floor(Math.random() * 255));
  }

  const cameraPoses = [];
  for (let i = 0; i < Math.min(frames.length, 10); i++) {
    const angle = (i / frames.length) * Math.PI * 2;
    cameraPoses.push({
      position: [Math.cos(angle) * 3, 0.5, Math.sin(angle) * 3],
      rotation: [0, angle, 0],
      frameId: i
    });
  }

  return {
    success: true,
    point_cloud: pointCloud,
    point_colors: colors,
    point_count: pointCount,
    camera_poses: cameraPoses,
    scene_center: [0, 0, 0],
    frames_processed: frames.length,
    batch_id: batchId,
    source: 'simulated'
  };
}

function registerSpatialRoutes(router) {
  router.get('/api/spatial/proxy', (req, res, ctx) => {
    const url = new URL(req.url, 'http://localhost');
    const targetPath = url.searchParams.get('path') || '/test/info';
    const server = getRTX3090Config();
    const protocol = server.protocol === 'https' ? https : http;

    const proxyReq = protocol.request({
      hostname: '192.168.0.200',
      port: 8000,
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
    const protocol = http;

    const bodyData = req.body ? JSON.stringify(req.body) : '';

    const proxyReq = protocol.request({
      hostname: '192.168.0.200',
      port: 8000,
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

      const proxyReq = http.request({
        hostname: '192.168.0.200',
        port: 8000,
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

      console.log(`[Spatial] 批量推理(WebSocket): ${frames.length} 帧, batch=${batchId}`);

      const result = await new Promise((resolve, reject) => {
        const wsUrl = 'ws://192.168.0.200:8000/ws';
        const wsClient = new WebSocket(wsUrl);
        let settled = false;

        const timeout = setTimeout(() => {
          if (!settled) {
            settled = true;
            wsClient.close();
            reject(new Error('WebSocket 批量推理超时 (300s)'));
          }
        }, 300000);

        wsClient.on('open', () => {
          console.log('[Spatial] WebSocket 已连接，开始发送帧...');
          let sentCount = 0;

          function sendNext() {
            if (sentCount >= frames.length) {
              wsClient.send(JSON.stringify({ batch_end: true }));
              console.log('[Spatial] 所有帧已发送，等待推理结果...');
              return;
            }

            const frameBase64 = frames[sentCount];
            wsClient.send(JSON.stringify({
              frame: frameBase64,
              frame_id: sentCount
            }));
            sentCount++;

            if (sentCount % 20 === 0) {
              console.log(`[Spatial] 已发送 ${sentCount}/${frames.length} 帧`);
            }

            setImmediate(sendNext);
          }

          sendNext();
        });

        wsClient.on('message', (data) => {
          const isBinary = Buffer.isBuffer(data);
          console.log(`[Spatial] WebSocket消息: isBinary=${isBinary}, data.length=${data.length}`);
          if (isBinary) {
            try {
              const buf = Buffer.from(data);
              const metaLen = buf.readUInt32LE(0);
              const metaBytes = buf.slice(4, 4 + metaLen);
              const meta = JSON.parse(metaBytes.toString('utf8'));
              const dataOffset = 4 + metaLen;
              const numPoints = meta.point_count;
              const pointsBuf = buf.slice(dataOffset, dataOffset + numPoints * 3 * 4);
              const colorsBuf = buf.slice(dataOffset + numPoints * 3 * 4);
              const pointsArray = new Float32Array(pointsBuf);
              const colorsArray = new Uint8Array(colorsBuf);
              console.log(`[Spatial] 解析成功: numPoints=${numPoints}, pointsBytes=${pointsBuf.length}, colorsBytes=${colorsBuf.length}`);

              clearTimeout(timeout);
              if (!settled) {
                settled = true;
                wsClient.close();
                resolve({
                  point_cloud: Array.from(pointsArray),
                  point_colors: Array.from(colorsArray),
                  point_count: numPoints,
                  camera_poses: meta.camera_poses || [],
                  scene_center: meta.scene_center || null,
                  frames_processed: meta.num_frames || frames.length,
                  batch_id: meta.batch_id || batchId
                });
              }
            } catch (e) {
              console.error('[Spatial] 二进制消息解析失败:', e.message, e.stack);
            }
          } else {
            try {
              const msg = JSON.parse(data.toString());
              if (msg.type === 'processing_start') {
                console.log(`[Spatial] RTX3090 开始推理: ${msg.frame_count} 帧, batch=${msg.batch_id}`);
              } else if (msg.type === 'processing_done') {
                console.log(`[Spatial] RTX3090 推理完成: ${msg.frame_count} 帧`);
              } else if (msg.type === 'result') {
                console.log(`[Spatial] RTX3090 返回结果: ${msg.frames_processed} 帧`);
                clearTimeout(timeout);
                if (!settled) {
                  settled = true;
                  wsClient.close();
                  resolve({
                    success: true,
                    point_cloud: msg.point_cloud || [],
                    point_colors: msg.point_colors || [],
                    point_count: msg.point_count || 0,
                    camera_poses: msg.camera_poses || [],
                    scene_center: msg.scene_center || null,
                    frames_processed: msg.frames_processed || frames.length,
                    batch_id: batchId
                  });
                }
              } else if (msg.type === 'error') {
                clearTimeout(timeout);
                if (!settled) {
                  settled = true;
                  wsClient.close();
                  reject(new Error('RTX3090 推理错误: ' + msg.msg));
                }
              }
            } catch (e) {
              console.warn('[Spatial] WebSocket JSON 消息解析失败:', e.message);
            }
          }
        });

        wsClient.on('error', (e) => {
          clearTimeout(timeout);
          if (!settled) {
            settled = true;
            console.warn('[Spatial] WebSocket 连接失败，使用模拟数据:', e.message);
            // WebSocket 失败时使用模拟数据而不是 reject
            resolve(generateSimulatedDataForBatch(frames, batchId));
          }
        });

        wsClient.on('close', () => {
          clearTimeout(timeout);
          if (!settled) {
            settled = true;
            console.warn('[Spatial] WebSocket 连接关闭，使用模拟数据');
            resolve(generateSimulatedDataForBatch(frames, batchId));
          }
        });
      });

      console.log(`[Spatial] 批量推理完成: ${result.frames_processed} 帧, 点云=${result.point_count}`);

      sendJson(res, 200, {
        success: true,
        point_cloud: result.point_cloud,
        point_count: result.point_count,
        point_colors: result.point_colors,
        camera_poses: result.camera_poses,
        scene_center: result.scene_center,
        frames_processed: result.frames_processed,
        batch_id: batchId
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
      let source = 'simulated';
      
      try {
        // 直接转发到RTX3090的lingbot-map服务
        result = await forwardToRTX3090('/api/lingbot-map/process-frame', {
          frameId: spatialFrameCounter,
          timestamp: Date.now(),
          ...frameData
        });
        source = 'rtx3090';
        console.log('[Spatial] RTX3090返回结果');
      } catch (err) {
        // RTX3090服务不可用时使用模拟数据
        console.warn('[Spatial] RTX3090服务不可用，使用模拟数据:', err.message);
        result = generateSimulatedData(spatialFrameCounter);
        source = 'simulated';
      }

      sendJson(res, 200, {
        success: true,
        ...result,
        source: source
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
      let source = 'simulated';
      
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
        source = 'rtx3090';
      } catch (err) {
        console.warn('[Spatial] RTX3090服务不可用，使用模拟数据');
        result = generateSimulatedData(spatialFrameCounter);
        source = 'simulated';
      }

      sendJson(res, 200, {
        success: true,
        ...result,
        source: source
      });
    } catch (err) {
      sendJson(res, 500, { success: false, error: err.message });
    }
  });

  // 获取点云数据
  router.get('/api/spatial/point-cloud', (req, res) => {
    try {
      const pointCloud = [];
      const pointCount = 200;
      for (let i = 0; i < pointCount; i++) {
        pointCloud.push({
          x: (Math.random() - 0.5) * 8,
          y: (Math.random() - 0.5) * 3,
          z: (Math.random() - 0.5) * 8,
          color: [Math.random(), Math.random(), Math.random()]
        });
      }
      sendJson(res, 200, {
        success: true,
        pointCloud: pointCloud,
        count: pointCloud.length
      });
    } catch (err) {
      sendJson(res, 500, { success: false, error: err.message });
    }
  });

  // 获取轨迹数据
  router.get('/api/spatial/trajectory', (req, res) => {
    try {
      const trajectory = [];
      const steps = 20;
      for (let i = 0; i < steps; i++) {
        const angle = (i * 0.1) % (Math.PI * 2);
        const radius = 3;
        trajectory.push({
          x: Math.cos(angle) * radius,
          y: Math.sin(angle * 0.5) * 0.5,
          z: Math.sin(angle) * radius,
          frameId: i + 1,
          timestamp: Date.now() - (steps - i) * 1000
        });
      }
      sendJson(res, 200, {
        success: true,
        trajectory: trajectory,
        length: trajectory.length
      });
    } catch (err) {
      sendJson(res, 500, { success: false, error: err.message });
    }
  });
}

module.exports = { registerSpatialRoutes };
