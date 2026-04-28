'use strict';

const fs = require('fs');
const path = require('path');
const { sendJson } = require('../router');
const { DATA_DIR } = require('../../utils/paths');

let isSpatialCapturing = false;
let spatialFrameCounter = 0;
let spatialMode = 'server';

function registerSpatialRoutes(router) {
  // 场景图API
  router.get('/api/scene-graph', (req, res) => {
    const sgFile = path.join(DATA_DIR, 'scene_graph', 'scene_graph.json');
    if (fs.existsSync(sgFile)) {
      res.writeHead(200, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
      fs.createReadStream(sgFile).pipe(res);
      return;
    }
    sendJson(res, 404, { error: 'scene_graph.json not found' });
  });

  // PLY文件服务
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

  // 对象图片服务
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

  // 空间记忆API - 开始采集
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

  // 空间记忆API - 停止采集
  router.post('/api/spatial/stop', (req, res) => {
    try {
      isSpatialCapturing = false;
      console.log('[Spatial] 停止空间记忆采集');
      sendJson(res, 200, { success: true, message: '空间记忆采集已停止' });
    } catch (err) {
      sendJson(res, 500, { success: false, error: err.message });
    }
  });

  // 空间记忆API - 获取状态
  router.get('/api/spatial/status', (req, res) => {
    sendJson(res, 200, {
      success: true,
      isCapturing: isSpatialCapturing,
      frameCount: spatialFrameCounter,
      mode: spatialMode
    });
  });

  // 空间记忆API - 模拟处理帧（用于演示）
  router.post('/api/spatial/process-frame', (req, res) => {
    try {
      spatialFrameCounter++;
      
      // 模拟处理延迟
      setTimeout(() => {
        // 生成模拟的3D点云数据
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

        // 生成模拟的相机轨迹
        const angle = (spatialFrameCounter * 0.1) % (Math.PI * 2);
        const radius = 3;
        const trajectoryPoint = {
          x: Math.cos(angle) * radius,
          y: Math.sin(angle * 0.5) * 0.5,
          z: Math.sin(angle) * radius,
          frameId: spatialFrameCounter,
          timestamp: Date.now()
        };

        sendJson(res, 200, {
          success: true,
          frameId: spatialFrameCounter,
          pointCloud: pointCloud,
          trajectoryPoint: trajectoryPoint,
          processingTime: Math.floor(Math.random() * 200) + 50
        });
      }, Math.floor(Math.random() * 300) + 100);
    } catch (err) {
      sendJson(res, 500, { success: false, error: err.message });
    }
  });

  // 空间记忆API - 获取点云数据
  router.get('/api/spatial/point-cloud', (req, res) => {
    try {
      // 生成模拟点云数据
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

  // 空间记忆API - 获取轨迹数据
  router.get('/api/spatial/trajectory', (req, res) => {
    try {
      // 生成模拟轨迹
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