// spatial.js - 空间记忆模块合并文件
// 整合: spatial_api.js, spatial_ui.js, spatial_visualizer.js, spatial_map.js

// ============== 工具函数 ==============
/** 安全的 URL 拼接（修复 //batch 问题） */
function buildUrl(path) {
  const base = BATCH_SERVER_URL || '';
  if (!path) return base;
  if (path.startsWith('/')) return base + path;
  return base + '/' + path;
}

/** 安全的 fetch 包装 */
async function safeFetch(url, options) {
  const fullUrl = buildUrl(url);
  const response = await fetch(fullUrl, options);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  return response;
}

/** Three.js 资源释放工具 */
function disposeObject(obj) {
  if (!obj) return;
  if (obj.geometry) {
    obj.geometry.dispose();
    for (let attr in obj.geometry.attributes) {
      if (obj.geometry.attributes[attr]) {
        obj.geometry.attributes[attr].dispose();
      }
    }
  }
  if (obj.material) {
    if (Array.isArray(obj.material)) {
      obj.material.forEach(m => m.dispose());
    } else {
      obj.material.dispose();
    }
  }
}

/** 相机位姿平滑（指数移动平均 + 步长限制） */
function smoothCameraPose(rawPose) {
  if (!SpatialState.smoothPose) {
    SpatialState.smoothPose = {
      t: [...rawPose.t_c2w],
      R: rawPose.R_c2w.map(row => [...row])
    };
    return SpatialState.smoothPose;
  }

  const alpha = SpatialState.smoothAlpha;
  const maxStep = SpatialState.maxPoseStep;

  // 平滑平移
  for (let i = 0; i < 3; i++) {
    let diff = rawPose.t_c2w[i] - SpatialState.smoothPose.t[i];
    if (Math.abs(diff) > maxStep) {
      diff = diff > 0 ? maxStep : -maxStep;
    }
    SpatialState.smoothPose.t[i] += alpha * diff;
  }

  // 平滑旋转（简单 EMA）
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const rawR = rawPose.R_c2w[i][j];
      const currentR = SpatialState.smoothPose.R[i][j];
      SpatialState.smoothPose.R[i][j] = currentR + alpha * (rawR - currentR);
    }
  }

  return SpatialState.smoothPose;
}

/**
 * 将 Base64 编码字符串转换为 Float32Array
 * @param {string} base64Str - Base64 编码的字符串
 * @returns {Float32Array} 解码后的 Float32Array
 */
function base64ToFloat32Array(base64Str) {
  if (!base64Str) return new Float32Array(0);
  
  // 解码 Base64 为 ArrayBuffer
  const binaryStr = atob(base64Str);
  const len = binaryStr.length;
  const bytes = new Uint8Array(len);
  
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryStr.charCodeAt(i);
  }
  
  // 转换为 Float32Array
  return new Float32Array(bytes.buffer);
}

// ============== 全局状态封装 ==============
const SpatialState = {
  isCapturing: false,
  frameCounter: 0,
  totalFramesCollected: 0,
  currentBatchId: null,
  isInitialBatch: true,
  captureTimer: null,
  collectedFrames: [],
  isBatchProcessing: false,
  videoStream: null,
  currentView: '3d',
  isUploading: false,
  isInferenceStarted: false,
  onLogMessage: null,
  onStatusUpdate: null,
  fetchBatchId: null,
  currentFetchFrame: 0,
  isFetchingFrames: false,
  totalFramesAvailable: 0,
  framePointClouds: {},
  framePointCloudObjects: {},
  camerasData: {},
  accumulatedPoints: [],
  accumulatedColors: [],
  frameRanges: {},
  cameraFollowEnabled: false,
  currentCameraTargetPos: null,
  currentCameraLookAt: null,
  cameraFollowDistance: 2.0,
  cameraFollowSmoothFactor: 0.2,
  cameraFrustumsVisible: true,
  cameraFrustums: [],
  sceneCenter: [0, 0, 0],
  sceneScale: 1.0,
  renderLoopTimer: null,
  animationId: null,
  resizeObserver: null,
  domCache: null,
  smoothPose: null,
  smoothAlpha: 0.2,
  maxPoseStep: 0.1,
};

// ============== Legacy variable compatibility (old code references these) ==============
Object.defineProperties(window, {
  spatialIsCapturing: {
    get: () => SpatialState.isCapturing,
    set: (val) => SpatialState.isCapturing = val
  },
  spatialFrameCounter: {
    get: () => SpatialState.frameCounter,
    set: (val) => SpatialState.frameCounter = val
  },
  totalFramesCollected: {
    get: () => SpatialState.totalFramesCollected,
    set: (val) => SpatialState.totalFramesCollected = val
  },
  currentBatchId: {
    get: () => SpatialState.currentBatchId,
    set: (val) => SpatialState.currentBatchId = val
  },
  isInitialBatch: {
    get: () => SpatialState.isInitialBatch,
    set: (val) => SpatialState.isInitialBatch = val
  },
  spatialCaptureTimer: {
    get: () => SpatialState.captureTimer,
    set: (val) => SpatialState.captureTimer = val
  },
  collectedFrames: {
    get: () => SpatialState.collectedFrames,
    set: (val) => SpatialState.collectedFrames = val
  },
  isBatchProcessing: {
    get: () => SpatialState.isBatchProcessing,
    set: (val) => SpatialState.isBatchProcessing = val
  },
  spatialVideoStream: {
    get: () => SpatialState.videoStream,
    set: (val) => SpatialState.videoStream = val
  },
  isUploading: {
    get: () => SpatialState.isUploading,
    set: (val) => SpatialState.isUploading = val
  },
  isInferenceStarted: {
    get: () => SpatialState.isInferenceStarted,
    set: (val) => SpatialState.isInferenceStarted = val
  },
  onLogMessage: {
    get: () => SpatialState.onLogMessage,
    set: (val) => SpatialState.onLogMessage = val
  },
  onStatusUpdate: {
    get: () => SpatialState.onStatusUpdate,
    set: (val) => SpatialState.onStatusUpdate = val
  }
});

console.log('✅ 已启用旧变量兼容层');

// ============== 全局常量 ==============
/** 批次重叠帧数：批次处理完成后帧计数器的回退值 */
var SPATIAL_OVERLAP = 100;
/** 采集帧宽度（像素） */
var SPATIAL_FRAME_WIDTH = 320;
/** 采集帧高度（像素） */
var SPATIAL_FRAME_HEIGHT = 240;
/** 目标采集总帧数（用户可通过UI「总帧数」输入框修改） */
var spatialCaptureTargetFrames = Infinity;
var spatialKeyframeInterval = 1;  // Default: every frame is a keyframe (same as viser when <= 320 frames)
var spatialMaxImages = null;
/** 当前采集FPS（用户可通过UI「采集FPS」输入框修改） */
var spatialCaptureFps = 5;

// 批次服务配置
/** 批次服务地址（空字符串=走8080代理，避免跨域） */
var BATCH_SERVER_URL = '';

// ============== 模块导出对象 ==============
/** API层模块：负责采集、上传、批次处理 */
window.SpatialApi = {};
/** 3D可视化模块：负责Three.js场景、点云渲染、相机可视化 */
window.SpatialVisualizer = {};

// ============== API 层 (spatial_api.js) ==============

/**
 * 设置API回调函数
 * 供外部模块注册事件处理函数
 * @param {Object} callbacks - 回调函数对象
 * @param {Function} callbacks.onLogMessage - 日志消息输出时的回调
 * @param {Function} callbacks.onStatusUpdate - 采集状态变化时的回调
 */
function setSpatialApiCallbacks(callbacks) {
  if (callbacks.onLogMessage) onLogMessage = callbacks.onLogMessage;
  if (callbacks.onStatusUpdate) onStatusUpdate = callbacks.onStatusUpdate;
}

/**
 * 添加日志消息到浏览器控制台
 * 日志类型包括：'info'（普通信息）、'ok'（成功，绿色）、'err'（错误，红色）
 * @param {string} msg - 日志消息内容
 * @param {string} [type='info'] - 日志类型：'info'|'ok'|'err'
 */
/**
 * 添加日志消息到浏览器控制台
 * 日志类型包括：'info'（普通信息）、'ok'（成功，绿色）、'err'（错误，红色）
 * @param {string} msg - 日志消息内容
 * @param {string} [type='info'] - 日志类型：'info'|'ok'|'err'
 */
function addLog(msg, type) {
  const logMsg = msg;
  const logType = type || 'info';

  if (logType === 'err') {
    console.error('[Spatial] ' + logMsg);
  } else if (logType === 'ok') {
    console.log('%c[Spatial] ' + logMsg, 'color: #4caf50; font-weight: bold');
  } else {
    console.log('[Spatial] ' + logMsg);
  }
}

/**
 * 更新状态信息
 * 触发onStatusUpdate回调，通知外部模块状态变化
 * @param {Object} status - 状态对象（包含frameCount、pointCount等字段）
 */
/**
 * 更新状态信息
 * 触发onStatusUpdate回调，通知外部模块状态变化
 * @param {Object} status - 状态对象（包含frameCount、pointCount等字段）
 */
function updateStatus(status) {
  if (onStatusUpdate) onStatusUpdate(status);
}

/**
 * 获取当前采集帧率
 * @returns {number} 当前FPS值
 */
function getCurrentFPS() {
  return spatialCaptureFps;
}

/**
 * 计算帧间隔时间
 * 根据当前FPS计算两帧之间的时间间隔（毫秒）
 * @returns {number} 帧间隔时间（ms）
 */
function getFrameInterval() {
  return 1000 / getCurrentFPS();
}

/**
 * 初始化连接
 * 检查批次服务器是否可达，建立与服务器的通信
 */
function initConnection() {
  console.log('[Spatial API] 使用批次服务模式');
  checkBatchServerStatus();
}

/**
 * 启动连续采集
 * 按照设定的FPS持续采集帧数据，通过setTimeout递归调用实现
 */
function startContinuousCapture() {
  console.log('[Spatial API] startContinuousCapture: 被调用, spatialIsCapturing=' + spatialIsCapturing);
  
  if (!spatialIsCapturing) return;

  // 首次采集时，初始化批次号和进度条
  if (spatialFrameCounter === 0 && collectedFrames.length === 0) {
    // 生成新的批次号
    currentBatchId = generateBatchId();
    isInitialBatch = true;
    addLog('开始采集 (' + getCurrentFPS() + ' FPS)，批次号: ' + currentBatchId + '，目标 ' + spatialCaptureTargetFrames + ' 帧', 'info');
    showCaptureProgress();
  }

  // 采集当前帧
  collectFrame();

  // 递归调度下一帧采集
  if (spatialIsCapturing) {
    spatialCaptureTimer = setTimeout(startContinuousCapture, getFrameInterval());
  }
}

/**
 * 显示采集进度条
 * 在摄像头画面底部显示进度条
 */
function showCaptureProgress() {
  var bar = document.getElementById('captureProgressBar');
  if (bar) bar.style.display = 'block';
  updateCaptureProgress();
}

/**
 * 隐藏采集进度条
 */
function hideCaptureProgress() {
  var bar = document.getElementById('captureProgressBar');
  if (bar) bar.style.display = 'none';
}

/**
 * 更新采集进度条
 * 根据已采集帧数和目标帧数更新进度条宽度和文字
 */
function updateCaptureProgress() {
  var fill = document.getElementById('captureProgressFill');
  var text = document.getElementById('captureProgressText');
  if (!fill || !text) return;
  
  if (spatialCaptureTargetFrames === Infinity || spatialCaptureTargetFrames <= 0) {
    fill.style.width = '0%';
    text.textContent = '已采集 ' + totalFramesCollected + ' 帧';
  } else {
    var pct = Math.min(100, Math.round((totalFramesCollected / spatialCaptureTargetFrames) * 100));
    fill.style.width = pct + '%';
    text.textContent = totalFramesCollected + ' / ' + spatialCaptureTargetFrames + ' 帧';
  }
}

/**
 * 采集当前帧数据
 * 调用外部注入的captureCurrentFrame函数获取图像，存入待上传队列
 */
async function collectFrame() {
  // ✅ 流式模式下即使 isBatchProcessing=true 也要继续采集
  // 只有批量模式才需要停止采集
  if (!spatialIsCapturing || (isBatchProcessing && !isInferenceStarted)) return;

  // 调用外部设置的帧采集函数
  var frameData = null;
  if (captureCurrentFrame) {
    frameData = captureCurrentFrame();
  }
  if (!frameData) {
    console.log('[Spatial API] collectFrame: 无帧数据, captureCurrentFrame=' + (captureCurrentFrame ? 'defined' : 'null'));
    return;
  }

  console.log('[Spatial API] collectFrame: 收到帧, image.length=' + (frameData.image ? frameData.image.length : 0));

  // 将帧数据加入队列
  collectedFrames.push(frameData.image);
  spatialFrameCounter++;
  totalFramesCollected++;

  updateCaptureProgress();

  updateStatus({
    frameCount: totalFramesCollected,
    totalFrames: totalFramesCollected,
    collectedFrames: totalFramesCollected,
    targetFrames: spatialCaptureTargetFrames,
    batchId: currentBatchId
  });

  // 每10帧上传一次（流式处理模式）
  if (collectedFrames.length >= 10) {
    await uploadPendingFrames();  // ✅ 添加 await
    
    // 流式处理：第一次上传后启动推理
    if (!isInferenceStarted && !isBatchProcessing && currentBatchId) {
      isInferenceStarted = true;
      addLog('启动流式推理...', 'info');
      await startStreamingInference(currentBatchId);  // ✅ 添加 await
    }
  }

  // 达到目标帧数后自动停止
  if (totalFramesCollected >= spatialCaptureTargetFrames) {
    console.log('[Spatial API] 达到目标帧数 ' + spatialCaptureTargetFrames + '，自动停止采集');
    await uploadPendingFrames();  // ✅ 添加 await
    stopSpatialCapture();
  }

  // 达到目标帧数后提交处理（兼容批量模式）
  if (totalFramesCollected >= spatialCaptureTargetFrames && !isBatchProcessing) {
    submitBatch();
  }
}

/**
 * 上传待处理的帧
 * 从collectedFrames队列中取出最多50帧上传到服务器
 * @returns {Promise<void>}
 */
async function uploadPendingFrames() {
  if (isUploading || collectedFrames.length === 0 || !currentBatchId) return;
  
  // 每次最多上传50帧
  const framesToUpload = collectedFrames.splice(0, 50);
  await uploadFramesToBatchServer(currentBatchId, framesToUpload);
}

/**
 * 启动流式推理
 * 向服务器发送POST请求启动3D重建推理，并启动点云逐帧拉取
 * @param {string} batchId - 批次号
 * @returns {Promise<void>}
 */
async function startStreamingInference(batchId) {
  if (!batchId) return;
  
  isBatchProcessing = true;

  // ✅ 在发送请求前重新计算关键帧间隔，确保传递正确的值到后端
  let currentKeyframeInterval = spatialKeyframeInterval;
  if (spatialMaxImages !== null) {
    if (spatialMaxImages > 320) {
      currentKeyframeInterval = Math.floor((spatialMaxImages + 319) / 320);
    } else {
      currentKeyframeInterval = 1;
    }
  }

  try {
    // 向服务器发送推理启动请求
    const response = await fetch(`${BATCH_SERVER_URL}/batch/${batchId}/start_inference`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ batch_id: batchId, confidence_threshold: 0.1, keyframe_interval: currentKeyframeInterval, max_images: spatialMaxImages })
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      addLog('启动推理失败: ' + errorText, 'err');
      isBatchProcessing = false;
      return;
    }
    
    const result = await response.json();
    if (result.success) {
      addLog('推理已启动', 'ok');
      
      // 启动点云逐帧拉取（使用3D可视化模块的流式加载）
      if (typeof SpatialVisualizer !== 'undefined' && SpatialVisualizer.startFrameByFrameFetch) {
        SpatialVisualizer.startFrameByFrameFetch(batchId, null);
      }
    } else {
      addLog('启动推理失败: ' + (result.error || 'unknown'), 'err');
      isBatchProcessing = false;
    }
  } catch (err) {
    addLog('启动推理请求失败: ' + err.message, 'err');
    console.error('启动推理详细错误:', err);
    isBatchProcessing = false;
  }
}

/** 外部注入的帧采集函数引用 */
var captureCurrentFrame = null;

/**
 * 设置帧采集函数
 * 由外部模块注入实际采集帧的逻辑（通常是从摄像头或视频流获取图像）
 * @param {Function} func - 帧采集函数，返回包含image字段（base64编码）的对象
 */
function setCaptureFrameFunc(func) {
  captureCurrentFrame = func;
}

/**
 * 获取当前采集状态
 * 返回包含所有采集相关状态的快照对象
 * @returns {Object} 采集状态对象
 */
function getSpatialCaptureState() {
  return {
    isCapturing: spatialIsCapturing,
    frameCounter: spatialFrameCounter,
    totalFrames: totalFramesCollected,
    currentBatchId: currentBatchId,
    collectedFrames: collectedFrames.length,
    isBatchProcessing: isBatchProcessing,
    isUploading: isUploading
  };
}

/**
 * 设置采集状态
 * 控制采集开始/停止，停止时自动清除定时器
 * @param {boolean} value - true表示开始采集，false表示停止采集
 */
function setSpatialCapturing(value) {
  spatialIsCapturing = value;
  if (!value) {
    if (spatialCaptureTimer) {
      clearTimeout(spatialCaptureTimer);
      spatialCaptureTimer = null;
    }
    
    // ✅ 停止采集时上传剩余帧并结束推理
    if (currentBatchId) {
      uploadPendingFrames();
      sendFinishInference(currentBatchId);
    }
  }
}

/**
 * 发送结束推理请求
 * 通知服务器完成当前批次的3D重建推理，服务器将释放计算资源
 * @param {string} batchId - 批次号
 * @returns {Promise<void>}
 */
async function sendFinishInference(batchId) {
  if (!batchId) return;
  
  try {
    const response = await fetch(`${BATCH_SERVER_URL}/batch/${batchId}/finish_inference`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ batch_id: batchId })
    });
    
    if (response.ok) {
      const result = await response.json();
      if (result.success) {
        addLog('推理已结束', 'ok');
      } else {
        addLog('结束推理失败: ' + (result.error || 'unknown'), 'err');
      }
    } else {
      const errorText = await response.text();
      addLog('结束推理请求失败: ' + errorText, 'err');
    }
  } catch (err) {
    addLog('结束推理请求异常: ' + err.message, 'err');
    console.error('结束推理详细错误:', err);
  }
}

/**
 * 获取单帧点云和相机位姿并渲染
 * 从服务器获取指定帧的点云数据和相机位姿，更新3D场景显示
 * @param {string} batchId - 批次号
 * @param {number} frameIndex - 帧索引
 * @returns {Promise<void>}
 */
async function fetchAndRenderFrame(batchId, frameIndex) {
  if (!batchId || frameIndex === undefined) return;
  
  try {
    // ✅ 获取点云数据
    const pointCloudResponse = await fetch(`${BATCH_SERVER_URL}/batch/${batchId}/frame/${frameIndex}/point_cloud`);
    
    if (!pointCloudResponse.ok) {
      console.warn(`[Spatial] 帧 ${frameIndex} 点云获取失败: ${pointCloudResponse.status}`);
      return;
    }
    
    const pointCloudData = await pointCloudResponse.json();
    
    if (!pointCloudData.success) {
      console.warn(`[Spatial] 帧 ${frameIndex} 无点云数据`);
      return;
    }
    
    var points_b64 = pointCloudData.points_b64;
    var colors_b64 = pointCloudData.colors_b64;
    var confs_b64 = pointCloudData.confs_b64;
    var total_points = pointCloudData.total_points || 0;

    if (!points_b64) {
      console.warn("[Spatial] frame " + frameIndex + " missing points_b64");
      return;
    }

    var points = base64ToFloat32Array(points_b64);
    var colors = colors_b64 ? base64ToFloat32Array(colors_b64) : new Float32Array(total_points * 3).fill(0.5);
    var confs = confs_b64 ? base64ToFloat32Array(confs_b64) : new Float32Array(total_points).fill(1.0);
    
    // ✅ 获取相机位姿
    const cameraResponse = await fetch(`${BATCH_SERVER_URL}/batch/${batchId}/frame/${frameIndex}/camera`);
    
    if (cameraResponse.ok) {
      const cameraData = await cameraResponse.json();
      if (cameraData.success && cameraData.camera) {
        if (!window.camerasData) window.camerasData = [];
        window.camerasData[frameIndex] = cameraData.camera;
        
        // ✅ 更新相机轨迹线
        if (typeof updateTrajectoryLine === 'function') {
          updateTrajectoryLine();
        }
      }
    }
    
    // ✅ 渲染点云
    if (typeof SpatialVisualizer !== 'undefined' && SpatialVisualizer.addFramePointCloud) {
      SpatialVisualizer.addFramePointCloud(points, colors, confs, frameIndex);
    }
    
    addLog(`帧 ${frameIndex} 点云渲染完成，${points.length / 3} 点`, 'ok');
    
  } catch (err) {
    console.error(`[Spatial] 帧 ${frameIndex} 获取失败:`, err.message);
    addLog(`帧 ${frameIndex} 获取失败: ${err.message}`, 'err');
  }
}

/**
 * 重置采集状态
 * 清除所有计数器、队列和批次信息，恢复初始状态
 */
function resetSpatialState() {
  spatialFrameCounter = 0;
  totalFramesCollected = 0;
  currentBatchId = null;
  isInitialBatch = true;
  collectedFrames = [];
  uploadQueue = [];
  isBatchProcessing = false;
  isUploading = false;
}

// ============== 批次服务 API ==============

/**
 * 生成唯一批次号
 * 格式：batch_YYYYMMDD_HHMMSS（基于当前时间戳）
 * @returns {string} 批次号字符串
 */
function generateBatchId() {
  const now = new Date();
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, '0');
  const d = String(now.getDate()).padStart(2, '0');
  const h = String(now.getHours()).padStart(2, '0');
  const min = String(now.getMinutes()).padStart(2, '0');
  const s = String(now.getSeconds()).padStart(2, '0');
  return `batch_${y}${m}${d}_${h}${min}${s}`;
}

/**
 * 检查批次服务状态
 * 通过HTTP GET请求验证服务器是否可达
 * @returns {Promise<boolean>} 服务是否可达
 */
async function checkBatchServerStatus() {
  try {
    const response = await fetch(`${BATCH_SERVER_URL}/`, {
      method: 'GET',
      timeout: 5000
    });
    if (response.ok) {
      addLog('批次服务已连接', 'ok');
      return true;
    } else {
      console.warn('[Spatial API] 批次服务返回错误状态:', response.status);
      return false;
    }
  } catch (err) {
    console.warn('[Spatial API] 批次服务不可达:', err.message);
    return false;
  }
}

/**
 * 上传帧到批次服务
 * 将采集的图像帧批量上传到RTX3090服务器进行3D重建
 * @param {string} batchId - 批次号
 * @param {string[]} frames - base64编码的图像数组
 * @returns {Promise<Object>} 上传结果 {success, totalUploaded, message}
 */
async function uploadFramesToBatchServer(batchId, frames) {
  if (frames.length === 0) return { success: false };
  
  isUploading = true;
  const formData = new FormData();
  
  frames.forEach((frame, index) => {
    const blob = base64ToBlob(frame, 'image/jpeg');
    const frameNum = spatialFrameCounter - frames.length + index;
    formData.append('files', blob, `frame_${String(frameNum).padStart(3, '0')}.jpg`);
  });
  
  try {
    const response = await fetch(`${BATCH_SERVER_URL}/batch/${batchId}/frames`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      addLog(`上传帧失败 (${response.status}): ${errorData.detail || 'unknown'}`, 'err');
      return { success: false, error: errorData.detail };
    }
    
    const result = await response.json();
    addLog(`上传 ${frames.length} 帧成功`, 'ok');
    return result;
  } catch (err) {
    addLog('上传帧失败: ' + err.message, 'err');
    return { success: false, error: err.message };
  } finally {
    isUploading = false;
  }
}

/**
 * 获取批次状态
 * @param {string} batchId - 批次号
 * @returns {Promise<Object|null>} 批次状态信息
 */
async function getBatchStatus(batchId) {
  try {
    const response = await fetch(`${BATCH_SERVER_URL}/batch/${batchId}/status`);
    return await response.json();
  } catch (err) {
    console.warn('[Spatial API] 获取批次状态失败:', err.message);
    return null;
  }
}

/**
 * 获取批次点云数据
 * @param {string} batchId - 批次号
 * @returns {Promise<Object|null>} 点云数据（包含points、colors等）
 */
async function getBatchPointCloud(batchId) {
  try {
    const response = await fetch(`${BATCH_SERVER_URL}/batch/${batchId}/point_cloud`);
    return await response.json();
  } catch (err) {
    console.warn('[Spatial API] 获取点云失败:', err.message);
    return null;
  }
}

/**
 * Base64 转 Blob
 */
function base64ToBlob(base64, mimeType) {
  const byteString = atob(base64.split(',')[1] || base64);
  const ab = new ArrayBuffer(byteString.length);
  const ia = new Uint8Array(ab);
  for (let i = 0; i < byteString.length; i++) {
    ia[i] = byteString.charCodeAt(i);
  }
  return new Blob([ab], { type: mimeType });
}

// ============== API 模块导出 ==============
window.SpatialApi = {
  init: initConnection,
  setCaptureFrameFunc: setCaptureFrameFunc,
  setCallbacks: setSpatialApiCallbacks,
  getState: getSpatialCaptureState,
  setCapturing: setSpatialCapturing,
  reset: resetSpatialState,
  checkServerStatus: checkBatchServerStatus,
  startStatusCheck: function() { /* 状态检查方法 - 保持向后兼容 */ },
  startContinuousCapture: startContinuousCapture,
  // 批次服务相关
  generateBatchId: generateBatchId,
  uploadFrames: uploadFramesToBatchServer,
  getBatchStatus: getBatchStatus,
  getPointCloud: getBatchPointCloud,
  checkBatchServerStatus: checkBatchServerStatus
};

// ============== 3D 可视化层 (spatial_visualizer.js) ==============
// 负责：Three.js场景初始化、点云渲染、相机位姿可视化、轨迹展示

// Three.js 核心库
let THREE = null;
// 轨道控制器（鼠标旋转/平移/缩放）
let OrbitControls = null;
// PLY点云加载器
let PLYLoader = null;
// Three.js场景对象
let scene = null;
// 3D相机对象
let camera3d = null;
// WebGL渲染器
let renderer = null;
// 动画帧ID（用于cancelAnimationFrame）
let animationId = null;
// 点云渲染对象
let pointCloud = null;
// 点云组（包含所有点云）
let cloudGroup = null;
// 相机视锥体组
let frustumGroup = null;
// 轨迹管状几何体
let trajectoryTube = null;
// 相机视锥体列表
let cameraFrustums = [];
// 统计信息：FPS、顶点数
let visualizerStats = { fps: 0, vertices: 0 };
// 帧时间计数器
let frameTime = 0;
// 帧计数器
let frameCount = 0;
// GUI：降采样步长
let guiDownsample = 5;
/** GUI: point cloud point size (matches viser default) */
let guiPointSize = 0.00001;
/** GUI: confidence threshold for point filtering (same as viser default) */
let guiConfThreshold = 0.7;
// 统计信息更新回调
let onStatsUpdate = null;

/**
 * 动态加载Three.js库
 * 使用import()动态导入，避免阻塞页面加载
 * @returns {Promise<typeof THREE>} Three.js库对象
 */
async function loadThreeJS() {
  if (THREE) return THREE;
  try {
    // 动态导入Three.js核心库和插件
    THREE = await import('three');
    const { OrbitControls: OC } = await import('three/addons/controls/OrbitControls.js');
    OrbitControls = OC;
    const { PLYLoader: PL } = await import('three/addons/loaders/PLYLoader.js');
    PLYLoader = PL;
    console.log('[Spatial Visualizer] Three.js, OrbitControls and PLYLoader loaded successfully');
    return THREE;
  } catch (err) {
    console.error('[Spatial Visualizer] Failed to load Three.js:', err);
    throw err;
  }
}

/**
 * 初始化3D场景
 * 创建Three.js场景、相机、渲染器、网格辅助线、坐标轴等
 * @returns {Promise<void>}
 */
async function init3DScene() {
  if (!THREE) await loadThreeJS();

  const container = document.getElementById('spatialCanvasContainer');
  const canvas = document.getElementById('spatialCanvas');

  if (!container || !canvas) {
    console.error('[Spatial Visualizer] Container or canvas not found');
    return;
  }

  // 创建场景并设置背景色
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xd0d0d0);

  const width = container.clientWidth || 800;
  const height = container.clientHeight || 600;
  
  console.log('[Spatial Visualizer] Container size:', width, 'x', height);
  
  // 直接设置canvas尺寸
  canvas.width = width;
  canvas.height = height;
  
  // 创建透视相机
  camera3d = new THREE.PerspectiveCamera(60, width / height, 0.1, 10000);
  camera3d.position.set(0, 0, 5);
  camera3d.lookAt(0, 0, 0);
  camera3d.up.set(0, 1, 0);

  // 创建WebGL渲染器
  renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: false });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  // 添加网格辅助线
  const gridHelper = new THREE.GridHelper(20, 20, 0xaaaaaa, 0xcccccc);
  scene.add(gridHelper);

  // 添加世界坐标轴（Viser 风格：红X、绿Y、蓝Z）
  const worldAxes = new THREE.AxesHelper(1.0);
  worldAxes.material.linewidth = 3;
  scene.add(worldAxes);

  // 创建点云几何体和材质
  const pointCloudGeometry = new THREE.BufferGeometry();
  const pointCloudMaterial = new THREE.PointsMaterial({
    size: 0.001,
    vertexColors: true,
    sizeAttenuation: true,
    transparent: false,
    opacity: 1.0,
    depthWrite: true,
    depthTest: true
  });
  pointCloud = new THREE.Points(pointCloudGeometry, pointCloudMaterial);
  window.pointCloud = pointCloud;

  // 创建点云组和相机视锥体组
  cloudGroup = new THREE.Group();
  frustumGroup = new THREE.Group();
  cloudGroup.scale.set(-1, -1, 1);
  frustumGroup.scale.set(-1, -1, 1);
  scene.add(cloudGroup);
  scene.add(frustumGroup);

  // 创建轨道控制器（鼠标交互）
  window.controls = new OrbitControls(camera3d, renderer.domElement);
  window.controls.enableDamping = true;
  window.controls.dampingFactor = 0.08;
  window.controls.minDistance = 0.01;
  window.controls.maxDistance = 5000;
  window.controls.target.set(0, 0, 0);

  // ResizeObserver - 更可靠的尺寸监听
  if (SpatialState.resizeObserver) {
    SpatialState.resizeObserver.disconnect();
  }
  SpatialState.resizeObserver = new ResizeObserver(entries => {
    for (let entry of entries) {
      const newWidth = entry.contentRect.width;
      const newHeight = entry.contentRect.height;
      if (camera3d && renderer) {
        camera3d.aspect = newWidth / newHeight;
        camera3d.updateProjectionMatrix();
        renderer.setSize(newWidth, newHeight);
      }
    }
  });
  SpatialState.resizeObserver.observe(container);

  // 启动动画循环
  animate();
  
  // 清理旧的监听器
  window.removeEventListener('resize', onWindowResize);
  
  console.log('[Spatial Visualizer] 3D scene initialized with size:', width, 'x', height);
}

/**
 * 窗口大小变化处理（旧版保留用于兼容）
 */
function onWindowResize() {
  const container = document.getElementById('spatialCanvasContainer');
  if (!container) return;
  
  const width = container.clientWidth || 800;
  const height = container.clientHeight || 600;
  
  if (camera3d) {
    camera3d.aspect = width / height;
    camera3d.updateProjectionMatrix();
  }
  
  if (renderer) {
    renderer.setSize(width, height);
  }
}

function checkAndFixCanvasSize() {
  const container = document.getElementById('spatialCanvasContainer');
  const canvas = document.getElementById('spatialCanvas');
  if (!container || !canvas || !camera3d || !renderer) return;

  const containerWidth = container.clientWidth;
  const containerHeight = container.clientHeight;
  const canvasWidth = canvas.width;
  const canvasHeight = canvas.height;

  if (canvasWidth !== containerWidth || canvasHeight !== containerHeight) {
    canvas.width = containerWidth;
    canvas.height = containerHeight;
    camera3d.aspect = containerWidth / containerHeight;
    camera3d.updateProjectionMatrix();
    renderer.setSize(containerWidth, containerHeight);
    console.log('[Spatial Visualizer] Canvas resized to match container:', containerWidth, 'x', containerHeight);
  }
}

function animate() {
  animationId = requestAnimationFrame(animate);

  // 每10帧检查一次canvas尺寸
  if (frameCount % 10 === 0) {
    checkAndFixCanvasSize();
  }

  const currentTime = performance.now();
  frameCount++;
  if (currentTime - frameTime >= 1000) {
    visualizerStats.fps = frameCount;
    if (onStatsUpdate) {
      onStatsUpdate({ fps: visualizerStats.fps });
    }
    frameCount = 0;
    frameTime = currentTime;
  }

  if (window.controls) {
    window.controls.update();
  }

  if (cameraFollowEnabled && currentCameraTargetPos && currentCameraLookAt) {
    camera3d.position.lerp(currentCameraTargetPos, cameraFollowSmoothFactor);
    window.controls.target.lerp(currentCameraLookAt, cameraFollowSmoothFactor);
  }

  renderer.render(scene, camera3d);
}

function updateCameraTrajectory(cameraPoses) {
  if (!THREE || !scene || !cameraPoses || cameraPoses.length < 2) return;

  if (trajectoryTube) {
    scene.remove(trajectoryTube);
    if (trajectoryTube.geometry) trajectoryTube.geometry.dispose();
    if (trajectoryTube.material) trajectoryTube.material.dispose();
    trajectoryTube = null;
  }

  const points = cameraPoses.map(pose => {
    const pos = pose.position;
    return new THREE.Vector3(pos[0], pos[1], pos[2]);
  });

  if (points.length < 2) return;

  const curve = new THREE.CatmullRomCurve3(points);
  const numSegments = Math.max(20, points.length * 2);
  const tubeRadius = 0.005;
  const geometry = new THREE.TubeGeometry(curve, numSegments, tubeRadius, 8, false);

  const colors = [];
  for (let i = 0; i < geometry.attributes.position.count; i++) {
    const t = i / geometry.attributes.position.count;
    const hue = t * 0.8 + 0.15;
    const color = new THREE.Color().setHSL(hue, 0.9, 0.6);
    colors.push(color.r, color.g, color.b);
  }
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

  const material = new THREE.MeshBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.8
  });

  trajectoryTube = new THREE.Mesh(geometry, material);
  scene.add(trajectoryTube);
}

function updateCameraFrustums(cameraPoses) {
  if (!THREE || !scene) return;

  cameraFrustums.forEach(obj => {
    frustumGroup.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      if (Array.isArray(obj.material)) {
        obj.material.forEach(m => m.dispose());
      } else {
        obj.material.dispose();
      }
    }
  });
  cameraFrustums = [];

  if (!cameraPoses || cameraPoses.length === 0) return;

  // 绘制相机轨迹线
  if (cameraPoses.length > 1) {
    const trajectoryPoints = [];
    cameraPoses.forEach(pose => {
      const pos = pose.position;
      trajectoryPoints.push(new THREE.Vector3(pos[0], pos[1], pos[2]));
    });

    const curve = new THREE.CatmullRomCurve3(trajectoryPoints);
    const curvePoints = curve.getPoints(cameraPoses.length * 2);
    const trajectoryGeom = new THREE.BufferGeometry().setFromPoints(curvePoints);
    const trajectoryMat = new THREE.LineBasicMaterial({ 
      color: 0xff6b6b, 
      transparent: true, 
      opacity: 0.6
    });
    const trajectoryLine = new THREE.Line(trajectoryGeom, trajectoryMat);
    frustumGroup.add(trajectoryLine);
    cameraFrustums.push(trajectoryLine);
  }

  const step = Math.max(1, Math.floor(cameraPoses.length / 60));

  cameraPoses.forEach((pose, index) => {
    if (index % step !== 0 && index !== cameraPoses.length - 1) return;

    const pos = pose.position;
    const camPos = new THREE.Vector3(pos[0], pos[1], pos[2]);

    // Render camera coordinate axes (same convention as viser)
    if (pose.rotation) {
      const R = pose.rotation;
      const xAxis = new THREE.Vector3(R[0][0], R[1][0], R[2][0]);
      const yAxis = new THREE.Vector3(R[0][1], R[1][1], R[2][1]);
      const zAxis = new THREE.Vector3(R[0][2], R[1][2], R[2][2]);

      const axisLen = 0.1;
      const axisRadius = 0.004;

      // X 轴（红色）
      const xGeom = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLen, 8);
      xGeom.translate(0, axisLen / 2, 0);
      xGeom.rotateZ(-Math.PI / 2);
      const xMat = new THREE.MeshBasicMaterial({ color: 0xff3333, transparent: true, opacity: 0.9 });
      const xAxisMesh = new THREE.Mesh(xGeom, xMat);
      xAxisMesh.position.copy(camPos);
      xAxisMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), xAxis);
      frustumGroup.add(xAxisMesh);
      cameraFrustums.push(xAxisMesh);

      // Y 轴（绿色）
      const yGeom = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLen, 8);
      yGeom.translate(0, axisLen / 2, 0);
      const yMat = new THREE.MeshBasicMaterial({ color: 0x33ff33, transparent: true, opacity: 0.9 });
      const yAxisMesh = new THREE.Mesh(yGeom, yMat);
      yAxisMesh.position.copy(camPos);
      yAxisMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), yAxis);
      frustumGroup.add(yAxisMesh);
      cameraFrustums.push(yAxisMesh);

      // Z 轴（蓝色）
      const zGeom = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLen, 8);
      zGeom.translate(0, axisLen / 2, 0);
      zGeom.rotateX(Math.PI / 2);
      const zMat = new THREE.MeshBasicMaterial({ color: 0x3366ff, transparent: true, opacity: 0.9 });
      const zAxisMesh = new THREE.Mesh(zGeom, zMat);
      zAxisMesh.position.copy(camPos);
      zAxisMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), zAxis);
      frustumGroup.add(zAxisMesh);
      cameraFrustums.push(zAxisMesh);

      // 相机中心点
      const centerGeom = new THREE.SphereGeometry(0.0075, 8, 8);
      const centerMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 });
      const centerMesh = new THREE.Mesh(centerGeom, centerMat);
      centerMesh.position.copy(camPos);
      frustumGroup.add(centerMesh);
      cameraFrustums.push(centerMesh);
    } else {
      // 没有旋转数据时，只显示位置标记（兼容旧数据）
      const hue = (index / cameraPoses.length) * 0.8 + 0.15;
      const markerColor = new THREE.Color().setHSL(hue, 0.8, 0.6);
      const markerGeom = new THREE.SphereGeometry(0.015, 12, 12);
      const markerMat = new THREE.MeshBasicMaterial({ color: markerColor, transparent: true, opacity: 0.9 });
      const marker = new THREE.Mesh(markerGeom, markerMat);
      marker.position.copy(camPos);
      frustumGroup.add(marker);
      cameraFrustums.push(marker);
    }
  });
}

function reset3DCamera() {
  if (!camera3d || !window.controls) return;

  var activeCloud = mergedPointCloud || pointCloud;
  if (activeCloud && activeCloud.geometry && activeCloud.geometry.attributes.position) {
    const bbox = new THREE.Box3().setFromObject(activeCloud);
    const center = new THREE.Vector3();
    bbox.getCenter(center);
    const size = new THREE.Vector3();
    bbox.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);

    camera3d.position.set(
      center.x + maxDim * 0.7,
      center.y + maxDim * 0.5,
      center.z + maxDim * 0.7
    );
    camera3d.lookAt(center);
    window.controls.target.copy(center);
  } else {
    if (sceneScale > 0) {
      fitCameraToScene();
    } else {
      camera3d.position.set(3, 2, 3);
      camera3d.lookAt(0, 0, 0);
      window.controls.target.set(0, 0, 0);
    }
  }
  window.controls.update();
}

/**
 * Use backend-provided sceneCenter and sceneScale to position camera.
 * Called after fetchMetadata completes.
 */
function fitCameraToScene() {
  if (!camera3d || !window.controls) return;

  var center = new THREE.Vector3(sceneCenter[0], sceneCenter[1], sceneCenter[2]);
  var dist = Math.max(sceneScale * 1.5, 1.0);

  camera3d.position.set(
    center.x + dist * 0.7,
    center.y + dist * 0.5,
    center.z + dist * 0.7
  );
  camera3d.lookAt(center);
  window.controls.target.copy(center);
  window.controls.update();

  // Dynamically adjust camera near/far based on scene scale
  if (sceneScale > 0) {
    camera3d.near = Math.max(0.01, sceneScale * 0.001);
    camera3d.far = Math.max(100, sceneScale * 10);
    camera3d.updateProjectionMatrix();
  }

  // Dynamically resize grid and axes helper
  if (sceneScale > 0) {
    var gridSize = Math.pow(10, Math.ceil(Math.log10(sceneScale)));
    scene.children.forEach(function(child) {
      if (child.isGridHelper) {
        scene.remove(child);
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      }
    });
    var gridHelper = new THREE.GridHelper(gridSize, Math.round(gridSize), 0xaaaaaa, 0xcccccc);
    scene.add(gridHelper);

    scene.children.forEach(function(child) {
      if (child.isAxesHelper) {
        scene.remove(child);
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      }
    });
    var axesSize = Math.pow(10, Math.floor(Math.log10(sceneScale * 0.5)));
    var worldAxes = new THREE.AxesHelper(Math.max(0.1, axesSize));
    worldAxes.material.linewidth = 3;
    scene.add(worldAxes);
  }

  addLog("Camera fitted: center=[" + sceneCenter.map(function(v) { return v.toFixed(2); }).join(",") + "], scale=" + sceneScale.toFixed(2), "ok");
}


function setViewDirection(direction) {
  if (!camera3d || !window.controls) return;

  let center = new THREE.Vector3(0, 0, 0);
  let scale = 3;

  var activeCloud = mergedPointCloud || pointCloud;
  if (activeCloud && activeCloud.geometry && activeCloud.geometry.attributes.position) {
    const bbox = new THREE.Box3().setFromObject(activeCloud);
    bbox.getCenter(center);
    const size = new THREE.Vector3();
    bbox.getSize(size);
    scale = Math.max(size.x, size.y, size.z) * 1.5;
  }

  camera3d.position.set(
    center.x + direction[0] * scale,
    center.y + direction[1] * scale,
    center.z + direction[2] * scale
  );
  camera3d.lookAt(center);
  window.controls.target.copy(center);
  window.controls.update();
}

function setViewToFirstCamera() {
  if (cameraFrustums.length === 0) return;

  const firstFrustum = cameraFrustums.find(obj => obj instanceof THREE.Mesh);
  if (firstFrustum) {
    camera3d.position.set(
      firstFrustum.position.x + 0.5,
      firstFrustum.position.y + 0.5,
      firstFrustum.position.z + 0.5
    );
    camera3d.lookAt(firstFrustum.position);
    window.controls.target.copy(firstFrustum.position);
    window.controls.update();
  }
}

function togglePointCloud() {
  // 单一点云对象模式
  if (mergedPointCloud) {
    mergedPointCloud.visible = !mergedPointCloud.visible;
    const btn = document.getElementById('togglePointCloudBtn');
    if (btn) btn.textContent = mergedPointCloud.visible ? '☁️ 点云' : '☁️ 点云(隐藏)';
    return;
  }
  
  // 兼容旧的独立对象逻辑
  Object.keys(framePointCloudObjects).forEach(function(key) {
    var obj = framePointCloudObjects[key];
    obj.visible = !obj.visible;
  });
  if (pointCloud) {
    pointCloud.visible = !pointCloud.visible;
  }
  const btn = document.getElementById('togglePointCloudBtn');
  if (btn) btn.textContent = pointCloud ? (pointCloud.visible ? '☁️ 点云' : '☁️ 点云(隐藏)') : '☁️ 点云';
}

/**
 * 切换轨迹线显示
 * 连接所有相机位置形成移动轨迹
 */
function toggleTrajectory() {
  if (trajectoryTube) {
    trajectoryTube.visible = !trajectoryTube.visible;
    const btn = document.getElementById('viewPathBtn');
    if (btn) btn.textContent = trajectoryTube.visible ? '🛤️ 轨迹' : '🛤️ 轨迹(隐藏)';
  }
}

/**
 * 设置可视化器回调函数
 * @param {Object} callbacks - 回调函数对象
 * @param {Function} callbacks.onStatsUpdate - 统计信息更新回调（FPS、顶点数）
 */
function setVisualizerCallbacks(callbacks) {
  if (callbacks.onStatsUpdate) onStatsUpdate = callbacks.onStatsUpdate;
}

/**
 * 获取WebGL渲染器
 * @returns {THREE.WebGLRenderer} 渲染器对象
 */
function getRenderer() {
  return renderer;
}

/**
 * 获取Three.js场景
 * @returns {THREE.Scene} 场景对象
 */
function getScene() {
  return scene;
}

/**
 * 获取3D相机
 * @returns {THREE.PerspectiveCamera} 相机对象
 */
function getCamera() {
  return camera3d;
}

/**
 * 获取当前场景数据（点云+相机位姿）
 * 用于导出或其他模块访问
 * @returns {Object} 包含points、colors、cameraPoses的对象
 */
function getCurrentSceneData() {
  var data = {
    points: [],
    colors: [],
    cameraPoses: []
  };

  var activeCloud = mergedPointCloud || pointCloud;
  if (activeCloud && activeCloud.geometry && activeCloud.geometry.attributes.position) {
    var posAttr = activeCloud.geometry.attributes.position;
    var colAttr = activeCloud.geometry.attributes.color;
    var arr = posAttr.array;
    for (var i = 0; i < arr.length; i++) {
      data.points.push(arr[i]);
    }
    if (colAttr) {
      var colArr = colAttr.array;
      for (var i = 0; i < colArr.length; i++) {
        data.colors.push(Math.round(colArr[i] * 255));
      }
    }
  }

  if (cameraFrustums.length > 0) {
    cameraFrustums.forEach(function(obj) {
      if (obj.position) {
        data.cameraPoses.push({
          position: [obj.position.x, obj.position.y, obj.position.z]
        });
      }
    });
  }

  return data;
}

// ============== 3D 可视化模块导出 ==============
/** 3D可视化模块对外暴露的所有接口 */
window.SpatialVisualizer = {
  init: init3DScene,                            // 初始化3D场景
  resetCamera: reset3DCamera,                   // 重置相机位置
  setViewDirection: setViewDirection,           // 设置相机视角方向
  setViewToFirstCamera: setViewToFirstCamera,   // 切换到第一个相机视角
  togglePointCloud: togglePointCloud,           // 切换点云显示
  toggleTrajectory: toggleTrajectory,           // 切换轨迹显示
  setCallbacks: setVisualizerCallbacks,         // 设置回调函数
  getRenderer: getRenderer,                     // 获取渲染器
  getScene: getScene,                           // 获取场景
  getCamera: getCamera,                         // 获取相机
  getCurrentSceneData: getCurrentSceneData,     // 获取当前场景数据
  toggleCameraFrustums: toggleVisualizerCameraFrustums,           // 切换相机视锥体
  exportToGLB: exportVisualizerToGLB,           // 导出为GLB文件
  startFrameByFrameFetch: startFrameByFrameFetch,                 // 启动逐帧拉取
  updateAccumulatedPointCloud: updateAccumulatedPointCloud,       // 更新累积点云
  fetchManifest: fetchManifest,                 // 获取清单文件
  fetchExtrinsics: fetchExtrinsics,             // 获取外参
  fetchIntrinsics: fetchIntrinsics,             // 获取内参
  fetchDepth: fetchDepth,                       // 获取深度图
  fetchFrameImage: fetchFrameImage,             // 获取帧图像
  fetchAllExtraData: fetchAllExtraData,         // 获取所有额外数据
  enableCameraFollow: enableCameraFollow,       // 启用相机跟随
  disableCameraFollow: disableCameraFollow,     // 禁用相机跟随
  setMaxVisibleFrames: function(value) { maxVisibleFrames = value; updateVisibleFrames(); },  // 设置最大可见帧数
  setCameraFollowDistance: function(value) { cameraFollowDistance = value; },  // 设置相机跟随距离
  setCameraFollowSmoothFactor: function(value) { cameraFollowSmoothFactor = value; }  // 设置相机跟随平滑因子
};

// ============== 逐帧点云加载状态 ==============
/** 当前完整的点云数据（所有帧累积） */
let currentPointCloudData = null;
/** 累积的点云坐标（扁平数组，每3个元素为一个xyz坐标） */
let accumulatedPoints = [];
/** 累积的点云颜色（扁平数组，每3个元素为一个rgb颜色） */
let accumulatedColors = [];
/** 累积的点云置信度（每1个元素为一个置信值） */
let accumulatedConfs = [];
/** 每帧独立点云数据：[{positions: Float32Array, colors: Float32Array, confs: Float32Array}] */
let framePointClouds = [];
/** 可用的总帧数 */
let totalFramesAvailable = 0;
/** 当前正在加载的帧索引 */
let currentFetchFrame = 0;
/** 是否正在拉取帧数据 */
let isFetchingFrames = false;
/** 当前拉取使用的批次号 */
let fetchBatchId = null;
/** 每帧点云在累积数组中的索引范围 [{start: 起始索引, end: 结束索引}] */
let frameRanges = [];
/** 单一点云对象（合并所有帧，与 live_viewer.py 一致） */
let mergedPointCloud = null;
/** 每帧独立的 THREE.Points 对象（保留用于兼容旧逻辑） */
let framePointCloudObjects = {};
let maxVisibleFrames = Infinity;
let cameraFollowEnabled = false;
let currentFollowFrameIndex = -1;
let cameraFollowDistance = 0.8;
let cameraFollowSmoothFactor = 0.15;
let currentCameraTargetPos = null;
let currentCameraLookAt = null;
let smoothedRawCamPos = null;
let smoothedRawLookDir = null;
const POSE_SMOOTH_FACTOR = 0.25;
/** 场景中心坐标（从metadata.json获取） */
let sceneCenter = [0, 0, 0];
/** 场景缩放比例（从metadata.json获取） */
let sceneScale = 1.0;
/** 相机位姿数据（cameras.json内容） */
let camerasData = null;
/** 元数据（metadata.json内容） */
let metadata = null;

// ✅ 逐步更新相机视锥体和轨迹（每加载一帧就添加一个相机）
function updateCameraFrustumsIncremental(frameIndex) {
  if (!THREE || !scene || !camerasData || !camerasData[frameIndex]) return;
  
  const cam = camerasData[frameIndex];
  if (!(cam.R_c2w || cam.R_w2c) || !(cam.t_c2w || cam.t_w2c)) return;
  
  updateTrajectoryLine();
}

let cameraTrajectoryLine = null;

// 更新轨迹线
function updateTrajectoryLine() {
  if (cameraTrajectoryLine) {
    frustumGroup.remove(cameraTrajectoryLine);
    if (cameraTrajectoryLine.geometry) cameraTrajectoryLine.geometry.dispose();
    if (cameraTrajectoryLine.material) cameraTrajectoryLine.material.dispose();
    cameraTrajectoryLine = null;
  }
  
  const validCameras = [];
  for (let i = 0; i < camerasData.length; i++) {
    const c = camerasData[i];
    if (c && (c.t_c2w || c.t_w2c) && Array.isArray(c.t_c2w || c.t_w2c) && (c.t_c2w || c.t_w2c).length === 3) {
      validCameras.push(c);
    }
  }
  if (validCameras.length < 2) return;
  
  const trajectoryPoints = [];
  // Use raw camera positions (same coordinate system as point cloud and viser)
  for (let i = 0; i < validCameras.length; i++) {
    const cam = validCameras[i];
    const t = cam.t_c2w || cam.t_w2c;
    trajectoryPoints.push(new THREE.Vector3(t[0], t[1], t[2]));
  }
  
  const curve = new THREE.CatmullRomCurve3(trajectoryPoints);
  const curvePoints = curve.getPoints(validCameras.length * 3);
  const trajectoryGeom = new THREE.BufferGeometry().setFromPoints(curvePoints);
  const trajectoryMat = new THREE.LineBasicMaterial({ color: 0xff3333, linewidth: 2, transparent: true, opacity: 1.0 });
  cameraTrajectoryLine = new THREE.Line(trajectoryGeom, trajectoryMat);
  frustumGroup.add(cameraTrajectoryLine);
}

// ✅ 跳转到指定帧的相机位置
function flyToCamera(frameIndex) {
  if (!camera3d || !window.controls || !camerasData || frameIndex >= camerasData.length) return;
  
  const cam = camerasData[frameIndex];
  if (!cam) return;
  
  const t_raw = cam.t_c2w || cam.t_w2c;
  const R_raw = (cam.R_c2w || cam.R_w2c).flat();
  
  // Camera world position — use C2W translation directly (same as viser)
  const camPos = new THREE.Vector3(t_raw[0], t_raw[1], t_raw[2]);
  
  // Camera forward direction from rotation matrix (same as viser)
  const forward = new THREE.Vector3(R_raw[2], R_raw[5], R_raw[8]);
  forward.normalize();
  
  const viewPos = camPos.clone().addScaledVector(forward, cameraFollowDistance);
  const lookAtPoint = camPos.clone().addScaledVector(forward, 1.0);
  
  animateCamera(viewPos, lookAtPoint);
}

// 平滑动画过渡相机位置
function animateCamera(targetPos, targetLookAt) {
  if (!camera3d || !window.controls) return;
  
  const startPos = camera3d.position.clone();
  const startTarget = window.controls.target.clone();
  const duration = 500; // 500ms 动画时长
  const startTime = performance.now();
  
  function animate(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    
    // 使用缓动函数
    const eased = 1 - Math.pow(1 - progress, 3);
    
    // 插值位置
    camera3d.position.lerpVectors(startPos, targetPos, eased);
    window.controls.target.lerpVectors(startTarget, targetLookAt, eased);
    window.controls.update();
    
    if (progress < 1) {
      requestAnimationFrame(animate);
    }
  }
  
  requestAnimationFrame(animate);
}

function enableCameraFollow() {
  cameraFollowEnabled = true;
  currentCameraTargetPos = null;
  currentCameraLookAt = null;
  smoothedRawCamPos = null;
  smoothedRawLookDir = null;
  addLog('相机跟随已启用', 'ok');
}

function disableCameraFollow() {
  cameraFollowEnabled = false;
  currentCameraTargetPos = null;
  currentCameraLookAt = null;
  smoothedRawCamPos = null;
  smoothedRawLookDir = null;
  currentFollowFrameIndex = -1;
  
  var imgEl = document.getElementById('frameImagePreview');
  var labelEl = document.getElementById('frameImageLabel');
  if (imgEl) imgEl.style.display = 'none';
  if (labelEl) labelEl.style.display = 'none';
  
  addLog('相机跟随已禁用', 'ok');
}

function updateCameraFollow(frameIndex) {
  if (!cameraFollowEnabled || !camera3d || !window.controls || !camerasData || frameIndex >= camerasData.length) return;
  
  const cam = camerasData[frameIndex];
  if (!cam) return;
  
  const t_raw = cam.t_c2w || cam.t_w2c;
  const R_raw = (cam.R_c2w || cam.R_w2c).flat();
  
  // Camera world position — use C2W translation directly (same as viser)
  const camWorldPos = new THREE.Vector3(t_raw[0], t_raw[1], t_raw[2]);
  
  // Camera forward direction from rotation matrix (same as viser)
  const forward = new THREE.Vector3(R_raw[2], R_raw[5], R_raw[8]);
  forward.normalize();
  
  // 平滑处理
  if (!smoothedRawCamPos) {
    smoothedRawCamPos = camWorldPos.clone();
    smoothedRawLookDir = forward.clone();
  } else {
    smoothedRawCamPos.lerp(camWorldPos, POSE_SMOOTH_FACTOR);
    smoothedRawLookDir.lerp(forward, POSE_SMOOTH_FACTOR).normalize();
  }
  
  // 视角位置 = 相机位置，注视点 = 相机位置 + 朝向方向
  const followPos = smoothedRawCamPos.clone();
  const lookAtPoint = smoothedRawCamPos.clone().addScaledVector(smoothedRawLookDir, 1.0);
  
  currentCameraTargetPos = followPos;
  currentCameraLookAt = lookAtPoint;
  
  if (currentFollowFrameIndex !== frameIndex) {
    currentFollowFrameIndex = frameIndex;
    updateFrameImagePreview(fetchBatchId, frameIndex);
  }
}

function toggleVisualizerCameraFrustums(visible) {
  if (frustumGroup) frustumGroup.visible = visible;
}

function exportVisualizerToGLB() {
  return new Promise((resolve, reject) => {
    try {
      const sceneData = getCurrentSceneData();
      
      const glbData = {
        points: sceneData.points,
        colors: sceneData.colors,
        cameraPoses: sceneData.cameraPoses
      };
      
      const blob = new Blob([JSON.stringify(glbData)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const filename = 'point_cloud_' + new Date().toISOString().slice(0,19).replace(/[:-]/g,'') + '.json';
      
      resolve({
        success: true,
        url: url,
        filename: filename
      });
    } catch (err) {
      reject({
        success: false,
        error: err.message
      });
    }
  });
}

async function startFrameByFrameFetch(batchId, totalFrames) {
  // 隐藏批次模式点云（使用单一点云对象模式）
  if (pointCloud) {
    pointCloud.visible = false;
  }
  
  // 清理旧的单一点云对象
  if (mergedPointCloud) {
    cloudGroup.remove(mergedPointCloud);
    mergedPointCloud.geometry.dispose();
    mergedPointCloud.material.dispose();
    mergedPointCloud = null;
  }
  
  // 清理旧的独立对象
  Object.keys(framePointCloudObjects).forEach(function(key) {
    var obj = framePointCloudObjects[key];
    if (obj) {
      cloudGroup.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    }
  });
  framePointCloudObjects = {};
  
  // 重置状态
  framePointCloudObjects = {};
  framePointClouds = [];
  accumulatedPoints = [];
  accumulatedColors = [];
  frameRanges = [];
  totalFramesAvailable = totalFrames;
  currentFetchFrame = 0;
  isFetchingFrames = true;
  fetchBatchId = batchId;
  camerasData = [];
  
  // ✅ 先加载 metadata，后续需要用它判断是否做坐标变换
  if (!metadata) {
    await fetchMetadata(batchId);
  }
  
  enableCameraFollow();
  
  if (totalFrames) {
    addLog('开始逐帧拉取点云，共 ' + totalFrames + ' 帧', 'info');
  } else {
    addLog('开始流式拉取点云...', 'info');
  }
  
  // ✅ 流式模式：启动轮询
  if (!totalFrames) {
    startStreamingFetchLoop();
  } else {
    await fetchNextFrame();
    // 点云拉取完成后获取额外数据
    await fetchAllExtraData(batchId, totalFrames);
  }
}

function loadPLY(url) {
  return new Promise((resolve, reject) => {
    if (!PLYLoader) { reject(new Error('PLYLoader not available')); return; }
    new PLYLoader().load(url, resolve, undefined, reject);
  });
}

async function fetchNextFrame() {
  if (!isFetchingFrames) {
    isFetchingFrames = false;
    var totalFrames = Object.keys(framePointCloudObjects).length;
    addLog('所有帧点云拉取完成，共 ' + totalFrames + ' 帧', 'ok');
    
    disableCameraFollow();
    
    return;
  }
  
  // ✅ 批量模式：检查是否达到总帧数
  if (totalFramesAvailable && currentFetchFrame >= totalFramesAvailable) {
    isFetchingFrames = false;
    var totalFrames = Object.keys(framePointCloudObjects).length;
    addLog('所有帧点云拉取完成，共 ' + totalFrames + ' 帧', 'ok');
    disableCameraFollow();
    return;
  }
  
  // ✅ 流式模式：先检查是否有新帧处理完成
  let hasNewFrame = false;
  let currentStatus = 'unknown';
  let currentProcessedFrames = 0;
  
  try {
    const statusResponse = await fetch(`${BATCH_SERVER_URL}/batch/${fetchBatchId}/status`);
    if (statusResponse.ok) {
      const statusData = await statusResponse.json();
      currentStatus = statusData.status || 'unknown';
      currentProcessedFrames = statusData.processed_frames || 0;
      
      // 检查是否有新帧处理完成
      if (currentProcessedFrames > currentFetchFrame) {
        hasNewFrame = true;
        console.log(`[流式拉取] 检测到新帧: processed_frames=${currentProcessedFrames}, currentFetchFrame=${currentFetchFrame}`);
      }
      // 如果推理完成但还有帧没拉取，继续拉取
      if (currentStatus === 'completed' && currentFetchFrame < currentProcessedFrames) {
        hasNewFrame = true;
        console.log(`[流式拉取] 推理完成，继续拉取剩余帧: currentFetchFrame=${currentFetchFrame}, processed_frames=${currentProcessedFrames}`);
      }
    }
  } catch (err) {
    console.warn('检查状态失败:', err.message);
  }
  
  if (!hasNewFrame && !totalFramesAvailable) {
    if (currentStatus === 'completed' && currentFetchFrame >= currentProcessedFrames) {
      isFetchingFrames = false;
      var totalFrames = Object.keys(framePointCloudObjects).length;
      addLog('所有帧点云拉取完成，共 ' + totalFrames + ' 帧', 'ok');
      disableCameraFollow();
      return;
    }
    
    console.log(`[流式拉取] 暂无新帧，等待中... status=${currentStatus}, processed_frames=${currentProcessedFrames}, currentFetchFrame=${currentFetchFrame}`);
    if (isFetchingFrames) {
      setTimeout(fetchNextFrame, 500);
    }
    return;
  }
  
  // ✅ 流式模式：有新帧或批量模式：继续拉取当前帧
  if (!totalFramesAvailable) {
    addLog('检测到新帧，开始拉取帧 ' + currentFetchFrame, 'info');
  }
  
  try {
    const [pointCloudResponse, cameraResponse] = await Promise.all([
      fetch(BATCH_SERVER_URL + '/batch/' + fetchBatchId + '/frame/' + currentFetchFrame + '/point_cloud'),
      fetch(BATCH_SERVER_URL + '/batch/' + fetchBatchId + '/frame/' + currentFetchFrame + '/camera')
    ]);
    
    // ✅ 检查点云请求是否成功
    if (!pointCloudResponse.ok) {
      // 404表示帧还没处理完，等待后重试同一帧
      if (pointCloudResponse.status === 404) {
        addLog('帧 ' + currentFetchFrame + ' 正在处理中，等待...', 'info');
        if (isFetchingFrames) {
          setTimeout(fetchNextFrame, 500);
        }
        return;
      }
      addLog('帧 ' + currentFetchFrame + ' 请求失败: ' + pointCloudResponse.status, 'err');
      currentFetchFrame++;
      if (isFetchingFrames) {
        setTimeout(fetchNextFrame, 100);
      }
      return;
    }
    
    if (cameraResponse && cameraResponse.ok) {
      const cameraResult = await cameraResponse.json();
      if (cameraResult.success && cameraResult.camera) {
        camerasData[currentFetchFrame] = cameraResult.camera;
      }
    }
    
    // ✅ 从 API 获取点云数据
    const pointCloudResult = await pointCloudResponse.json();
    if (pointCloudResult.success) {
      const { points_b64, colors_b64, confs_b64, total_points } = pointCloudResult;
      const numVertices = total_points;
      
      // 解码 Base64 编码的数据
      const flatPositions = base64ToFloat32Array(points_b64);
      const flatColorsArr = colors_b64 ? base64ToFloat32Array(colors_b64) : new Float32Array(numVertices * 3).fill(0.5);
      const flatConfsArr = confs_b64 ? base64ToFloat32Array(confs_b64) : new Float32Array(numVertices).fill(1.0);
      

      
      framePointClouds[currentFetchFrame] = {
        positions: flatPositions,
        colors: flatColorsArr,
        confs: flatConfsArr
      };
      
      addFramePointCloudToScene(currentFetchFrame);
      
      try {
        updateCameraFrustumsIncremental(currentFetchFrame);
      } catch (e) {
        console.warn('Camera frustum update failed for frame ' + currentFetchFrame + ': ' + e.message);
      }
      updateTrajectoryLine();
      
      var totalRenderedFrames = Object.keys(framePointClouds).length;
      addLog('帧 ' + currentFetchFrame + (totalFramesAvailable ? '/' + totalFramesAvailable : '') + ' 点云加载完成，共 ' + numVertices + ' 点，累计 ' + totalRenderedFrames + ' 帧', 'ok');
      
      currentFetchFrame++;
      
      // ✅ 流式模式：先检查状态再继续拉取，避免频繁请求
      // 批量模式：立即继续拉取下一帧
      if (isFetchingFrames) {
        if (totalFramesAvailable) {
          setTimeout(fetchNextFrame, 50);
        } else {
          // 流式模式：重新检查状态，等待新帧
          setTimeout(fetchNextFrame, 100);
        }
      }
    } else {
      addLog('帧 ' + currentFetchFrame + ' API 响应不成功', 'info');
      currentFetchFrame++;
      if (isFetchingFrames) {
        setTimeout(fetchNextFrame, 100);
      }
    }
  } catch (err) {
    addLog('帧 ' + currentFetchFrame + ' 加载失败: ' + err.message, 'err');
    currentFetchFrame++;
    if (isFetchingFrames) {
      setTimeout(fetchNextFrame, 100);
    }
  }
}

/**
 * 启动流式拉取循环（用于实时推理场景）
 */
function startStreamingFetchLoop() {
  if (!isFetchingFrames || !fetchBatchId) return;
  
  // ✅ 简化逻辑：直接调用 fetchNextFrame，它会自己循环
  fetchNextFrame();
  
  // 设置独立的状态监控定时器（用于检测推理完成）
  const monitorStatus = async () => {
    if (!isFetchingFrames) return;
    
    try {
      const statusResponse = await fetch(`${BATCH_SERVER_URL}/batch/${fetchBatchId}/status`);
      if (statusResponse.ok) {
        const statusData = await statusResponse.json();
        if (statusData.status === 'completed') {
          // 推理完成，标记为完成状态
          addLog('推理完成，等待拉取剩余帧...', 'info');
        }
      }
    } catch (err) {
      console.warn('状态监控失败:', err.message);
    }
    
    if (isFetchingFrames) {
      setTimeout(monitorStatus, 2000);
    }
  };
  
  // 启动状态监控
  setTimeout(monitorStatus, 1000);
}

function addFramePointCloudToScene(frameIndex) {
  if (!THREE || !scene || !cloudGroup) return;

  const frameData = framePointClouds[frameIndex];
  if (!frameData) return;

  const positions = frameData.positions;
  const colors = frameData.colors;
  const confs = frameData.confs || new Float32Array(positions.length / 3);
  const numPoints = positions.length / 3;
  if (numPoints === 0) return;

  // 优化：单次遍历完成 isfinite + 置信度过滤 + 下采样
  const stride = Math.max(1, guiDownsample);
  const confThreshold = guiConfThreshold;
  
  // 预分配数组（预估大小，避免多次扩容）
  var maxSize = Math.ceil(numPoints / stride);
  var filteredPositions = new Float32Array(maxSize * 3);
  var filteredColors = new Float32Array(maxSize * 3);
  var count = 0;
  
  for (var i = 0; i < numPoints; i++) {
    // Step 1: isfinite 过滤
    var x = positions[i * 3];
    var y = positions[i * 3 + 1];
    var z = positions[i * 3 + 2];
    if (!isFinite(x) || !isFinite(y) || !isFinite(z)) continue;
    
    // Step 2: 置信度阈值过滤
    if (confs[i] <= confThreshold) continue;
    
    // Step 3: 下采样（每隔 stride 个点取一个）
    if (i % stride !== 0) continue;
    
    filteredPositions[count * 3] = x;
    filteredPositions[count * 3 + 1] = y;
    filteredPositions[count * 3 + 2] = z;
    filteredColors[count * 3] = colors[i * 3];
    filteredColors[count * 3 + 1] = colors[i * 3 + 1];
    filteredColors[count * 3 + 2] = colors[i * 3 + 2];
    count++;
  }
  
  if (count === 0) return;
  
  // 截取实际大小的数组
  filteredPositions = filteredPositions.slice(0, count * 3);
  filteredColors = filteredColors.slice(0, count * 3);
  
  // 累积到全局数组
  var startIdx = accumulatedPoints.length / 3;
  for (var j = 0; j < count; j++) {
    accumulatedPoints.push(filteredPositions[j * 3], filteredPositions[j * 3 + 1], filteredPositions[j * 3 + 2]);
    accumulatedColors.push(filteredColors[j * 3], filteredColors[j * 3 + 1], filteredColors[j * 3 + 2]);
  }
  
  // 记录该帧在累积数组中的范围
  frameRanges[frameIndex] = {
    start: startIdx,
    end: startIdx + count
  };

  // 更新单一点云对象
  updateMergedPointCloud();

  visualizerStats.vertices = accumulatedPoints.length / 3;
  if (onStatsUpdate) {
    onStatsUpdate({ vertices: visualizerStats.vertices });
  }

  if (cameraFollowEnabled) {
    updateCameraFollow(frameIndex);
  }
}

function updateMergedPointCloud() {
  if (!THREE || !scene || !cloudGroup) return;
  
  const numPoints = accumulatedPoints.length / 3;
  if (numPoints === 0) return;
  
  // 安全检查：验证数据不包含 NaN
  for (let i = 0; i < accumulatedPoints.length; i++) {
    if (!isFinite(accumulatedPoints[i])) {
      console.warn('[Spatial] Found NaN/Inf in accumulatedPoints at index', i);
      return;
    }
  }
  
  // 优化：使用 Float32Array.from() 直接转换，避免手动循环
  const positionsAttr = new THREE.BufferAttribute(Float32Array.from(accumulatedPoints), 3);
  const colorsAttr = new THREE.BufferAttribute(Float32Array.from(accumulatedColors), 3);
  
  if (!mergedPointCloud) {
    var geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', positionsAttr);
    geometry.setAttribute('color', colorsAttr);
    
    try {
      geometry.computeBoundingSphere();
    } catch (e) {
      console.warn('[Spatial] computeBoundingSphere failed, skipping:', e.message);
    }
    
    var material = new THREE.PointsMaterial({
      size: guiPointSize,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: false,
      opacity: 1.0,
      depthWrite: true,
      depthTest: true
    });
    mergedPointCloud = new THREE.Points(geometry, material);
    cloudGroup.add(mergedPointCloud);
  } else {
    // 优化：复用 geometry，只更新 attribute
    mergedPointCloud.geometry.setAttribute('position', positionsAttr);
    mergedPointCloud.geometry.setAttribute('color', colorsAttr);
    
    try {
      mergedPointCloud.geometry.computeBoundingSphere();
    } catch (e) {
      console.warn('[Spatial] computeBoundingSphere failed, skipping:', e.message);
    }
  }
}

function updateVisibleFrames() {
  // Always show all accumulated points — no frame limit (matches viser behavior).
  // Viser uses a deque(maxlen=300) that silently drops the oldest frame buffers
  // but the merged point cloud already contains all historical points.
  // Here we keep ALL points visible since accumulatedPoints never drops old frames.
  if (mergedPointCloud) {
    mergedPointCloud.geometry.setDrawRange(0, accumulatedPoints.length / 3);
  }
  
  // Legacy compatibility
  if (cameraFollowEnabled) {
    Object.keys(framePointCloudObjects).forEach(function(key) {
      if (framePointCloudObjects[key]) {
        framePointCloudObjects[key].visible = true;
      }
    });
    return;
  }
}

function updateAllFramePointCloudsVisibility() {
  Object.keys(framePointCloudObjects).forEach(function(key) {
    framePointCloudObjects[key].visible = true;
  });
}

function updateAccumulatedPointCloud() {
  updateAllFramePointCloudsVisibility();
  var overlay = document.getElementById('canvasOverlay');
  if (overlay) overlay.style.display = 'none';
}

function percentile(arr, p) {
  if (arr.length === 0) return 0;
  const sorted = arr.slice().sort((a, b) => a - b);
  const index = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

// 新增：额外数据存储
let cameraExtrinsics = null;
let cameraIntrinsics = null;
let depthData = {};
let frameImages = {};
let manifestData = null;

// 新增：获取文件清单
async function fetchManifest(batchId) {
  try {
    const response = await fetch(BATCH_SERVER_URL + '/batch/' + batchId + '/manifest');
    if (response.ok) {
      const result = await response.json();
      if (result.success) {
        manifestData = result.manifest;
        addLog('获取文件清单成功', 'ok');
        return manifestData;
      }
    }
    addLog('获取文件清单失败', 'err');
    return null;
  } catch (err) {
    addLog('获取文件清单异常: ' + err.message, 'err');
    return null;
  }
}

// 新增：获取相机位姿
async function fetchExtrinsics(batchId) {
  try {
    const response = await fetch(BATCH_SERVER_URL + '/batch/' + batchId + '/extrinsic');
    if (response.ok) {
      const result = await response.json();
      if (result.success) {
        cameraExtrinsics = result.extrinsic;
        addLog('获取相机位姿成功，共 ' + cameraExtrinsics.length + ' 帧', 'ok');
        return cameraExtrinsics;
      }
    }
    addLog('获取相机位姿失败', 'err');
    return null;
  } catch (err) {
    addLog('获取相机位姿异常: ' + err.message, 'err');
    return null;
  }
}

// 新增：获取相机内参
async function fetchIntrinsics(batchId) {
  try {
    const response = await fetch(BATCH_SERVER_URL + '/batch/' + batchId + '/intrinsic');
    if (response.ok) {
      const result = await response.json();
      if (result.success) {
        cameraIntrinsics = result.intrinsic;
        addLog('获取相机内参成功', 'ok');
        return cameraIntrinsics;
      }
    }
    addLog('获取相机内参失败', 'err');
    return null;
  } catch (err) {
    addLog('获取相机内参异常: ' + err.message, 'err');
    return null;
  }
}

// 新增：获取单帧深度图
async function fetchDepth(batchId, frameIndex) {
  try {
    const response = await fetch(BATCH_SERVER_URL + '/batch/' + batchId + '/frame/' + frameIndex + '/depth');
    if (response.ok) {
      const result = await response.json();
      if (result.success) {
        depthData[frameIndex] = result.depth;
        return result.depth;
      }
    }
    return null;
  } catch (err) {
    return null;
  }
}

// 新增：获取单帧图像
async function fetchFrameImage(batchId, frameIndex) {
  try {
    const response = await fetch(BATCH_SERVER_URL + '/batch/' + batchId + '/frame/' + frameIndex + '/image');
    if (response.ok) {
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      frameImages[frameIndex] = url;
      return url;
    }
    return null;
  } catch (err) {
    return null;
  }
}

async function updateFrameImagePreview(batchId, frameIndex) {
  const imgEl = document.getElementById('frameImagePreview');
  const labelEl = document.getElementById('frameImageLabel');
  if (!imgEl || !labelEl) return;
  
  if (!cameraFollowEnabled) {
    imgEl.style.display = 'none';
    labelEl.style.display = 'none';
    return;
  }
  
  const url = await fetchFrameImage(batchId, frameIndex);
  if (url) {
    imgEl.src = url;
    imgEl.style.display = 'block';
    labelEl.textContent = '帧 #' + frameIndex;
    labelEl.style.display = 'block';
  }
}

// 新增：获取 metadata.json（与 see/ 一致）
async function fetchMetadata(batchId) {
  try {
    const response = await fetch(BATCH_SERVER_URL + '/batch/' + batchId + '/metadata');
    if (response.ok) {
      const result = await response.json();
      if (result.success) {
        metadata = result.metadata;
        sceneCenter = metadata.scene_center || [0, 0, 0];
        sceneScale = metadata.scene_scale || 1.0;
        addLog('获取 metadata 成功，中心: [' + sceneCenter.map(function(v) { return v.toFixed(2); }).join(', ') + '], 尺度: ' + sceneScale.toFixed(2), 'ok');
        fitCameraToScene();
        return metadata;
      }
    }
    addLog('获取 metadata 失败', 'err');
    return null;
  } catch (err) {
    addLog('获取 metadata 异常: ' + err.message, 'err');
    return null;
  }
}

// 新增：获取 cameras.json（与 see/ 一致）
async function fetchCameras(batchId) {
  try {
    const response = await fetch(BATCH_SERVER_URL + '/batch/' + batchId + '/cameras');
    if (response.ok) {
      const result = await response.json();
      if (result.success) {
        // ✅ 如果已经有逐帧累积的数据，不覆盖，只补充缺失的帧
        if (camerasData && camerasData.length > 0) {
          const newCameras = result.cameras;
          for (let i = 0; i < newCameras.length; i++) {
            if (!camerasData[i]) {
              camerasData[i] = newCameras[i];
            }
          }
          addLog('补充 cameras.json 数据，共 ' + camerasData.length + ' 帧', 'ok');
        } else {
          camerasData = result.cameras;
          addLog('获取 cameras.json 成功，共 ' + camerasData.length + ' 帧', 'ok');
        }
        return camerasData;
      }
    }
    addLog('获取 cameras.json 失败', 'err');
    return null;
  } catch (err) {
    addLog('获取 cameras.json 异常: ' + err.message, 'err');
    return null;
  }
}

// 新增：获取所有额外数据（与 see/ 一致）
async function fetchAllExtraData(batchId, totalFrames) {
  addLog('开始获取额外数据...', 'info');
  
  // 获取 metadata 和 cameras（与 see/ 一致）
  await Promise.all([
    fetchMetadata(batchId),
    fetchCameras(batchId),
    fetchManifest(batchId)
  ]);
  
  // 使用 cameras.json 数据构建视锥体（与 see/ 完全一致）
  if (camerasData && camerasData.length > 0) {
    buildFrustumsFromCamerasData(camerasData);
    addLog('相机视锥体构建完成，共 ' + camerasData.length + ' 个', 'ok');
  }
  
  addLog('额外数据获取完成', 'ok');
}

// 从 cameras.json 构建视锥体（Viser 风格：完整相机坐标系 + 位姿平滑）
function buildFrustumsFromCamerasData(camData) {
  if (!THREE || !scene) return;
  
  // 清除旧的视锥体（安全释放资源）
  cameraFrustums.forEach(obj => {
    frustumGroup.remove(obj);
    disposeObject(obj);
  });
  cameraFrustums = [];
  
  const S = camData.length;
  if (S === 0) return;
  
  // 计算采样步长（最多显示 60 个相机坐标系）
  const step = Math.max(1, Math.floor(S / 60));
  
  // 重置位姿平滑
  SpatialState.smoothPose = null;
  
  for (let i = 0; i < S; i++) {
    if (i % step !== 0 && i !== S - 1) continue;
    
    const cam = camData[i];
    const t = cam.t_c2w || cam.t_w2c;
    const R = cam.R_c2w || cam.R_w2c;
    if (!t || !R) continue;
    
    // 位姿平滑
    const smoothPose = smoothCameraPose(cam);
    
    // Camera world position — use C2W directly (same as viser)
    const camPos = new THREE.Vector3(-smoothPose.t[0], -smoothPose.t[1], smoothPose.t[2]);
    
    // Extract camera coordinate axes from rotation matrix (same as viser)
    const xAxis = new THREE.Vector3(-smoothPose.R[0][0], -smoothPose.R[1][0], smoothPose.R[2][0]);
    const yAxis = new THREE.Vector3(-smoothPose.R[0][1], -smoothPose.R[1][1], smoothPose.R[2][1]);
    const zAxis = new THREE.Vector3(-smoothPose.R[0][2], -smoothPose.R[1][2], smoothPose.R[2][2]);
    
    const axisLen = 0.1;
    const axisRadius = 0.004;
    
    // X 轴（红色）
    const xGeom = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLen, 8);
    xGeom.translate(0, axisLen / 2, 0);
    xGeom.rotateZ(-Math.PI / 2);
    const xMat = new THREE.MeshBasicMaterial({ color: 0xff3333, transparent: true, opacity: 0.9 });
    const xAxisMesh = new THREE.Mesh(xGeom, xMat);
    xAxisMesh.position.copy(camPos);
    xAxisMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), xAxis);
    frustumGroup.add(xAxisMesh);
    cameraFrustums.push(xAxisMesh);
    
    // Y 轴（绿色）
    const yGeom = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLen, 8);
    yGeom.translate(0, axisLen / 2, 0);
    const yMat = new THREE.MeshBasicMaterial({ color: 0x33ff33, transparent: true, opacity: 0.9 });
    const yAxisMesh = new THREE.Mesh(yGeom, yMat);
    yAxisMesh.position.copy(camPos);
    yAxisMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), yAxis);
    frustumGroup.add(yAxisMesh);
    cameraFrustums.push(yAxisMesh);
    
    // Z 轴（蓝色）
    const zGeom = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLen, 8);
    zGeom.translate(0, axisLen / 2, 0);
    zGeom.rotateX(Math.PI / 2);
    const zMat = new THREE.MeshBasicMaterial({ color: 0x3366ff, transparent: true, opacity: 0.9 });
    const zAxisMesh = new THREE.Mesh(zGeom, zMat);
    zAxisMesh.position.copy(camPos);
    zAxisMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), zAxis);
    frustumGroup.add(zAxisMesh);
    cameraFrustums.push(zAxisMesh);
    
    // 相机中心点
    const centerGeom = new THREE.SphereGeometry(0.0075, 8, 8);
    const centerMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 });
    const centerMesh = new THREE.Mesh(centerGeom, centerMat);
    centerMesh.position.copy(camPos);
    frustumGroup.add(centerMesh);
    cameraFrustums.push(centerMesh);
  }
  
  // 绘制相机轨迹线（连接所有相机位置）
  if (S > 1) {
    const trajectoryPoints = [];
    SpatialState.smoothPose = null;
    for (let i = 0; i < S; i++) {
      const cam = camData[i];
      const t = cam.t_c2w || cam.t_w2c;
      if (t) {
        const smoothPose = smoothCameraPose(cam);
        trajectoryPoints.push(new THREE.Vector3(-smoothPose.t[0], -smoothPose.t[1], smoothPose.t[2]));
      }
    }
    if (trajectoryPoints.length > 1) {
      const curve = new THREE.CatmullRomCurve3(trajectoryPoints);
      const curvePoints = curve.getPoints(S * 2);
      const trajectoryGeom = new THREE.BufferGeometry().setFromPoints(curvePoints);
      const trajectoryMat = new THREE.LineBasicMaterial({ color: 0xff6b6b, transparent: true, opacity: 0.6 });
      const trajectoryLine = new THREE.Line(trajectoryGeom, trajectoryMat);
      frustumGroup.add(trajectoryLine);
      cameraFrustums.push(trajectoryLine);
    }
  }
}

function generateDepthColors(positions) {
  const colors = [];
  const numPoints = positions.length / 3;
  
  let minZ = Infinity;
  let maxZ = -Infinity;
  
  for (let i = 0; i < numPoints; i++) {
    const z = positions[i * 3 + 2];
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  
  const range = maxZ - minZ;
  
  for (let i = 0; i < numPoints; i++) {
    const z = positions[i * 3 + 2];
    const normalizedZ = (z - minZ) / range;
    const r = normalizedZ;
    const g = 0.5;
    const b = 1.0 - normalizedZ;
    colors.push(r, g, b);
  }
  
  return colors;
}

function generateFrameColors(frameCount) {
  const colors = [];
  const numPoints = currentPointCloudData.positions.length / 3;
  
  for (let i = 0; i < numPoints; i++) {
    const frameIndex = Math.floor(i / (numPoints / frameCount));
    const normalizedFrame = frameIndex / frameCount;
    const r = normalizedFrame;
    const g = 0.5;
    const b = 1.0 - normalizedFrame;
    colors.push(r, g, b);
  }
  
  return colors;
}

function rebuildCameraFrustums() {
  if (!THREE || !scene) return;
  
  cameraFrustums.forEach(frustum => {
    frustumGroup.remove(frustum);
    if (frustum.geometry) frustum.geometry.dispose();
    if (frustum.material) frustum.material.dispose();
  });
  cameraFrustums = [];
  
  if (camerasData && camerasData.length > 0) {
    buildFrustumsFromCamerasData(camerasData);
  }
}

/**
 * 初始化空间记忆系统
 * 依次初始化摄像头、3D场景、事件监听、API和可视化模块
 */
async function initSpatialMemory() {
  switchToLocalCamera();
  await init3DScene();
  initEventListeners();
  initApiAndVisualizer();

  const startBtn = document.getElementById('spatialStartCaptureBtn');
  const stopBtn = document.getElementById('spatialStopCaptureBtn');
  
  console.log('[Spatial] 初始化按钮状态:', {
    startBtnExists: !!startBtn,
    stopBtnExists: !!stopBtn,
    startBtnDisabledBefore: startBtn?.disabled,
    stopBtnDisabledBefore: stopBtn?.disabled
  });
  
  if (startBtn) {
    startBtn.disabled = false;
    console.log('[Spatial] 开始采集按钮已启用');
  }
  if (stopBtn) {
    stopBtn.disabled = true;
    console.log('[Spatial] 停止采集按钮已禁用');
  }
}

/**
 * 初始化API和可视化模块
 * 设置回调函数、模块间通信、启动服务状态检查
 */
function initApiAndVisualizer() {
  if (typeof SpatialApi === 'undefined' || typeof SpatialVisualizer === 'undefined') {
    console.error('[Spatial] 模块未加载');
    return;
  }

  SpatialApi.setCallbacks({
    onLogMessage: function(msg, type) {
      const logMsg = '[Spatial] ' + msg;
      const logType = type || 'info';
      
      if (logType === 'err') {
        console.error(logMsg);
      } else if (logType === 'ok') {
        console.log('%c' + logMsg, 'color: #4caf50; font-weight: bold');
      } else {
        console.log(logMsg);
      }
    },
    onStatusUpdate: function(status) {
      if (status.pointCount !== undefined) {
        var pcEl = document.getElementById('pointCount');
        if (pcEl) pcEl.textContent = status.pointCount;
      }
      if (status.batchId !== undefined) {
        var bcEl = document.getElementById('batchCount');
        if (bcEl) bcEl.textContent = status.batchId;
      }
      if (status.progress !== undefined) {
        var ppEl = document.getElementById('processProgress');
        if (ppEl) ppEl.textContent = status.progress;
      }
      if (status.frameCount !== undefined) {
        spatialFrameCounter = status.frameCount;
        var fcEl = document.getElementById('frameCount');
        if (fcEl) fcEl.textContent = status.frameCount;
      }
      if (status.collectedFrames !== undefined && status.targetFrames !== undefined) {
        var fcEl = document.getElementById('frameCount');
        if (fcEl) fcEl.textContent = status.collectedFrames + '/' + status.targetFrames;
        var ppEl = document.getElementById('processProgress');
        if (ppEl) ppEl.textContent = Math.round(status.collectedFrames / status.targetFrames * 100) + '%';
      }
      if (status.processing !== undefined) {
        var procEl = document.getElementById('stepProcessingStatus');
        var d3El = document.getElementById('step3DStatus');
        if (status.processing) {
          if (procEl) { procEl.textContent = '处理中'; procEl.className = 'step-status active'; }
        } else {
          if (procEl) { procEl.textContent = '完成'; procEl.className = 'step-status done'; }
          if (d3El) {
            setTimeout(function() {
              if (d3El) { d3El.textContent = '完成'; d3El.className = 'step-status done'; }
            }, 500);
          }
        }
        var ppEl = document.getElementById('processProgress');
        if (ppEl && status.progress) ppEl.textContent = status.progress;
      }
    }
  });

  SpatialVisualizer.setCallbacks({
    onStatsUpdate: function(stats) {
      // Removed right panel - stats no longer displayed
    }
  });

  SpatialApi.init();
  SpatialApi.startStatusCheck();
  SpatialApi.setCaptureFrameFunc(captureCurrentFrameData);
}

/**
 * 切换空间视图模式
 * @param {string} view - 视图模式（'3d'或'2d'，2D功能已移除）
 */
function switchSpatialView(view) {
  currentSpatialView = view;
  var container3d = document.getElementById('spatialCanvasContainer');
  var btn3d = document.getElementById('view3DBtn');

  if (view === '2d') {
    // 2D地图功能已移除，切换到2D时保持3D视图
    if (container3d) container3d.style.display = 'block';
    if (btn3d) btn3d.classList.add('active');
    addLog('2D地图功能已移除，请使用3D视图', 'info');
  } else {
    if (container3d) container3d.style.display = 'block';
    if (btn3d) btn3d.classList.add('active');
  }
}
async function switchToLocalCamera() {
  console.log('[Spatial] switchToLocalCamera called');
  const img = document.getElementById('spatialCameraImg');
  const video = document.getElementById('spatialCameraVideo');
  const placeholder = document.getElementById('spatialCameraPlaceholder');

  if (img) img.style.display = 'none';
  if (placeholder) placeholder.style.display = 'none';

  try {
    // 获取所有可用的视频设备
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');
    
    console.log('[Spatial] 可用摄像头:', videoDevices);
    
    if (videoDevices.length === 0) {
      throw new Error('未找到摄像头设备');
    }
    
    // 如果有多个摄像头，让用户选择
    let selectedDeviceId = videoDevices[0].deviceId;
    
    if (videoDevices.length > 1) {
      // 创建选择对话框
      const deviceNames = videoDevices.map((d, i) => `${i + 1}. ${d.label || `摄像头 ${i + 1}`}`);
      const choice = prompt(`检测到多个摄像头，请选择：\n${deviceNames.join('\n')}\n\n输入数字选择（默认1）：`);
      const index = parseInt(choice) - 1;
      if (!isNaN(index) && index >= 0 && index < videoDevices.length) {
        selectedDeviceId = videoDevices[index].deviceId;
      }
    }
    
    // 使用选中的设备
    const stream = await navigator.mediaDevices.getUserMedia({ 
      video: { deviceId: { exact: selectedDeviceId } } 
    });
    
    spatialVideoStream = stream;
    if (video) {
      video.srcObject = stream;
      video.style.display = 'block';
      video.play().catch(e => console.warn('[Spatial] video play failed:', e));
    }
    
    console.log('[Spatial] 已选择摄像头:', videoDevices.find(d => d.deviceId === selectedDeviceId)?.label || '未知设备');
    
    const cameraStatusEl = document.getElementById('spatialCameraStatus');
    if (cameraStatusEl) {
      cameraStatusEl.innerHTML = '<span class="status-dot online"></span> 本地摄像头';
    }
    
  } catch (err) {
    console.error('[Spatial] 无法访问本地摄像头:', err);
    if (placeholder) {
      placeholder.style.display = 'flex';
      placeholder.querySelector('span').textContent = '无法访问摄像头: ' + err.message;
    }
  }
}

function captureCurrentFrameData() {
  var canvas = document.createElement('canvas');
  canvas.width = SPATIAL_FRAME_WIDTH;
  canvas.height = SPATIAL_FRAME_HEIGHT;
  var ctx = canvas.getContext('2d');

  var video = document.getElementById('spatialCameraVideo');
  console.log('[Spatial] captureCurrentFrameData: video元素存在=' + (!!video));
  console.log('[Spatial] captureCurrentFrameData: video.videoWidth=' + (video ? video.videoWidth : 'undefined'));
  
  if (!video || !video.videoWidth) {
    console.warn('[Spatial] captureCurrentFrameData: 视频元素不存在或没有数据');
    return null;
  }
  
  try {
    ctx.drawImage(video, 0, 0, SPATIAL_FRAME_WIDTH, SPATIAL_FRAME_HEIGHT);
    var dataUrl = canvas.toDataURL('image/jpeg', 0.6);
    var base64 = dataUrl.split(',')[1];
    console.log('[Spatial] captureCurrentFrameData: 本地摄像头采集成功, 数据长度=' + base64.length);
    return { image: base64 };
  } catch (e) {
    console.error('[Spatial] captureCurrentFrameData: 本地摄像头 drawImage 失败:', e.message);
    return null;
  }
}

/**
 * 开始空间记忆采集
 * 读取UI输入参数，启动API采集
 */
function startSpatialCapture() {
  console.log('[Spatial UI] startSpatialCapture 被调用');
  if (typeof SpatialApi !== 'undefined') {
    // 读取采集参数
    var fpsInput = document.getElementById('captureFpsInput');
    var frameCountInput = document.getElementById('captureFrameCountInput');
    var maxImagesInput = document.getElementById('maxImagesInput');
    if (fpsInput && fpsInput.value) {
      spatialCaptureFps = parseInt(fpsInput.value) || 5;
    }
    if (frameCountInput && frameCountInput.value) {
      spatialCaptureTargetFrames = parseInt(frameCountInput.value) || 300;
    }
    if (maxImagesInput && maxImagesInput.value) {
      spatialMaxImages = parseInt(maxImagesInput.value);
      spatialCaptureTargetFrames = spatialMaxImages;
      if (spatialMaxImages > 320) {
        spatialKeyframeInterval = Math.floor((spatialMaxImages + 319) / 320);
      } else {
        spatialKeyframeInterval = 1;
      }
    } else {
      spatialMaxImages = null;
      spatialKeyframeInterval = 1;  // Default when maxImages not set (match viser)
    }

    SpatialApi.setCapturing(true);
    SpatialApi.reset();
    SpatialApi.getState().isCapturing = true;
    if (SpatialApi.startContinuousCapture) {
      SpatialApi.startContinuousCapture();
    }

    const startBtn = document.getElementById('spatialStartCaptureBtn');
    const stopBtn = document.getElementById('spatialStopCaptureBtn');
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;

    updateStepStatus('stepCapture', 'active');
    updateStepStatus('stepProcessing', 'pending');
    updateStepStatus('step3D', 'pending');

    addLog('空间记忆采集已启动 (FPS=' + spatialCaptureFps + ', 目标帧数=' + spatialCaptureTargetFrames + ')', 'ok');
  } else {
    addLog('SpatialApi 模块未加载，无法采集', 'err');
  }
}

/**
 * 停止空间记忆采集
 * 通知API停止采集，更新按钮状态
 */
function stopSpatialCapture() {
  if (typeof SpatialApi !== 'undefined') {
    SpatialApi.setCapturing(false);

    const startBtn = document.getElementById('spatialStartCaptureBtn');
    const stopBtn = document.getElementById('spatialStopCaptureBtn');
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;

    updateStepStatus('stepCapture', 'pending');
    updateStepStatus('stepProcessing', 'pending');
    updateStepStatus('step3D', 'pending');
    hideCaptureProgress();
    
    addLog('空间记忆采集已停止', 'info');
  }
}

async function forceStopProcessing() {
  addLog('正在强制停止处理...', 'info');
  
  if (typeof SpatialApi !== 'undefined') {
    SpatialApi.setCapturing(false);
  }
  
  isFetchingFrames = false;
  isBatchProcessing = false;
  isInferenceStarted = false;
  disableCameraFollow();
  
  if (spatialCaptureTimer) {
    clearTimeout(spatialCaptureTimer);
    spatialCaptureTimer = null;
  }
  
  if (currentBatchId) {
    try {
      await fetch(`${BATCH_SERVER_URL}/batch/${currentBatchId}/force_stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ batch_id: currentBatchId })
      });
    } catch (err) {
      console.warn('强制停止请求失败:', err.message);
    }
  }
  
  var imgEl = document.getElementById('frameImagePreview');
  var labelEl = document.getElementById('frameImageLabel');
  if (imgEl) imgEl.style.display = 'none';
  if (imgEl) imgEl.src = '';
  if (labelEl) labelEl.style.display = 'none';
  
  currentCameraTargetPos = null;
  currentCameraLookAt = null;
  smoothedRawCamPos = null;
  smoothedRawLookDir = null;
  currentFollowFrameIndex = -1;
  
  totalFramesAvailable = 0;
  currentFetchFrame = 0;
  fetchBatchId = null;
  spatialFrameCounter = 0;
  totalFramesCollected = 0;
  hideCaptureProgress();
  currentBatchId = null;
  isInitialBatch = true;
  collectedFrames = [];
  isUploading = false;
  
  const startBtn = document.getElementById('spatialStartCaptureBtn');
  const stopBtn = document.getElementById('spatialStopCaptureBtn');
  if (startBtn) startBtn.disabled = false;
  if (stopBtn) stopBtn.disabled = true;
  
  updateStepStatus('stepCapture', 'pending');
  updateStepStatus('stepProcessing', 'pending');
  updateStepStatus('step3D', 'pending');
  
  addLog('处理已停止，已渲染的点云保留在窗口中', 'ok');
}

/**
 * 初始化事件监听器
 * 绑定所有UI按钮的点击事件、输入框变化事件等
 */
function initEventListeners() {
  const startBtn = document.getElementById('spatialStartCaptureBtn');
  const stopBtn = document.getElementById('spatialStopCaptureBtn');
  const forceStopBtn = document.getElementById('spatialForceStopBtn');
  const resetCameraBtn = document.getElementById('resetCameraBtn');
  const togglePcdBtn = document.getElementById('togglePointCloudBtn');

  if (startBtn) {
    startBtn.addEventListener('click', startSpatialCapture);
    console.log('[Spatial] 开始采集按钮事件监听器已绑定');
  } else {
    console.error('[Spatial] 开始采集按钮不存在，无法绑定事件');
  }

  if (stopBtn) {
    stopBtn.addEventListener('click', stopSpatialCapture);
    console.log('[Spatial] 停止采集按钮事件监听器已绑定');
  } else {
    console.error('[Spatial] 停止采集按钮不存在，无法绑定事件');
  }

  if (forceStopBtn) {
    forceStopBtn.addEventListener('click', forceStopProcessing);
    console.log('[Spatial] 结束处理按钮事件监听器已绑定');
  } else {
    console.error('[Spatial] 结束处理按钮不存在，无法绑定事件');
  }

  if (resetCameraBtn) {
    resetCameraBtn.addEventListener('click', () => {
      SpatialVisualizer.resetCamera();
    });
  }

  if (togglePcdBtn) {
    togglePcdBtn.addEventListener('click', () => {
      SpatialVisualizer.togglePointCloud();
    });
  }
}

/**
 * 切换视角到全局概览
 */
function resetViewToOverview() {
  SpatialVisualizer.setViewDirection([0.5, -0.6, 0.6]);
}

/**
 * 切换视角到正前方
 */
function resetViewToFront() {
  SpatialVisualizer.setViewDirection([0.0, 0.0, 1.0]);
}

/**
 * 切换视角到俯视图
 */
function resetViewToTop() {
  SpatialVisualizer.setViewDirection([0.0, -1.0, 0.0]);
}

/**
 * 更新轨迹SVG（已废弃，保留占位）
 */
function updateTrajectorySvg() {}

// 恢复右上传入帧显示元素
(function() {
  const container = document.getElementById('spatialCanvasContainer');
  if (container && !document.getElementById('frameImageLabel')) {
    const label = document.createElement('div');
    label.id = 'frameImageLabel';
    label.className = 'frame-image-label';
    label.style.display = 'none';
    label.textContent = '帧 #0';
    
    const img = document.createElement('img');
    img.id = 'frameImagePreview';
    img.className = 'frame-image-preview';
    img.style.display = 'none';
    
    container.appendChild(label);
    container.appendChild(img);
    console.log('✅ 已恢复右上传入帧显示元素');
  }
})();

window.addEventListener('load', async () => {
  if (document.getElementById('spatialCanvasContainer')) {
    try {
      await initSpatialMemory();
      console.log('[Spatial] 初始化完成');
    } catch (err) {
      console.error('[Spatial] 初始化失败:', err);
    }
  }
});

function updateStepStatus(stepId, status) {
  const stepEl = document.getElementById(stepId + 'Status');
  if (!stepEl) return;
  console.log('[Spatial UI] updateStepStatus:', stepId, '->', status);
  
  stepEl.classList.remove('pending', 'active', 'done', 'error');
  stepEl.classList.add(status);
  
  const statusText = {
    'pending': '等待',
    'active': '进行中',
    'done': '完成',
    'error': '失败'
  };
  stepEl.textContent = statusText[status] || status;
}

window.reset3DCamera = () => SpatialVisualizer.resetCamera();
window.setViewDirection = (dir) => SpatialVisualizer.setViewDirection(dir);
window.switchSpatialView = switchSpatialView;
window.spatialStartCapture = startSpatialCapture;
window.spatialStopCapture = stopSpatialCapture;
window.addLog = addLog;
window.updateStepStatus = updateStepStatus;
window.togglePathVisibility = () => {};
window.startTestProcess = startTestProcess;
window.toggleCameraFrustums = toggleCameraFrustums;
window.exportToGLB = exportToGLB;

function toggleCameraFrustums() {
  if (frustumGroup) frustumGroup.visible = !frustumGroup.visible;
}

function exportToGLB() {
  addLog('开始导出GLB文件...', 'info');
  SpatialVisualizer.exportToGLB().then(result => {
    if (result.success) {
      addLog('GLB文件导出成功: ' + result.filename, 'ok');
      
      const link = document.createElement('a');
      link.href = result.url;
      link.download = result.filename;
      link.click();
    } else {
      addLog('GLB导出失败: ' + result.error, 'err');
    }
  }).catch(err => {
    addLog('GLB导出异常: ' + err.message, 'err');
  });
}

async function startTestProcess() {
  const testBtn = document.getElementById('spatialTestBtn');
  if (testBtn) testBtn.disabled = true;
  
  addLog('🧪 启动测试模式...', 'info');
  
  updateStepStatus('stepCapture', 'active');
  updateStepStatus('stepUpload', 'pending');
  updateStepStatus('stepProcessing', 'pending');
  updateStepStatus('stepDownload', 'pending');
  updateStepStatus('step3D', 'pending');
  
  updateProgressDisplay('uploadProgress', '0%');
  updateProgressDisplay('downloadProgress', '0%');
  
  let testBatchId = null;
  let logPollTimer = null;
  let lastLogCount = 0;
  
  const pollLogs = async () => {
    if (!testBatchId) return;
    try {
      const logResponse = await fetch(`${BATCH_SERVER_URL}/batch/${testBatchId}/logs`);
      if (logResponse.ok) {
        const logData = await logResponse.json();
        const logs = logData.logs || [];
        
        if (logs.length > lastLogCount) {
          const newLogs = logs.slice(lastLogCount);
          newLogs.forEach(log => {
            addLog(log.message, log.type);
          });
          lastLogCount = logs.length;
        }
      }
    } catch (err) {
      console.error('获取测试日志失败:', err);
    }
  };
  
  try {
    addLog('调用测试接口...');
    
    const response = await fetch(`${BATCH_SERVER_URL}/test/process_local_images`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      addLog('测试接口调用失败: ' + errorText, 'err');
      updateStepStatus('stepProcessing', 'error');
      if (testBtn) testBtn.disabled = false;
      return;
    }
    
    const result = await response.json();
    testBatchId = result.batch_id;
    
    addLog('测试批次创建: ' + testBatchId, 'ok');
    updateBatchIdDisplay(testBatchId);
    
    logPollTimer = setInterval(pollLogs, 1000);
    
    updateStepStatus('stepCapture', 'done');
    updateStepStatus('stepUpload', 'done');
    updateStepStatus('stepProcessing', 'active');
    
    if (result.success) {
      updateProgressDisplay('uploadProgress', '100%');
      updateStepStatus('stepProcessing', 'done');
      addLog('测试处理完成: ' + result.total_points + ' 点, ' + result.num_frames + ' 帧', 'ok');
      
      updateStepStatus('stepDownload', 'active');
      updateProgressDisplay('downloadProgress', '逐帧拉取点云...');
      addLog('开始逐帧拉取测试点云数据...');
      
      updateStepStatus('step3D', 'active');
      
      const totalFrames = result.num_frames || 0;
      if (totalFrames > 0 && typeof SpatialVisualizer !== 'undefined' && SpatialVisualizer.startFrameByFrameFetch) {
        await SpatialVisualizer.startFrameByFrameFetch(testBatchId, totalFrames);
        updateProgressDisplay('downloadProgress', '100%');
        updateStepStatus('stepDownload', 'done');
        updateStepStatus('step3D', 'done');
        addLog('✅ 测试逐帧点云拉取完成', 'ok');
      } else {
        addLog('无法启动逐帧拉取: totalFrames=' + totalFrames, 'err');
        updateStepStatus('stepDownload', 'error');
        updateStepStatus('step3D', 'error');
      }
      
      clearInterval(logPollTimer);
      if (testBtn) testBtn.disabled = false;
      return;
    } else {
      addLog('测试处理失败: ' + (result.error || 'unknown'), 'err');
      updateStepStatus('stepProcessing', 'error');
      clearInterval(logPollTimer);
      if (testBtn) testBtn.disabled = false;
      return;
    }
  } catch (err) {
    addLog('测试请求失败: ' + err.message, 'err');
    console.error('测试详细错误:', err);
    updateStepStatus('stepProcessing', 'error');
    clearInterval(logPollTimer);
    if (testBtn) testBtn.disabled = false;
  }
}