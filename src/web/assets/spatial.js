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

/** 相机位姿平滑（仅位置平滑，旋转不平滑避免拐弯变形） */
function smoothCameraPose(rawPose) {
  if (!SpatialState.smoothPose) {
    SpatialState.smoothPose = {
      t: [...rawPose.t_c2w],
      R: rawPose.R_c2w.map(row => [...row])
    };
    return SpatialState.smoothPose;
  }

  const alpha = 0.1;
  const maxStep = SpatialState.maxPoseStep;

  for (let i = 0; i < 3; i++) {
    let diff = rawPose.t_c2w[i] - SpatialState.smoothPose.t[i];
    if (Math.abs(diff) > maxStep) {
      diff = diff > 0 ? maxStep : -maxStep;
    }
    SpatialState.smoothPose.t[i] += alpha * diff;
  }

  SpatialState.smoothPose.R = rawPose.R_c2w.map(row => [...row]);

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
var SPATIAL_FRAME_WIDTH = 518;
/** 采集帧高度（像素） */
var SPATIAL_FRAME_HEIGHT = 518;
/** 目标采集总帧数（用户可通过UI「总帧数」输入框修改） */
var spatialCaptureTargetFrames = Infinity;
var spatialKeyframeInterval = 1;  // Default: every frame is a keyframe (same as viser when <= 320 frames)
var spatialMaxImages = 200;
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


// ============== 3D Visualization Module (viser-compatible) ==============

// Three.js references (populated by loadThreeJS)
let THREE = null;
let PLYLoader = null;

// Scene objects
let scene = null;
let camera3d = null;
let renderer = null;
let controls = null; // removed OrbitControls, kept for compat
let animationId = null;
let flightYaw = 0, flightPitch = 0;
let flightKeys = {};
let flightLeftDown = false, flightRightDown = false;
let flightLastMouseX = 0, flightLastMouseY = 0;

// Point cloud — single merged BufferGeometry, like viser
let mergedPoints = null;
let accumulatedPositions = [];   // flat Float32-compatible array
let accumulatedColors = [];     // flat Float32-compatible array
let frameRanges = {};           // {frameIndex: {start, count}}

// Trajectory
let trajectoryLine = null;

// Camera frustum meshes
let frustumMeshes = [];

// Parameters (matching viser defaults)
let guiDownsample = 10;
let guiPointSize = 0.00001;
let guiConfThreshold = 0.7;

// Stats
let visualizerStats = { fps: 0, vertices: 0 };
let frameTime = 0;
let frameCount = 0;
let onStatsUpdate = null;

// Camera data (populated by fetchNextFrame)
let camerasData = [];

// Camera follow state
let cameraFollowEnabled = false;
let cameraFollowDistance = 0.8;
let followSmoothedPos = null;
let followLookTarget = null;
const FOLLOW_SMOOTH = 0.25;

// Data fetching state
let fetchBatchId = null;
let isFetchingFrames = false;
let currentFetchFrame = 0;
let totalFramesAvailable = 0;
let framePointClouds = [];       // raw per-frame data before filtering
let framePointCloudObjects = {}; // legacy compat

// Image preview elements
let currentFollowFrameIndex = -1;
  if (camera3d) { const dir = new THREE.Vector3(); camera3d.getWorldDirection(dir); flightPitch = Math.asin(dir.y); flightYaw = Math.atan2(-dir.x, -dir.z); }

// Scene center from metadata (for camera fitting only, NOT for coordinate transform)
let metadata = null;
let sceneCenter = [0, 0, 0];
let sceneScale = 1.0;

// Sliding window
const MAX_FRAMES = 300;
const MAX_FRUSTUMS = 60;

// ---------- Three.js loader ----------

async function loadThreeJS() {
  if (THREE) return THREE;
  try {
    THREE = await import('three');
    const { PLYLoader: PL } = await import('three/addons/loaders/PLYLoader.js');
    PLYLoader = PL;
    console.log('[Spatial] Three.js and PLYLoader loaded successfully');
    return THREE;
  } catch (err) {
    console.error('[Spatial] Failed to load Three.js:', err);
    throw err;
  }
}

// ---------- 3D Scene Setup ----------

async function init3DScene() {
  const container = document.getElementById('spatialCanvasContainer');
  if (!container) {
    console.error('[Spatial] Container #spatialCanvasContainer not found');
    return;
  }

  const loaded = await loadThreeJS();
  if (!loaded) {
    console.error('[Spatial] Three.js failed to load');
    return;
  }

  const width = container.clientWidth || 800;
  const height = container.clientHeight || 600;

  // Renderer
  const canvas = document.getElementById('spatialCanvas');
  renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: false });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0xffffff, 1);

  // Scene — no coordinate transform, raw world coordinates (like viser)
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);

  // Camera
  camera3d = new THREE.PerspectiveCamera(70, width / height, 0.001, 10000);
  camera3d.position.set(0, 0, 5);
  camera3d.lookAt(0, 0, 0);

  // Flight controls state
  flightYaw = 0;
  flightPitch = 0;
  flightKeys = {};
  flightLeftDown = false;
  flightRightDown = false;
  flightLastMouseX = 0;
  flightLastMouseY = 0;

  // Mouse: left drag = rotate, right drag = pan, scroll = zoom
  renderer.domElement.addEventListener('mousedown', onFlightMouseDown);
  renderer.domElement.addEventListener('mouseup', onFlightMouseUp);
  renderer.domElement.addEventListener('mousemove', onFlightMouseMove);
  renderer.domElement.addEventListener('wheel', onFlightWheel, { passive: false });
  renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());
  window.addEventListener('keydown', onFlightKeyDown);
  window.addEventListener('keyup', onFlightKeyUp);

  // ResizeObserver
  new ResizeObserver(entries => {
    for (const entry of entries) {
      const w = entry.contentRect.width;
      const h = entry.contentRect.height;
      if (camera3d && renderer) {
        camera3d.aspect = w / h;
        camera3d.updateProjectionMatrix();
        renderer.setSize(w, h);
      }
    }
  }).observe(container);


  // Start animation loop
  animate();

  console.log('[Spatial] 3D scene initialized (viser-compatible, no coord transform)');
}

function onWindowResize() {
  const container = document.getElementById('spatialCanvasContainer');
  if (!container) return;
  const w = container.clientWidth || 800;
  const h = container.clientHeight || 600;
  if (camera3d) {
    camera3d.aspect = w / h;
    camera3d.updateProjectionMatrix();
  }
  if (renderer) renderer.setSize(w, h);
}

// ---------- Flight Controls ----------

const FLIGHT_MOVE_SPEED = 0.08;
const FLIGHT_ZOOM_SPEED = 0.15;
const FLIGHT_SENSITIVITY = 0.003;

function onFlightMouseDown(e) {
  if (e.button === 0) { flightLeftDown = true; }
  if (e.button === 2) { flightRightDown = true; }
  flightLastMouseX = e.clientX;
  flightLastMouseY = e.clientY;
  e.preventDefault();
}

function onFlightMouseUp(e) {
  if (e.button === 0) { flightLeftDown = false; }
  if (e.button === 2) { flightRightDown = false; }
}

function onFlightMouseMove(e) {
  const dx = e.clientX - flightLastMouseX;
  const dy = e.clientY - flightLastMouseY;
  if (flightLeftDown) {
    flightYaw += dx * FLIGHT_SENSITIVITY;
    flightPitch -= dy * FLIGHT_SENSITIVITY;
    // No pitch limit
  }
  if (flightRightDown) {
    if (!camera3d) return;
    const right = new THREE.Vector3();
    right.crossVectors(camera3d.getWorldDirection(new THREE.Vector3()), new THREE.Vector3(0, 1, 0)).normalize();
    const up = new THREE.Vector3(0, 1, 0);
    const scale = FLIGHT_MOVE_SPEED * 0.03;
    camera3d.position.addScaledVector(right, -dx * scale);
    camera3d.position.addScaledVector(up, dy * scale);
  }
  flightLastMouseX = e.clientX;
  flightLastMouseY = e.clientY;
}

function onFlightWheel(e) {
  e.preventDefault();
  if (!camera3d || cameraFollowEnabled) return;
  const dir = camera3d.getWorldDirection(new THREE.Vector3());
  camera3d.position.addScaledVector(dir, e.deltaY > 0 ? -FLIGHT_ZOOM_SPEED : FLIGHT_ZOOM_SPEED);
}

function onFlightKeyDown(e) {
  flightKeys[e.key.toLowerCase()] = true;
  // Prevent browser defaults for flight keys
  if (['w','a','s','d','q','e'].includes(e.key.toLowerCase())) {
    e.preventDefault();
  }
}

function onFlightKeyUp(e) {
  flightKeys[e.key.toLowerCase()] = false;
}

function updateFlightMovement() {
  if (!camera3d || cameraFollowEnabled) return;
  const dir = camera3d.getWorldDirection(new THREE.Vector3());
  const right = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(0, 1, 0)).normalize();
  const up = new THREE.Vector3(0, 1, 0);
  const speed = FLIGHT_MOVE_SPEED;
  if (flightKeys['w']) camera3d.position.addScaledVector(dir, speed);
  if (flightKeys['s']) camera3d.position.addScaledVector(dir, -speed);
  if (flightKeys['a']) camera3d.position.addScaledVector(right, -speed);
  if (flightKeys['d']) camera3d.position.addScaledVector(right, speed);
  if (flightKeys['q']) camera3d.position.addScaledVector(up, -speed);
  if (flightKeys['e']) camera3d.position.addScaledVector(up, speed);
}

// ---------- Animation Loop ----------

function animate() {
  animationId = requestAnimationFrame(animate);

  // Update flight camera orientation from yaw/pitch (unless in follow mode)
  if (!cameraFollowEnabled && camera3d) {
    const euler = new THREE.Euler(flightPitch, flightYaw, 0, 'YXZ');
    camera3d.quaternion.setFromEuler(euler);
    updateFlightMovement();
  }

  // Camera follow (OpenCV y-down -> flip camera up)
  if (cameraFollowEnabled && followSmoothedPos && followLookTarget) {
    camera3d.position.copy(followSmoothedPos);
    camera3d.up.set(0, -1, 0);
    camera3d.lookAt(followLookTarget);
  }

  if (renderer && scene && camera3d) {
    frameCount++;
    const now = performance.now();
    if (now - frameTime >= 1000) {
      visualizerStats.fps = Math.round(frameCount / ((now - frameTime) / 1000));
      frameCount = 0;
      frameTime = now;
    }
    renderer.render(scene, camera3d);
  }
}

// ---------- Point Cloud ----------

function addFramePointCloudToScene(frameIndex) {
  if (!THREE || !scene) return;

  const frameData = framePointClouds[frameIndex];
  if (!frameData) return;

  const positions = frameData.positions;
  const colors = frameData.colors;
  const confs = frameData.confs || new Float32Array(positions.length / 3);
  const numPoints = positions.length / 3;
  if (numPoints === 0) return;

  // Filter: isfinite + confidence + downsample
  const stride = Math.max(1, guiDownsample);
  const confThreshold = guiConfThreshold;
  const maxSize = Math.ceil(numPoints / stride);
  const filteredPos = new Float32Array(maxSize * 3);
  const filteredCol = new Float32Array(maxSize * 3);
  let count = 0;

  for (let i = 0; i < numPoints; i++) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    if (!isFinite(x) || !isFinite(y) || !isFinite(z)) continue;
    if (confs[i] <= confThreshold) continue;
    if (i % stride !== 0) continue;

    // Raw world coordinates — no transform (matching viser)
    filteredPos[count * 3] = x;
    filteredPos[count * 3 + 1] = y;
    filteredPos[count * 3 + 2] = z;
    filteredCol[count * 3] = colors[i * 3];
    filteredCol[count * 3 + 1] = colors[i * 3 + 1];
    filteredCol[count * 3 + 2] = colors[i * 3 + 2];
    count++;
  }

  if (count === 0) return;

  // Record frame range
  const startIdx = accumulatedPositions.length / 3;
  for (let j = 0; j < count; j++) {
    accumulatedPositions.push(filteredPos[j * 3], filteredPos[j * 3 + 1], filteredPos[j * 3 + 2]);
    accumulatedColors.push(filteredCol[j * 3], filteredCol[j * 3 + 1], filteredCol[j * 3 + 2]);
  }
  frameRanges[frameIndex] = { start: startIdx, count };

  // Sliding window: keep max 300 frames
  const indices = Object.keys(frameRanges).map(Number).sort((a, b) => a - b);
  if (indices.length > MAX_FRAMES) {
    const toRemove = indices.slice(0, indices.length - MAX_FRAMES);
    let removedPts = 0;
    for (const idx of toRemove) {
      removedPts += frameRanges[idx].count;
      delete frameRanges[idx];
    }
    if (removedPts > 0) {
      accumulatedPositions = accumulatedPositions.slice(removedPts * 3);
      accumulatedColors = accumulatedColors.slice(removedPts * 3);
      // Adjust remaining ranges
      for (const idx of Object.keys(frameRanges).map(Number)) {
        frameRanges[idx].start -= removedPts;
      }
    }
  }

  // Update merged point cloud
  updateMergedPointCloud();
  visualizerStats.vertices = accumulatedPositions.length / 3;
}

function updateMergedPointCloud() {
  if (!THREE || !scene) return;
  const numPoints = accumulatedPositions.length / 3;
  if (numPoints === 0) return;
  if (accumulatedPositions.length !== accumulatedColors.length) {
    console.warn('[Spatial] accumulatedPositions/Colors length mismatch, skipping update');
    return;
  }

  try {
    const posAttr = new THREE.BufferAttribute(Float32Array.from(accumulatedPositions), 3);
    const colAttr = new THREE.BufferAttribute(Float32Array.from(accumulatedColors), 3);

    if (!mergedPoints) {
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', posAttr);
      geom.setAttribute('color', colAttr);
      const mat = new THREE.PointsMaterial({
        size: guiPointSize,
        vertexColors: true,
        sizeAttenuation: true,
        transparent: false,
        opacity: 1.0,
        depthWrite: true,
        depthTest: true,
      });
      mergedPoints = new THREE.Points(geom, mat);
      scene.add(mergedPoints);
    } else {
      mergedPoints.geometry.setAttribute('position', posAttr);
      mergedPoints.geometry.setAttribute('color', colAttr);
    }
  } catch (e) {
    console.error('[Spatial] updateMergedPointCloud failed:', e.message);
  }
}

// ---------- Trajectory ----------

function updateTrajectoryLine() {
  if (!THREE || !scene) return;

  try {
    if (trajectoryLine) {
      scene.remove(trajectoryLine);
      if (trajectoryLine.geometry) trajectoryLine.geometry.dispose();
      if (trajectoryLine.material) trajectoryLine.material.dispose();
      trajectoryLine = null;
    }

    const pts = [];
    for (let i = 0; i < camerasData.length; i++) {
      const c = camerasData[i];
      if (!c) continue;
      const t = c.t_c2w || c.t_w2c;
      if (!t || !Array.isArray(t) || t.length < 3) continue;
      if (!isFinite(t[0]) || !isFinite(t[1]) || !isFinite(t[2])) continue;
      pts.push(new THREE.Vector3(t[0], t[1], t[2]));
    }
    if (pts.length < 2) return;

    // CatmullRom spline matching viser: catmullrom type, tension=0.5
    const curve = new THREE.CatmullRomCurve3(pts, false, 'catmullrom', 0.5);
    const curvePts = curve.getPoints(pts.length * 3);
    const geom = new THREE.BufferGeometry().setFromPoints(curvePts);
    const mat = new THREE.LineBasicMaterial({
      color: 0x78c878,
      linewidth: 3,
      transparent: true,
      opacity: 1.0,
    });
    trajectoryLine = new THREE.Line(geom, mat);
    scene.add(trajectoryLine);
  } catch (e) {
    console.warn('[Spatial] updateTrajectoryLine failed:', e.message);
  }
}

// ---------- Camera Frustums ----------

function updateCameraFrustums() {
  if (!THREE || !scene) return;
  try {if (!THREE || !scene) return;

  // Remove old frustum meshes
  for (const m of frustumMeshes) {
    scene.remove(m);
    if (m.geometry) m.geometry.dispose();
    if (m.material) m.material.dispose();
  }
  frustumMeshes = [];

  const validCams = camerasData.filter(c => c && (c.t_c2w || c.t_w2c) && (c.R_c2w || c.R_w2c));
  if (validCams.length === 0) return;

  // Sample to max MAX_FRUSTUMS
  const step = Math.max(1, Math.floor(validCams.length / MAX_FRUSTUMS));
  const axisLen = 0.05;
  const axisRadius = 0.002;

  for (let i = 0; i < validCams.length; i++) {
    if (i % step !== 0 && i !== validCams.length - 1) continue;

    const cam = validCams[i];
    const t = cam.t_c2w || cam.t_w2c;
    const R = cam.R_c2w || cam.R_w2c;
    if (!t || !Array.isArray(t) || t.length < 3) continue;
    if (!R || !Array.isArray(R) || R.length < 3) continue;
    if (!isFinite(t[0]) || !isFinite(t[1]) || !isFinite(t[2])) continue;
    // Raw world coordinates — no transform
    const pos = new THREE.Vector3(t[0], t[1], t[2]);

    // Camera axes from rotation matrix columns
    const xAxis = new THREE.Vector3(R[0][0], R[1][0], R[2][0]);
    const yAxis = new THREE.Vector3(R[0][1], R[1][1], R[2][1]);
    const zAxis = new THREE.Vector3(R[0][2], R[1][2], R[2][2]);

    function makeAxis(dir, color) {
      const g = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLen, 8);
      g.translate(0, axisLen / 2, 0);
      if (color === 0xff3333) g.rotateZ(-Math.PI / 2); // X: rotate to point along X
      if (color === 0x3366ff) g.rotateX(Math.PI / 2);  // Z: rotate to point along Z
      const m = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9 });
      const mesh = new THREE.Mesh(g, m);
      mesh.position.copy(pos);
      if (color === 0x33ff33) {
        // Y axis (default cylinder direction)
        mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
      } else if (color === 0xff3333) {
        mesh.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), dir);
      } else {
        mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), dir);
      }
      scene.add(mesh);
      frustumMeshes.push(mesh);
    }

    makeAxis(xAxis, 0xff3333);
    makeAxis(yAxis, 0x33ff33);
    makeAxis(zAxis, 0x3366ff);

    // Center sphere
    const sg = new THREE.SphereGeometry(0.002, 8, 8);
    const sm = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 });
    const sphere = new THREE.Mesh(sg, sm);
    sphere.position.copy(pos);
    scene.add(sphere);
    frustumMeshes.push(sphere);
  }
  } catch (e) {
    console.warn('[Spatial] updateCameraFrustums failed:', e.message);
  }
}

// Helper: rebuild both trajectory and frustums
function updateTrajectoryAndFrustums() {
  updateTrajectoryLine();
  updateCameraFrustums();
}

// ---------- Camera Follow ----------

function enableCameraFollow() {
  cameraFollowEnabled = true;
  followSmoothedPos = null;
  followLookTarget = null;
  currentFollowFrameIndex = -1;
  if (camera3d) { const dir = new THREE.Vector3(); camera3d.getWorldDirection(dir); flightPitch = Math.asin(dir.y); flightYaw = Math.atan2(-dir.x, -dir.z); }
}

function disableCameraFollow() {
  cameraFollowEnabled = false;
  followSmoothedPos = null;
  followLookTarget = null;
  currentFollowFrameIndex = -1;
  if (camera3d) { const dir = new THREE.Vector3(); camera3d.getWorldDirection(dir); flightPitch = Math.asin(dir.y); flightYaw = Math.atan2(-dir.x, -dir.z); }
  const imgEl = document.getElementById('frameImagePreview');
  const labelEl = document.getElementById('frameImageLabel');
  if (imgEl) imgEl.style.display = 'none';
  if (labelEl) labelEl.style.display = 'none';
}

function updateCameraFollow(frameIndex) {
  if (!cameraFollowEnabled || !camera3d || frameIndex >= camerasData.length) return;

  const cam = camerasData[frameIndex];
  if (!cam) return;

  try {
    const t = cam.t_c2w || cam.t_w2c;
    const R_raw = cam.R_c2w || cam.R_w2c;
    if (!t || !R_raw || !Array.isArray(t) || t.length < 3) return;
    const R = R_raw.flat();
    if (!R || R.length < 9) return;

    // Raw world coordinates (matching scene objects)
    const camPos = new THREE.Vector3(t[0], t[1], t[2]);
    const forward = new THREE.Vector3(R[2], R[5], R[8]).normalize();
    const up = new THREE.Vector3(R[1], R[4], R[7]).normalize();

    // Viewer behind (0.5m) and above (0.3m) the tracked camera
    const viewPos = camPos.clone().addScaledVector(forward, -0.5).addScaledVector(up, -0.3);
    // Look at a point ahead of the tracked camera
    const lookTarget = camPos.clone().addScaledVector(forward, 2.0);

    // Smooth follow
    if (!followSmoothedPos) {
      followSmoothedPos = viewPos.clone();
      followLookTarget = lookTarget.clone();
    } else {
      followSmoothedPos.lerp(viewPos, FOLLOW_SMOOTH);
      followLookTarget.lerp(lookTarget, FOLLOW_SMOOTH);
    }

    if (currentFollowFrameIndex !== frameIndex) {
      currentFollowFrameIndex = frameIndex;
      updateFrameImagePreview(fetchBatchId, frameIndex);
    }
  } catch (e) {
    console.warn('updateCameraFollow error for frame ' + frameIndex + ':', e.message);
  }
}

// ---------- Camera Controls ----------

function reset3DCamera() {
  if (camera3d) {
    const cx = sceneCenter[0], cy = sceneCenter[1], cz = sceneCenter[2];
    camera3d.position.set(cx, cy, cz + sceneScale * 0.5);
    flightYaw = 0;
    flightPitch = 0;
  }
}

function fitCameraToScene() {
  if (!camera3d) return;
  const cx = sceneCenter[0], cy = sceneCenter[1], cz = sceneCenter[2];
  const dist = sceneScale * 0.8;
  camera3d.position.set(cx + dist * 0.5, cy + dist * 0.5, cz + dist);
  // Update flight yaw/pitch to look at scene center
  const lookDir = new THREE.Vector3(cx, cy, cz).sub(camera3d.position).normalize();
  const euler = new THREE.Euler().setFromQuaternion(
    new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, -1), lookDir)
  );
  flightYaw = euler.y;
  flightPitch = euler.x;
}

function setViewDirection(direction) {
  if (!camera3d) return;
  const cx = sceneCenter[0], cy = sceneCenter[1], cz = sceneCenter[2];
  const dist = sceneScale * 0.8;
  const dir = new THREE.Vector3(direction[0], direction[1], direction[2]).normalize();
  camera3d.position.copy(new THREE.Vector3(cx, cy, cz).addScaledVector(dir, dist));
  const lookDir = new THREE.Vector3(cx, cy, cz).sub(camera3d.position).normalize();
  const euler = new THREE.Euler().setFromQuaternion(
    new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, -1), lookDir)
  );
  flightYaw = euler.y;
  flightPitch = euler.x;
}

// ---------- Toggle Visibility ----------

function togglePointCloud() {
  if (mergedPoints) mergedPoints.visible = !mergedPoints.visible;
}

function toggleTrajectory() {
  if (trajectoryLine) trajectoryLine.visible = !trajectoryLine.visible;
}

function toggleCameraFrustums() {
  const visible = frustumMeshes.length > 0 ? !frustumMeshes[0].visible : true;
  for (const m of frustumMeshes) m.visible = visible;
}

// ---------- Export ----------

function getCurrentSceneData() {
  const data = { points: [], colors: [], cameraPoses: [] };
  if (mergedPoints) {
    const pos = mergedPoints.geometry.attributes.position.array;
    const col = mergedPoints.geometry.attributes.color;
    data.points = Array.from(pos);
    data.colors = col ? Array.from(col.array) : [];
  }
  for (const c of camerasData) {
    if (c) data.cameraPoses.push({ t_c2w: c.t_c2w, R_c2w: c.R_c2w });
  }
  return data;
}

function exportVisualizerToGLB() {
  return new Promise((resolve) => {
    try {
      const data = getCurrentSceneData();
      const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      resolve({
        success: true,
        url,
        filename: 'point_cloud_' + new Date().toISOString().slice(0, 19).replace(/[:-]/g, '') + '.json',
      });
    } catch (e) {
      resolve({ success: false, error: e.message });
    }
  });
}

// ---------- Frame Image Preview ----------

async function fetchFrameImage(batchId, frameIndex) {
  try {
    const resp = await fetch(BATCH_SERVER_URL + '/batch/' + batchId + '/frame/' + frameIndex + '/image');
    if (resp.ok) {
      const blob = await resp.blob();
      return URL.createObjectURL(blob);
    }
  } catch (e) {}
  return null;
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

// ---------- Metadata ----------

async function fetchMetadata(batchId) {
  try {
    const resp = await fetch(BATCH_SERVER_URL + '/batch/' + batchId + '/metadata');
    if (resp.ok) {
      const result = await resp.json();
      if (result.success) {
        metadata = result.metadata;
        sceneCenter = metadata.scene_center || [0, 0, 0];
        sceneScale = metadata.scene_scale || 1.0;
        fitCameraToScene();
        return metadata;
      }
    }
  } catch (e) {
    console.error('fetchMetadata error:', e);
  }
  return null;
}

// ---------- Utility ----------

function disposeObject(obj) {
  if (!obj) return;
  if (obj.geometry) {
    obj.geometry.dispose();
    for (const attr in obj.geometry.attributes) {
      if (obj.geometry.attributes[attr]) obj.geometry.attributes[attr].dispose();
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

// ---------- SpatialVisualizer Exports ----------

const SpatialVisualizer = {
  // Init
  init: init3DScene,
  loadThreeJS,

  // Point cloud
  addFramePointCloud: addFramePointCloudToScene,
  togglePointCloud,
  updateMergedPointCloud,

  // Trajectory
  updateTrajectoryLine,
  toggleTrajectory,

  // Camera frustums
  updateCameraFrustums,
  toggleCameraFrustums,

  // Camera controls
  resetCamera: reset3DCamera,
  setViewDirection,
  fitCameraToScene,

  // Camera follow
  enableCameraFollow,
  disableCameraFollow,
  updateCameraFollow,

  // Frame fetching (delegates to global functions)
  startFrameByFrameFetch,
  fetchMetadata,

  // Export
  getCurrentSceneData,
  exportToGLB,

  // Accessors
  getRenderer: () => renderer,
  getScene: () => scene,
  getCamera: () => camera3d,

  // Callbacks
  setCallbacks(cbs) {
    if (cbs.onStatsUpdate) onStatsUpdate = cbs.onStatsUpdate;
  },

  // Legacy compat
  fetchFrameImage,
  updateFrameImagePreview,
  addFramePointCloudToScene,
};

console.log('✅ SpatialVisualizer (viser-compatible) loaded');

async function startFrameByFrameFetch(batchId, totalFrames) {
  // 隐藏批次模式点云（使用单一点云对象模式）
  // 清理旧的单一点云对象
  if (mergedPoints) {
    scene.remove(mergedPoints);
    if (mergedPoints.geometry) mergedPoints.geometry.dispose();
    if (mergedPoints.material) mergedPoints.material.dispose();
    mergedPoints = null;
  }
  accumulatedPositions = [];
  accumulatedColors = [];
  frameRanges = {};
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
    // fetchAllExtraData removed — metadata already fetched, trajectory/frustums updated per-frame
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
    
    // Check for network errors
    if (!pointCloudResponse) {
      throw new Error('pointCloudResponse is null');
    }
    
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
        updateTrajectoryAndFrustums();
      } catch (e) {
        console.warn('Camera frustum update failed for frame ' + currentFetchFrame + ': ' + e.message);
      }
      try {
        updateCameraFollow(currentFetchFrame);
      } catch (e) {
        console.warn('Camera follow update failed:', e.message);
      }
      
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
    var dataUrl = canvas.toDataURL('image/jpeg', 0.85);
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

    // Tell backend that upload is complete
    if (currentBatchId) {
      sendFinishInference(currentBatchId);
    }
    
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
  
  // handled by followSmoothedPos in new module
  // handled by followLookTarget in new module
  followSmoothedPos = null;
  followLookTarget = null;
  currentFollowFrameIndex = -1;
  if (camera3d) { const dir = new THREE.Vector3(); camera3d.getWorldDirection(dir); flightPitch = Math.asin(dir.y); flightYaw = Math.atan2(-dir.x, -dir.z); }
  
  totalFramesAvailable = 0;
  currentFetchFrame = 0;
  fetchBatchId = null;
  spatialFrameCounter = 0;
  totalFramesCollected = 0;
  hideCaptureProgress();

    // Tell backend that upload is complete
    if (currentBatchId) {
      sendFinishInference(currentBatchId);
    }
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
  SpatialVisualizer.toggleCameraFrustums();
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