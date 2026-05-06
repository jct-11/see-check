var SPATIAL_BATCH_SIZE = 200;
var SPATIAL_OVERLAP = 100;
var SPATIAL_INITIAL_FPS = 10;
var SPATIAL_STEADY_FPS = 3;

var spatialIsCapturing = false;
var spatialFrameCounter = 0;
var totalFramesCollected = 0;
var currentBatchId = 0;
var isInitialBatch = true;
var spatialCaptureTimer = null;
var wsConnected = false;

var collectedFrames = [];
var isBatchProcessing = false;

var onPointCloudUpdate = null;
var onLogMessage = null;
var onStatusUpdate = null;
var onWebSocketStatusChange = null;

function setSpatialApiCallbacks(callbacks) {
  if (callbacks.onPointCloudUpdate) onPointCloudUpdate = callbacks.onPointCloudUpdate;
  if (callbacks.onLogMessage) onLogMessage = callbacks.onLogMessage;
  if (callbacks.onStatusUpdate) onStatusUpdate = callbacks.onStatusUpdate;
  if (callbacks.onWebSocketStatusChange) onWebSocketStatusChange = callbacks.onWebSocketStatusChange;
}

function addLog(msg, type) {
  if (onLogMessage) {
    onLogMessage(msg, type || 'info');
  } else {
    console.log('[Spatial API] ' + msg);
  }
}

function updateStatus(status) {
  if (onStatusUpdate) onStatusUpdate(status);
}

function notifyWebSocketStatus(isConnected) {
  wsConnected = isConnected;
  if (onWebSocketStatusChange) onWebSocketStatusChange(isConnected);
}

function getCurrentFPS() {
  return totalFramesCollected < SPATIAL_BATCH_SIZE ? SPATIAL_INITIAL_FPS : SPATIAL_STEADY_FPS;
}

function getFrameInterval() {
  return 1000 / getCurrentFPS();
}

async function checkRTXServerStatus() {
  try {
    var controller = new AbortController();
    var timeoutId = setTimeout(function() { controller.abort(); }, 5000);
    var response = await fetch('/api/spatial/proxy?path=/test/info', {
      method: 'GET',
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    if (response.ok) {
      addLog('RTX3090已连接', 'ok');
      notifyWebSocketStatus(true);
      return true;
    }
  } catch (err) {
    console.warn('[Spatial API] RTX3090不可达:', err.message);
  }
  notifyWebSocketStatus(false);
  return false;
}

function startRTXStatusCheck() {
  checkRTXServerStatus();
  setInterval(checkRTXServerStatus, 10000);
}

function initWebSocket() {
  console.log('[Spatial API] 使用批量推理模式');
  checkRTXServerStatus();
}

function startContinuousCapture() {
  console.log('[Spatial API] startContinuousCapture: 被调用, spatialIsCapturing=' + spatialIsCapturing);
  if (!spatialIsCapturing) return;

  if (spatialFrameCounter === 0 && collectedFrames.length === 0) {
    currentBatchId++;
    isInitialBatch = true;
    addLog('开始采集 (' + getCurrentFPS() + ' FPS)，目标 ' + SPATIAL_BATCH_SIZE + ' 帧', 'info');
  }

  collectFrame();

  if (spatialIsCapturing) {
    spatialCaptureTimer = setTimeout(startContinuousCapture, getFrameInterval());
  }
}

function collectFrame() {
  if (!spatialIsCapturing || isBatchProcessing) return;

  var frameData = null;
  if (captureCurrentFrame) {
    frameData = captureCurrentFrame();
  }
  if (!frameData) {
    console.log('[Spatial API] collectFrame: 无帧数据, captureCurrentFrame=' + (captureCurrentFrame ? 'defined' : 'null'));
    return;
  }

  console.log('[Spatial API] collectFrame: 收到帧, image.length=' + (frameData.image ? frameData.image.length : 0));

  collectedFrames.push(frameData.image);
  spatialFrameCounter++;
  totalFramesCollected++;

  updateStatus({
    frameCount: spatialFrameCounter,
    totalFrames: totalFramesCollected,
    collectedFrames: collectedFrames.length,
    targetFrames: SPATIAL_BATCH_SIZE
  });

  if (collectedFrames.length >= SPATIAL_BATCH_SIZE) {
    submitBatch();
  }
}

function submitBatch() {
  if (isBatchProcessing || collectedFrames.length === 0) {
    console.log('[Spatial API] submitBatch: 跳过, isBatchProcessing=' + isBatchProcessing + ', collectedFrames.length=' + collectedFrames.length);
    return;
  }

  console.log('[Spatial API] submitBatch: 提交 ' + collectedFrames.length + ' 帧');
  isBatchProcessing = true;
  var framesToSend = collectedFrames.splice(0, SPATIAL_BATCH_SIZE);

  addLog('提交 ' + framesToSend.length + ' 帧到RTX3090进行批量推理...', 'info');
  updateStatus({ processing: true, progress: '推理中...' });

  fetch('/api/spatial/batch-inference', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      frames: framesToSend,
      batch_id: currentBatchId
    })
  })
  .then(function(res) { return res.json(); })
  .then(function(result) {
    isBatchProcessing = false;

    if (!result.success) {
      addLog('批量推理失败: ' + (result.error || 'unknown'), 'err');
      updateStatus({ processing: false });
      return;
    }

    var points = result.point_cloud || [];
    var camPoses = result.camera_poses || [];
    var pointCount = result.point_count || (points.length / 3);

    if (points.length > 0 && onPointCloudUpdate) {
      onPointCloudUpdate({
        points: points,
        colors: result.point_colors || [],
        cameraPoses: camPoses,
        batchId: currentBatchId,
        replaceMode: true
      });
      addLog('点云生成完成: ' + pointCount + ' 点, ' + result.frames_processed + ' 帧', 'ok');
    } else {
      addLog('推理完成但无点云数据', 'warn');
    }

    updateStatus({
      processing: false,
      progress: '100%',
      pointCount: pointCount,
      frameCount: spatialFrameCounter,
      totalFrames: totalFramesCollected
    });

    spatialFrameCounter = SPATIAL_OVERLAP;
    isInitialBatch = false;

    if (spatialIsCapturing && collectedFrames.length >= SPATIAL_BATCH_SIZE) {
      submitBatch();
    }
  })
  .catch(function(err) {
    isBatchProcessing = false;
    addLog('批量推理请求失败: ' + err.message, 'err');
    updateStatus({ processing: false });
  });
}

var captureCurrentFrame = null;
function setCaptureFrameFunc(func) {
  captureCurrentFrame = func;
}

function getSpatialCaptureState() {
  return {
    isCapturing: spatialIsCapturing,
    frameCounter: spatialFrameCounter,
    totalFrames: totalFramesCollected,
    currentBatchId: currentBatchId,
    cameraMode: typeof spatialCameraMode !== 'undefined' ? spatialCameraMode : 'local',
    wsConnected: wsConnected,
    collectedFrames: collectedFrames.length,
    isBatchProcessing: isBatchProcessing
  };
}

function setSpatialCapturing(value) {
  spatialIsCapturing = value;
  if (!value) {
    if (spatialCaptureTimer) {
    clearTimeout(spatialCaptureTimer);
    spatialCaptureTimer = null;
  }
  }
}

function resetSpatialState() {
  spatialFrameCounter = 0;
  totalFramesCollected = 0;
  currentBatchId = 0;
  isInitialBatch = true;
  collectedFrames = [];
  isBatchProcessing = false;
}

function isWsConnected() {
  return wsConnected;
}

function disconnectWebSocket() {
  wsConnected = false;
}

window.SpatialApi = {
  init: initWebSocket,
  setCaptureFrameFunc: setCaptureFrameFunc,
  setCallbacks: setSpatialApiCallbacks,
  getState: getSpatialCaptureState,
  setCapturing: setSpatialCapturing,
  reset: resetSpatialState,
  isConnected: isWsConnected,
  disconnect: disconnectWebSocket,
  checkServerStatus: checkRTXServerStatus,
  startStatusCheck: startRTXStatusCheck,
  startContinuousCapture: startContinuousCapture
};
