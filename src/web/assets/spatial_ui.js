// spatial.js - 空间记忆模块入口文件
// 职责：初始化各子模块、事件绑定、UI 交互

// ============== 摄像头相关 ==============

let spatialCameraMode = 'server';
let spatialVideoStream = null;
var spatialFrameCounter = 0;
let currentSpatialView = '3d';
var spatialServerPollTimer = null;

// ============== 初始化 ==============

function initSpatialMemory() {
  initCamera();
  init3DSceneFromEntry();
  init2DMapFromEntry();
  initEventListeners();
  updateTrajectorySvg();
  initApiAndVisualizer();

  const startBtn = document.getElementById('spatialStartCaptureBtn');
  const stopBtn = document.getElementById('spatialStopCaptureBtn');
  if (startBtn) startBtn.disabled = false;
  if (stopBtn) stopBtn.disabled = true;

  syncServerCaptureStatus();
  setInterval(syncServerCaptureStatus, 3000);
}

// 初始化 API 和可视化模块的连接
function initApiAndVisualizer() {
  if (typeof SpatialApi === 'undefined' || typeof SpatialVisualizer === 'undefined') {
    console.error('[Spatial] 模块未加载');
    return;
  }

  SpatialApi.setCallbacks({
    onPointCloudUpdate: function(data) {
      SpatialVisualizer.updateScene(data);
      if (typeof SpatialMap !== 'undefined') {
        SpatialMap.updateData(data);
      }
    },
    onLogMessage: function(msg, type) {
      addLog(msg, type);
    },
    onStatusUpdate: function(status) {
      if (status.pointCount !== undefined) {
        document.getElementById('pointCount').textContent = status.pointCount;
      }
      if (status.batchId !== undefined) {
        document.getElementById('batchCount').textContent = status.batchId;
      }
      if (status.progress !== undefined) {
        document.getElementById('processProgress').textContent = status.progress;
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
    },
    onWebSocketStatusChange: function(isConnected) {
      updateWebSocketStatus(isConnected);
    }
  });

  SpatialVisualizer.setCallbacks({
    onStatsUpdate: function(stats) {
      if (stats.fps !== undefined) {
        document.getElementById('statFps').textContent = stats.fps;
      }
      if (stats.vertices !== undefined) {
        document.getElementById('statVertices').textContent = stats.vertices;
      }
    }
  });

  SpatialApi.init();
  SpatialApi.startStatusCheck();

  SpatialApi.setCaptureFrameFunc(captureCurrentFrameData);
}

async function init3DSceneFromEntry() {
  await SpatialVisualizer.init();
}

async function init2DMapFromEntry() {
  if (typeof SpatialMap === 'undefined') {
    console.warn('[Spatial] SpatialMap module not loaded');
    return;
  }
  SpatialMap.init();
  SpatialMap.setCallbacks({
    onMapClick: function(wx, wz, sx, sy) {
      addLog('地图点击: X=' + wx.toFixed(2) + ' Z=' + wz.toFixed(2), 'info');
    },
    onMapHover: function(wx, wz, sx, sy) {
      var posX = document.getElementById('posX');
      var posZ = document.getElementById('posZ');
      if (posX) posX.textContent = wx.toFixed(2);
      if (posZ) posZ.textContent = wz.toFixed(2);
    }
  });
}

function switchSpatialView(view) {
  currentSpatialView = view;
  var container3d = document.getElementById('spatialCanvasContainer');
  var container2d = document.getElementById('spatial2DMapContainer');
  var btn3d = document.getElementById('view3DBtn');
  var btn2d = document.getElementById('view2DBtn');

  if (view === '2d') {
    if (container3d) container3d.style.display = 'none';
    if (container2d) container2d.style.display = 'block';
    if (btn3d) btn3d.classList.remove('active');
    if (btn2d) btn2d.classList.add('active');
    if (typeof SpatialMap !== 'undefined') {
      if (typeof SpatialVisualizer !== 'undefined' && SpatialVisualizer.getCurrentSceneData) {
        var sceneData = SpatialVisualizer.getCurrentSceneData();
        if (sceneData.points.length > 0) {
          SpatialMap.updateData(sceneData);
        }
      }
      SpatialMap.render();
    }
  } else {
    if (container3d) container3d.style.display = 'block';
    if (container2d) container2d.style.display = 'none';
    if (btn3d) btn3d.classList.add('active');
    if (btn2d) btn2d.classList.remove('active');
  }
}

// 从服务器同步采集状态
async function syncServerCaptureStatus() {
  try {
    const response = await fetch('/api/status');
    const status = await response.json();
    
    // 更新摄像头状态显示 - 使用友好的名称
    const cameraStatusEl = document.getElementById('spatialCameraStatus');
    if (cameraStatusEl) {
      // 根据 sourceName 判断是服务器摄像头还是本地摄像头
      const isServerCamera = status.sourceName && status.sourceName.includes('ffmpeg');
      const cameraText = isServerCamera ? '服务器摄像头' : '本地摄像头';
      cameraStatusEl.innerHTML = '<span class="status-dot online"></span> ' + cameraText;
    }
    
    if (status.isCapturing !== undefined) {
      const isCapturing = Boolean(status.isCapturing);
      
      // 更新 SpatialApi 状态（如果可用）
      if (typeof SpatialApi !== 'undefined') {
        try {
          const apiState = SpatialApi.getState ? SpatialApi.getState() : null;
          if (!apiState || apiState.isCapturing !== isCapturing) {
            SpatialApi.setCapturing(isCapturing);
            if (SpatialApi.getState) {
              SpatialApi.getState().isCapturing = isCapturing;
            }
          }
        } catch (e) {
          console.warn('[Spatial] 更新 SpatialApi 状态失败:', e.message);
        }
      }
      
      // 更新按钮状态
      const startBtn = document.getElementById('spatialStartCaptureBtn');
      const stopBtn = document.getElementById('spatialStopCaptureBtn');
      if (startBtn) startBtn.disabled = isCapturing;
      if (stopBtn) stopBtn.disabled = !isCapturing;
    }
  } catch (err) {
    console.warn('[Spatial] 获取服务器状态失败:', err.message);
  }
}

// ============== 摄像头控制 ==============

function initCamera() {
  console.log('[Spatial] initCamera called, mode:', spatialCameraMode);
  if (spatialCameraMode === 'server') {
    switchToServerCamera();
  } else {
    switchToLocalCamera();
  }
}

function switchToServerCamera() {
  spatialCameraMode = 'server';
  console.log('[Spatial] switchToServerCamera called');
  const img = document.getElementById('spatialCameraImg');
  const video = document.getElementById('spatialCameraVideo');
  const placeholder = document.getElementById('spatialCameraPlaceholder');
  const cameraStatusEl = document.getElementById('spatialCameraStatus');

  if (spatialVideoStream) {
    spatialVideoStream.getTracks().forEach(track => track.stop());
    spatialVideoStream = null;
  }

  if (video) {
    video.style.display = 'none';
  }
  if (img) {
    img.style.display = 'block';
    // 立即设置图片源，避免等待轮询
    img.src = '/api/latest-frame?t=' + Date.now();
    startServerCameraPolling();
  }
  if (placeholder) {
    placeholder.style.display = 'none';
  }
  // 更新摄像头状态显示
  if (cameraStatusEl) {
    cameraStatusEl.innerHTML = '<span class="status-dot online"></span> 服务器摄像头';
  }
}

function switchToLocalCamera() {
  spatialCameraMode = 'local';
  console.log('[Spatial] switchToLocalCamera called');
  const img = document.getElementById('spatialCameraImg');
  const video = document.getElementById('spatialCameraVideo');
  const placeholder = document.getElementById('spatialCameraPlaceholder');

  if (img) img.style.display = 'none';
  if (placeholder) placeholder.style.display = 'none';

  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
      spatialVideoStream = stream;
      if (video) {
        video.srcObject = stream;
        video.style.display = 'block';
      }
    })
    .catch(err => {
      console.error('[Spatial] 无法访问本地摄像头:', err);
      if (placeholder) {
        placeholder.style.display = 'flex';
        placeholder.querySelector('.placeholder-text').textContent = '无法访问摄像头';
      }
    });
}

function toggleSpatialCamera() {
  const btn = document.getElementById('spatialCameraToggleBtn');
  const statusEl = document.getElementById('spatialCameraStatus');
  if (spatialCameraMode === 'local') {
    switchToServerCamera();
    if (btn) btn.textContent = '🖥️ 切换本地';
    if (statusEl) statusEl.innerHTML = '<span class="status-dot online"></span> 服务器摄像头';
  } else {
    switchToLocalCamera();
    if (btn) btn.textContent = '🖥️ 切换服务器';
    if (statusEl) statusEl.innerHTML = '<span class="status-dot online"></span> 本地摄像头';
  }
}

function startServerCameraPolling() {
  if (spatialServerPollTimer) {
    clearInterval(spatialServerPollTimer);
  }
  const img = document.getElementById('spatialCameraImg');
  if (!img) return;

  spatialServerPollTimer = setInterval(() => {
    if (spatialCameraMode === 'server') {
      img.src = '/api/latest-frame?t=' + Date.now();
    }
  }, 100);
}

var SPATIAL_FRAME_WIDTH = 640;
var SPATIAL_FRAME_HEIGHT = 480;

function captureCurrentFrameData() {
  var canvas = document.createElement('canvas');
  canvas.width = SPATIAL_FRAME_WIDTH;
  canvas.height = SPATIAL_FRAME_HEIGHT;
  var ctx = canvas.getContext('2d');

  if (spatialCameraMode === 'server') {
    var img = document.getElementById('spatialCameraImg');
    if (!img || !img.complete || !img.naturalWidth) return null;
    try {
      ctx.drawImage(img, 0, 0, SPATIAL_FRAME_WIDTH, SPATIAL_FRAME_HEIGHT);
      var dataUrl = canvas.toDataURL('image/jpeg', 0.8);
      var base64 = dataUrl.split(',')[1];
      return { image: base64 };
    } catch (e) {
      return null;
    }
  }

  var video = document.getElementById('spatialCameraVideo');
  if (!video || !video.videoWidth) return null;
  ctx.drawImage(video, 0, 0, SPATIAL_FRAME_WIDTH, SPATIAL_FRAME_HEIGHT);
  var dataUrl = canvas.toDataURL('image/jpeg', 0.8);
  var base64 = dataUrl.split(',')[1];
  return { image: base64 };
}

// ============== 采集控制 ==============

function startSpatialCapture() {
  console.log('[Spatial UI] startSpatialCapture 被调用');
  if (typeof SpatialApi !== 'undefined') {
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

    addLog('空间记忆采集已启动', 'ok');
  } else {
    addLog('SpatialApi 模块未加载，无法采集', 'err');
  }
}

function stopSpatialCapture() {
  if (typeof SpatialApi !== 'undefined') {
    SpatialApi.setCapturing(false);

    const startBtn = document.getElementById('spatialStartCaptureBtn');
    const stopBtn = document.getElementById('spatialStopCaptureBtn');
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;

    addLog('空间记忆采集已停止', 'info');
  }
}

function spatialStartCapture() {
  startSpatialCapture();
}

function spatialStopCapture() {
  stopSpatialCapture();
}

// ============== UI 更新 ==============

function updateWebSocketStatus(isConnected) {
  const statusEl = document.getElementById('websocketStatus');
  if (statusEl) {
    statusEl.innerHTML = isConnected
      ? '<span class="status-dot online"></span> RTX3090已连接'
      : '<span class="status-dot offline"></span> RTX3090未连接';
  }
}

// ============== 事件绑定 ==============

function initEventListeners() {
  const startBtn = document.getElementById('spatialStartCaptureBtn');
  const stopBtn = document.getElementById('spatialStopCaptureBtn');
  const switchCameraBtn = document.getElementById('spatialCameraToggleBtn');
  const resetCameraBtn = document.getElementById('resetCameraBtn');
  const togglePcdBtn = document.getElementById('togglePointCloudBtn');
  const cameraSelectBtns = document.querySelectorAll('.camera-source-btn');

  if (startBtn) {
    startBtn.addEventListener('click', () => {
      startSpatialCapture();
    });
  }

  if (stopBtn) {
    stopBtn.addEventListener('click', () => {
      stopSpatialCapture();
    });
  }

  if (switchCameraBtn) {
    switchCameraBtn.addEventListener('click', () => {
      toggleSpatialCamera();
    });
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

  cameraSelectBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.dataset.mode;
      if (mode === 'local') {
        switchToLocalCamera();
      } else {
        switchToServerCamera();
      }
    });
  });
}

// ============== 视角控制 ==============

function resetViewToOverview() {
  SpatialVisualizer.setViewDirection([0.5, -0.6, 0.6]);
}

function resetViewToFront() {
  SpatialVisualizer.setViewDirection([0.0, 0.0, 1.0]);
}

function resetViewToTop() {
  SpatialVisualizer.setViewDirection([0.0, -1.0, 0.0]);
}

// ============== 轨迹 SVG ==============

function updateTrajectorySvg() {
  const svg = document.getElementById('trajectorySvg');
  if (!svg) return;

  svg.innerHTML = '';

  const width = svg.clientWidth || 300;
  const height = svg.clientHeight || 150;

  if (typeof SpatialApi !== 'undefined') {
    const state = SpatialApi.getState();
  }
}

// ============== GUI 控制绑定 ==============

function setGuiPointSize(value) {
  SpatialVisualizer.setPointSize(value);
}

function setGuiDownsample(value) {
  SpatialVisualizer.setDownsample(value);
}

function setGuiShowCamera(checked) {
  SpatialVisualizer.setShowCamera(checked);
}

function setGuiCamSize(value) {
  SpatialVisualizer.setCamSize(value);
}

// ============== 日志 ==============

function addLog(msg, type = 'info') {
  const logEl = document.getElementById('spatialLog');
  if (!logEl) return;

  const time = new Date().toLocaleTimeString();
  const className = type === 'err' ? 'log-error' : type === 'ok' ? 'log-success' : 'log-info';
  const entry = document.createElement('div');
  entry.className = `log-entry ${className}`;
  entry.textContent = `[${time}] ${msg}`;

  logEl.appendChild(entry);
  logEl.scrollTop = logEl.scrollHeight;

  while (logEl.children.length > 100) {
    logEl.removeChild(logEl.firstChild);
  }
}

// ============== 步骤状态 ==============

function updateStepStatus(stepId, status) {
  const stepEl = document.getElementById(stepId);
  if (!stepEl) return;

  stepEl.classList.remove('step-pending', 'step-active', 'step-done', 'step-error');
  stepEl.classList.add(`step-${status}`);
}

// ============== 导出（兼容旧代码）==============

// 兼容旧的事件绑定方式
window.addEventListener('load', () => {
  if (document.getElementById('spatialCanvasContainer')) {
    initSpatialMemory();
  }
});

// 兼容旧的全域函数
window.reset3DCamera = () => SpatialVisualizer.resetCamera();
window.setViewDirection = (dir) => SpatialVisualizer.setViewDirection(dir);
window.switchSpatialView = switchSpatialView;
window.toggleSpatialCamera = toggleSpatialCamera;
window.spatialStartCapture = spatialStartCapture;
window.spatialStopCapture = spatialStopCapture;
window.addLog = addLog;
window.updateWebSocketStatus = updateWebSocketStatus;
window.updateStepStatus = updateStepStatus;
