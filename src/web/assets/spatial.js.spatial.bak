// spatial.js - 空间记忆模块
// 功能：摄像头采集、发送到RTX3090处理、返回3D图像、显示摄像头移动过程

let spatialIsCapturing = false;
let spatialCameraMode = 'server';
let spatialVideoStream = null;
let spatialServerPollTimer = null;
let spatialFrameCounter = 0;
let pointCloudData = [];
let cameraTrajectory = [];

// Three.js 相关变量
let scene, camera, renderer, mesh;
let animationId = null;
let isPathVisible = true;
let isPointCloudVisible = true;
let renderWireframe = false;
let renderTexture = false;
let renderLighting = true;

// 统计信息
let stats = { fps: 0, vertices: 0, faces: 0 };
let frameTime = 0;
let frameCount = 0;

// 初始化函数
function initSpatialMemory() {
  initCamera();
  init3DScene();
  initEventListeners();
  updateTrajectorySvg();
}

// 初始化摄像头
function initCamera() {
  console.log('[Spatial] initCamera called, mode:', spatialCameraMode);
  console.log('[Spatial] img element:', document.getElementById('spatialCameraImg'));
  console.log('[Spatial] placeholder element:', document.getElementById('spatialCameraPlaceholder'));
  if (spatialCameraMode === 'server') {
    switchToServerCamera();
  } else {
    switchToLocalCamera();
  }
}

// 切换到服务器摄像头
function switchToServerCamera() {
  spatialCameraMode = 'server';
  console.log('[Spatial] switchToServerCamera called');
  const img = document.getElementById('spatialCameraImg');
  const video = document.getElementById('spatialCameraVideo');
  const placeholder = document.getElementById('spatialCameraPlaceholder');
  
  if (spatialVideoStream) {
    spatialVideoStream.getTracks().forEach(t => t.stop());
    spatialVideoStream = null;
  }
  video.style.display = 'none';
  placeholder.style.display = 'flex';
  placeholder.innerHTML = '<span>正在连接服务器摄像头...</span>';
  img.style.display = 'none';
  
  img.onerror = function() {
    updateCameraStatus(false, '服务器摄像头不可用');
  };
  
  img.onload = function() {
    placeholder.style.display = 'none';
    img.style.cssText = 'width:100%;height:100%;object-fit:cover;display:block;';
    updateCameraStatus(true, '服务器摄像头');
  };
  
  img.src = '/api/latest-frame?t=' + Date.now();
  console.log('[Spatial] Setting img.src to:', img.src);
  
  if (spatialServerPollTimer) clearInterval(spatialServerPollTimer);
  spatialServerPollTimer = setInterval(() => {
    img.src = '/api/latest-frame?t=' + Date.now();
  }, 2000);
}

// 切换到本地摄像头
function switchToLocalCamera() {
  spatialCameraMode = 'local';
  const img = document.getElementById('spatialCameraImg');
  const video = document.getElementById('spatialCameraVideo');
  const placeholder = document.getElementById('spatialCameraPlaceholder');
  
  if (spatialServerPollTimer) {
    clearInterval(spatialServerPollTimer);
    spatialServerPollTimer = null;
  }
  img.style.display = 'none';
  
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    placeholder.style.display = 'flex';
    placeholder.innerHTML = '<span>本地摄像头不可用</span>';
    updateCameraStatus(false, '浏览器不支持');
    return;
  }
  
  placeholder.style.display = 'flex';
  placeholder.innerHTML = '<span>正在启动本地摄像头...</span>';
  video.style.display = 'none';
  
  navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })
    .then(stream => {
      spatialVideoStream = stream;
      video.srcObject = stream;
      video.style.cssText = 'width:100%;height:100%;object-fit:cover;';
      placeholder.style.display = 'none';
      updateCameraStatus(true, '本地摄像头');
      
      stream.getVideoTracks().forEach(track => {
        track.onended = () => {
          spatialVideoStream = null;
          video.style.display = 'none';
          placeholder.style.display = 'flex';
          placeholder.innerHTML = '<span>摄像头已断开</span>';
          updateCameraStatus(false, '已断开');
        };
      });
    })
    .catch(e => {
      placeholder.style.display = 'flex';
      placeholder.innerHTML = '<span>无法访问摄像头</span>';
      updateCameraStatus(false, '访问被拒绝');
    });
}

// 更新摄像头状态显示
function updateCameraStatus(isOnline, message) {
  const statusEl = document.getElementById('spatialCameraStatus');
  const dotClass = isOnline ? 'online' : 'offline';
  statusEl.innerHTML = `<span class="status-dot ${dotClass}"></span> ${message}`;
}

// 开始采集
async function spatialStartCapture() {
  if (spatialIsCapturing) return;
  
  try {
    const response = await fetch('/api/spatial/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: spatialCameraMode })
    });
    
    if (response.ok) {
      spatialIsCapturing = true;
      spatialFrameCounter = 0;
      cameraTrajectory = [];
      pointCloudData = [];
      
      document.getElementById('spatialStartCaptureBtn').disabled = true;
      document.getElementById('spatialStopCaptureBtn').disabled = false;
      
      updateStepStatus('stepCapture', 'active');
      updateStepStatus('stepProcessing', 'waiting');
      updateStepStatus('step3D', 'waiting');
      
      addLog('开始空间记忆采集', 'ok');
      
      // 开始模拟采集和处理流程
      startSimulation();
    }
  } catch (err) {
    addLog('启动失败: ' + err.message, 'err');
  }
}

// 停止采集
async function spatialStopCapture() {
  if (!spatialIsCapturing) return;
  
  try {
    await fetch('/api/spatial/stop', { method: 'POST' });
    
    spatialIsCapturing = false;
    document.getElementById('spatialStartCaptureBtn').disabled = false;
    document.getElementById('spatialStopCaptureBtn').disabled = true;
    
    updateStepStatus('stepCapture', 'done');
    updateStepStatus('stepProcessing', 'done');
    updateStepStatus('step3D', 'done');
    
    addLog('停止空间记忆采集', 'info');
  } catch (err) {
    addLog('停止失败: ' + err.message, 'err');
  }
}

// 模拟采集和处理流程
function startSimulation() {
  let step = 0;
  
  const simulateProcess = () => {
    if (!spatialIsCapturing) return;
    
    step++;
    spatialFrameCounter++;
    
    // 更新帧计数
    document.getElementById('frameCount').textContent = spatialFrameCounter;
    
    // 模拟摄像头移动轨迹
    const angle = (step * 0.1) % (Math.PI * 2);
    const radius = 3;
    const x = Math.cos(angle) * radius;
    const z = Math.sin(angle) * radius;
    const y = Math.sin(angle * 0.5) * 0.5;
    
    cameraTrajectory.push({ x, y, z });
    
    // 更新位置信息
    document.getElementById('posX').textContent = x.toFixed(2);
    document.getElementById('posY').textContent = y.toFixed(2);
    document.getElementById('posZ').textContent = z.toFixed(2);
    
    // 模拟处理进度
    const progress = Math.min(100, step * 5);
    document.getElementById('processProgress').textContent = progress + '%';
    
    // 模拟点云生成
    if (step % 3 === 0) {
      generatePointCloud();
    }
    
    // 更新步骤状态
    if (step <= 5) {
      updateStepStatus('stepCapture', 'active');
      addLog(`采集帧 #${step}`, 'ok');
    } else if (step <= 10) {
      updateStepStatus('stepCapture', 'done');
      updateStepStatus('stepProcessing', 'active');
      addLog(`处理帧 #${step - 5} (RTX3090)`, 'info');
    } else {
      updateStepStatus('stepProcessing', 'done');
      updateStepStatus('step3D', 'active');
      addLog('3D重建进行中...', 'info');
    }
    
    // 更新轨迹和3D场景
    updateTrajectorySvg();
    update3DScene();
    
    if (step < 20) {
      setTimeout(simulateProcess, 800);
    } else {
      addLog('3D重建完成', 'ok');
      updateStepStatus('step3D', 'done');
    }
  };
  
  simulateProcess();
}

// 生成模拟点云数据
function generatePointCloud() {
  const newPoints = [];
  for (let i = 0; i < 50; i++) {
    newPoints.push({
      x: (Math.random() - 0.5) * 6,
      y: (Math.random() - 0.5) * 2,
      z: (Math.random() - 0.5) * 6,
      color: [Math.random(), Math.random(), Math.random()]
    });
  }
  pointCloudData = pointCloudData.concat(newPoints);
  document.getElementById('pointCount').textContent = pointCloudData.length;
}

// 更新步骤状态
function updateStepStatus(stepId, status) {
  const statusEl = document.getElementById(stepId + 'Status');
  if (statusEl) {
    statusEl.textContent = status === 'active' ? '进行中' : status === 'done' ? '完成' : '等待';
    statusEl.className = 'step-status ' + status;
  }
}

// 添加日志
function addLog(message, type) {
  const logEl = document.getElementById('modelingLog');
  const now = new Date();
  const timeStr = now.getHours().toString().padStart(2, '0') + ':' +
                  now.getMinutes().toString().padStart(2, '0') + ':' +
                  now.getSeconds().toString().padStart(2, '0');
  const logItem = document.createElement('div');
  logItem.className = 'log-item ' + (type || '');
  logItem.textContent = '[' + timeStr + '] ' + message;
  logEl.insertBefore(logItem, logEl.firstChild);
  
  // 限制日志数量
  while (logEl.children.length > 50) {
    logEl.removeChild(logEl.lastChild);
  }
}

// 更新轨迹SVG
function updateTrajectorySvg() {
  const svg = document.getElementById('trajectorySvg');
  const width = svg.clientWidth || 400;
  const height = svg.clientHeight || 60;
  
  svg.innerHTML = '';
  
  if (cameraTrajectory.length < 2) return;
  
  // 找到边界
  const xs = cameraTrajectory.map(p => p.x);
  const zs = cameraTrajectory.map(p => p.z);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minZ = Math.min(...zs);
  const maxZ = Math.max(...zs);
  
  const rangeX = Math.max(maxX - minX, 0.1);
  const rangeZ = Math.max(maxZ - minZ, 0.1);
  
  // 生成路径
  let pathD = '';
  cameraTrajectory.forEach((point, index) => {
    const x = ((point.x - minX) / rangeX) * (width - 40) + 20;
    const y = height - ((point.z - minZ) / rangeZ) * (height - 20) - 10;
    
    if (index === 0) {
      pathD += `M ${x} ${y}`;
    } else {
      pathD += ` L ${x} ${y}`;
    }
  });
  
  // 绘制轨迹线
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', pathD);
  path.setAttribute('fill', 'none');
  path.setAttribute('stroke', '#4CAF50');
  path.setAttribute('stroke-width', '2');
  path.setAttribute('stroke-linecap', 'round');
  svg.appendChild(path);
  
  // 绘制起点
  const startPoint = cameraTrajectory[0];
  const startX = ((startPoint.x - minX) / rangeX) * (width - 40) + 20;
  const startY = height - ((startPoint.z - minZ) / rangeZ) * (height - 20) - 10;
  
  const startCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  startCircle.setAttribute('cx', startX);
  startCircle.setAttribute('cy', startY);
  startCircle.setAttribute('r', '4');
  startCircle.setAttribute('fill', '#4CAF50');
  svg.appendChild(startCircle);
  
  // 绘制当前位置
  if (cameraTrajectory.length > 0) {
    const endPoint = cameraTrajectory[cameraTrajectory.length - 1];
    const endX = ((endPoint.x - minX) / rangeX) * (width - 40) + 20;
    const endY = height - ((endPoint.z - minZ) / rangeZ) * (height - 20) - 10;
    
    const endCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    endCircle.setAttribute('cx', endX);
    endCircle.setAttribute('cy', endY);
    endCircle.setAttribute('r', '6');
    endCircle.setAttribute('fill', '#2196F3');
    endCircle.setAttribute('stroke', '#fff');
    endCircle.setAttribute('stroke-width', '2');
    svg.appendChild(endCircle);
  }
}

// 初始化3D场景
function init3DScene() {
  const container = document.getElementById('spatialCanvasContainer');
  const canvas = document.getElementById('spatialCanvas');
  
  // 创建场景
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0f);
  
  // 创建相机
  camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
  camera.position.set(5, 3, 5);
  camera.lookAt(0, 0, 0);
  
  // 创建渲染器
  renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  
  // 添加光源
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambientLight);
  
  const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
  directionalLight.position.set(5, 10, 5);
  scene.add(directionalLight);
  
  // 添加地面网格
  const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
  scene.add(gridHelper);
  
  // 添加轨迹线对象
  const trajectoryGeometry = new THREE.BufferGeometry();
  const trajectoryMaterial = new THREE.LineBasicMaterial({ color: 0x4CAF50 });
  window.trajectoryLine = new THREE.Line(trajectoryGeometry, trajectoryMaterial);
  scene.add(window.trajectoryLine);
  
  // 添加点云对象
  const pointCloudGeometry = new THREE.BufferGeometry();
  const pointCloudMaterial = new THREE.PointsMaterial({
    size: 0.05,
    vertexColors: true
  });
  window.pointCloud = new THREE.Points(pointCloudGeometry, pointCloudMaterial);
  scene.add(window.pointCloud);
  
  // 添加相机位置标记
  const cameraMarkerGeometry = new THREE.SphereGeometry(0.1, 16, 16);
  const cameraMarkerMaterial = new THREE.MeshBasicMaterial({ color: 0x2196F3 });
  window.cameraMarker = new THREE.Mesh(cameraMarkerGeometry, cameraMarkerMaterial);
  scene.add(window.cameraMarker);
  
  // 开始渲染循环
  animate();
  
  // 处理窗口大小变化
  window.addEventListener('resize', onWindowResize);
}

// 动画循环
function animate() {
  animationId = requestAnimationFrame(animate);
  
  // 更新FPS
  const currentTime = performance.now();
  frameCount++;
  if (currentTime - frameTime >= 1000) {
    stats.fps = frameCount;
    document.getElementById('statFps').textContent = stats.fps;
    frameCount = 0;
    frameTime = currentTime;
  }
  
  // 旋转场景（演示用）
  if (mesh) {
    mesh.rotation.y += 0.01;
  }
  
  renderer.render(scene, camera);
}

// 更新3D场景
function update3DScene() {
  // 更新轨迹线
  if (cameraTrajectory.length >= 2 && isPathVisible) {
    const positions = cameraTrajectory.flatMap(p => [p.x, p.y + 0.1, p.z]);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    window.trajectoryLine.geometry.dispose();
    window.trajectoryLine.geometry = geometry;
    window.trajectoryLine.visible = true;
  } else {
    window.trajectoryLine.visible = false;
  }
  
  // 更新点云
  if (pointCloudData.length > 0 && isPointCloudVisible) {
    const positions = pointCloudData.flatMap(p => [p.x, p.y, p.z]);
    const colors = pointCloudData.flatMap(p => p.color);
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    
    window.pointCloud.geometry.dispose();
    window.pointCloud.geometry = geometry;
    window.pointCloud.visible = true;
    
    stats.vertices = pointCloudData.length;
    document.getElementById('statVertices').textContent = stats.vertices;
  } else {
    window.pointCloud.visible = false;
  }
  
  // 更新相机标记位置
  if (cameraTrajectory.length > 0) {
    const lastPos = cameraTrajectory[cameraTrajectory.length - 1];
    window.cameraMarker.position.set(lastPos.x, lastPos.y + 0.2, lastPos.z);
  }
  
  // 更新渲染设置
  if (renderWireframe) {
    window.pointCloud.material.wireframe = true;
  } else {
    window.pointCloud.material.wireframe = false;
  }
  
  // 更新光照
  scene.traverse(function(obj) {
    if (obj.isLight) {
      obj.visible = renderLighting;
    }
  });
}

// 重置3D视角
function reset3DCamera() {
  camera.position.set(5, 3, 5);
  camera.lookAt(0, 0, 0);
}

// 切换轨迹可见性
function togglePathVisibility() {
  isPathVisible = !isPathVisible;
  update3DScene();
}

// 切换点云可见性
function togglePointCloud() {
  isPointCloudVisible = !isPointCloudVisible;
  update3DScene();
}

// 窗口大小变化处理
function onWindowResize() {
  const container = document.getElementById('spatialCanvasContainer');
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}

// 初始化事件监听器
function initEventListeners() {
  // 渲染设置
  document.getElementById('renderWireframe').addEventListener('change', function(e) {
    renderWireframe = e.target.checked;
    update3DScene();
  });
  
  document.getElementById('renderTexture').addEventListener('change', function(e) {
    renderTexture = e.target.checked;
  });
  
  document.getElementById('renderLighting').addEventListener('change', function(e) {
    renderLighting = e.target.checked;
    update3DScene();
  });
}

// 页面切换到空间记忆tab时初始化
function onSpatialTabShow() {
  console.log('[Spatial] onSpatialTabShow called, scene exists:', !!scene);
  if (!scene) {
    initSpatialMemory();
  } else {
    initCamera();
  }
  // 显示overlay
  if (pointCloudData.length === 0) {
    document.getElementById('canvasOverlay').style.display = 'flex';
  } else {
    document.getElementById('canvasOverlay').style.display = 'none';
  }
}

// 页面离开空间记忆tab时清理
function onSpatialTabHide() {
  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
}

// 模块导出（供common.js调用）
window.spatialStartCapture = spatialStartCapture;
window.spatialStopCapture = spatialStopCapture;
window.reset3DCamera = reset3DCamera;
window.togglePathVisibility = togglePathVisibility;
window.togglePointCloud = togglePointCloud;
window.onSpatialTabShow = onSpatialTabShow;
window.onSpatialTabHide = onSpatialTabHide;

// ─── 原有代码（高斯渲染和Scene Graph）保留 ─────────────────────
var _gsViewer = null;
var _gsReady = false;
var _gsLibCache = null;
var GS3D_CDN = 'https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.min.js';

function _gsShowLoading(msg) {
  document.getElementById('ply-empty').style.display = 'none';
  document.getElementById('ply-toolbar').style.display = 'none';
  document.getElementById('ply-loading-bar').style.background = '';
  document.getElementById('ply-loading-bar').style.width = '0%';
  document.getElementById('ply-loading-text').textContent = msg || '加载中...';
  document.getElementById('ply-loading').style.display = 'flex';
}

function _gsDoneLoading(name) {
  document.getElementById('ply-loading-bar').style.width = '100%';
  document.getElementById('ply-filename').textContent = name || 'point_cloud.ply';
  document.getElementById('ply-pt-count').textContent = '高斯渲染';
  setTimeout(function() {
    document.getElementById('ply-loading').style.display = 'none';
    document.getElementById('ply-toolbar').style.display = 'flex';
  }, 250);
}

function _withGS3D(callback) {
  if (_gsLibCache) { callback(_gsLibCache); return; }
  import(GS3D_CDN)
    .then(function(mod) { _gsLibCache = mod; callback(mod); })
    .catch(function(err) {
      document.getElementById('ply-loading-text').textContent = '渲染引擎加载失败: ' + (err.message || String(err));
      document.getElementById('ply-loading-bar').style.background = '#f44336';
      console.error('GS3D load error:', err);
    });
}

function _gsCreateViewer(GS3D) {
  var container = document.getElementById('ply-drop-zone');
  var opts = {
    rootElement: container,
    sharedMemoryForWorkers: false,
    dynamicScene: false,
    cameraUp: [0, -1, 0],
    initialCameraPosition: [0, 0, 5],
    initialCameraLookAt: [0, 0, 0]
  };
  if (GS3D.WebXRMode) opts.webXRMode = GS3D.WebXRMode.None;
  return new GS3D.Viewer(opts);
}

function _gsLoadScene(GS3D, url, name) {
  if (_gsViewer) { try { _gsViewer.dispose(); } catch(_) {} _gsViewer = null; }
  document.getElementById('ply-loading-bar').style.width = '12%';
  document.getElementById('ply-loading-text').textContent = '初始化渲染器...';

  var viewer = _gsCreateViewer(GS3D);
  _gsViewer = viewer;

  var progressCb = function(p) {
    document.getElementById('ply-loading-bar').style.width = Math.round(12 + p * 83) + '%';
    document.getElementById('ply-loading-text').textContent = '加载高斯点云 ' + Math.round(p * 100) + '%';
  };

  var loadFn = (typeof viewer.addSplatScene === 'function') ? viewer.addSplatScene.bind(viewer) : viewer.load.bind(viewer);
  var plyFormat = (GS3D.SceneFormat && GS3D.SceneFormat.Ply !== undefined) ? GS3D.SceneFormat.Ply : 2;
  var loadOpts = { progressCallback: progressCb, format: plyFormat };

  loadFn(url, loadOpts)
    .then(function() {
      _gsDoneLoading(name);
      viewer.start();
    })
    .catch(function(err) {
      document.getElementById('ply-loading-text').textContent = '加载失败: ' + (err.message || String(err));
      document.getElementById('ply-loading-bar').style.background = '#f44336';
      console.error('GS3D scene load error:', err);
    });
}

function initPLYViewer() {
  if (_gsReady) { return; }
  _gsReady = true;
  _gsShowLoading('加载高斯渲染引擎...');
  document.getElementById('ply-loading-bar').style.width = '5%';

  var dz = document.getElementById('ply-drop-zone');
  dz.addEventListener('dragover', function(e) { e.preventDefault(); dz.classList.add('drag-over'); });
  dz.addEventListener('dragleave', function() { dz.classList.remove('drag-over'); });
  dz.addEventListener('drop', function(e) {
    e.preventDefault(); dz.classList.remove('drag-over');
    var f = e.dataTransfer && e.dataTransfer.files[0];
    if (f && f.name.toLowerCase().endsWith('.ply')) {
      _gsShowLoading('读取 ' + f.name + '...');
      _withGS3D(function(GS3D) { _gsLoadScene(GS3D, URL.createObjectURL(f), f.name); });
    }
  });

  _withGS3D(function(GS3D) { _gsLoadScene(GS3D, '/api/ply/point_cloud.ply', 'point_cloud.ply'); });
}

function triggerPLYPicker() {
  var inp = document.getElementById('ply-file-input');
  inp.value = ''; inp.click();
}

function loadPLYFromInput(input) {
  if (!input.files[0]) return;
  var f = input.files[0];
  _gsShowLoading('读取 ' + f.name + '...');
  _withGS3D(function(GS3D) { _gsLoadScene(GS3D, URL.createObjectURL(f), f.name); });
}

function resetPlyCamera() {
  if (!_gsViewer) return;
  try {
    var cam = _gsViewer.camera;
    if (cam) { cam.position.set(0, 0, 5); cam.lookAt(0, 0, 0); }
  } catch(e) {}
}

function setPlyPointSize(v) {}
function onPlyResize() {}

// Scene Graph 部分
var sgInitialized = false;
var sgSimulation = null;
var _sgCategories = [];

function showNodeCard(d) {
  var card = document.getElementById('sg-node-card');
  var img = document.getElementById('sg-node-card-img');
  var wrap = document.getElementById('sg-card-img-wrap');
  var catEl = document.getElementById('sg-node-card-cat');
  var idEl = document.getElementById('sg-node-card-id');
  var descEl = document.getElementById('sg-node-card-desc');
  catEl.textContent = d.category;
  catEl.style.color = getCategoryColor(d.category, _sgCategories);
  idEl.textContent = '# ' + d.id;
  descEl.textContent = d.description || '';
  img.src = '/data/object/' + d.id + '.jpg';
  img.style.display = 'block';
  wrap.style.display = 'block';
  img.onerror = function() { wrap.style.display = 'none'; };
  img.onload = function() { wrap.style.display = 'block'; img.style.display = 'block'; };
  card.style.display = 'block';
  d3.selectAll('.sg-node .node-ring').attr('stroke-width', 2).attr('stroke-opacity', 1);
  d3.selectAll('.sg-node').filter(function(n) { return n.id === d.id; }).select('.node-ring')
    .attr('stroke', '#fff').attr('stroke-width', 3);
}

function hideNodeCard() {
  document.getElementById('sg-node-card').style.display = 'none';
  d3.selectAll('.sg-node .node-ring').attr('stroke-width', 2).attr('stroke-opacity', 1);
}

var CATEGORY_COLORS = [
  '#4FC3F7','#81C784','#FFB74D','#F48FB1','#CE93D8',
  '#80CBC4','#FFCC02','#FF8A65','#90CAF9','#A5D6A7',
  '#FFF176','#BCAAA4','#B39DDB','#80DEEA','#EF9A9A'
];

function getCategoryColor(cat, catList) {
  var idx = catList.indexOf(cat);
  return CATEGORY_COLORS[idx % CATEGORY_COLORS.length];
}

function initSceneGraph() {
  if (sgInitialized) return;
  sgInitialized = true;
  fetch('/api/scene-graph')
    .then(function(r) { return r.json(); })
    .then(function(data) { renderSceneGraph(data); })
    .catch(function(e) {
      document.getElementById('sg-stats').textContent = '加载失败';
      console.error('Scene graph load error:', e);
    });
}

function renderSceneGraph(data) {
  var nodes = data.nodes.map(function(n) { return { id: n.idx, category: n.category, description: n.description, center: n.center }; });
  var edges = data.edges.map(function(e) { return { source: e.obj1, target: e.obj2, relation: e.pretential_relation, location: e.location_relation }; });

  var nodeMap = {};
  nodes.forEach(function(n) { nodeMap[n.id] = n; });
  var validEdges = edges.filter(function(e) { return nodeMap[e.source] && nodeMap[e.target]; });

  var categories = Array.from(new Set(nodes.map(function(n) { return n.category; }))).sort();
  _sgCategories = categories;
  document.getElementById('sg-stats').textContent = nodes.length + ' 节点 · ' + validEdges.length + ' 关系';

  var legendEl = document.getElementById('sg-legend');
  legendEl.innerHTML = '';
  categories.forEach(function(cat) {
    var item = document.createElement('div');
    item.className = 'sg-legend-item';
    var dot = document.createElement('div');
    dot.className = 'sg-legend-dot';
    dot.style.background = getCategoryColor(cat, categories);
    var label = document.createElement('span');
    label.textContent = cat;
    item.appendChild(dot);
    item.appendChild(label);
    legendEl.appendChild(item);
  });

  var container = document.getElementById('sg-container');
  var svg = d3.select('#scene-graph-svg');
  svg.selectAll('*').remove();

  var W = container.clientWidth || 400;
  var H = container.clientHeight || 500;

  var zoom = d3.zoom().scaleExtent([0.2, 4]).on('zoom', function(event) {
    g.attr('transform', event.transform);
  });
  svg.call(zoom);

  var g = svg.append('g');

  var degreeMap = {};
  nodes.forEach(function(n) { degreeMap[n.id] = 0; });
  validEdges.forEach(function(e) { degreeMap[e.source] = (degreeMap[e.source]||0) + 1; degreeMap[e.target] = (degreeMap[e.target]||0) + 1; });

  var radiusMap = {};
  nodes.forEach(function(n) { radiusMap[n.id] = 14 + Math.min((degreeMap[n.id]||0) * 1.2, 8); });

  var defs = svg.append('defs');
  nodes.forEach(function(n) {
    var r = radiusMap[n.id];
    defs.append('clipPath').attr('id', 'nclip-' + n.id)
      .append('circle').attr('cx', 0).attr('cy', 0).attr('r', r);
  });

  var linkSel = g.append('g').selectAll('line')
    .data(validEdges).enter().append('line')
    .attr('class', 'sg-link')
    .attr('stroke', 'rgba(255,255,255,0.2)')
    .attr('stroke-width', 1);

  var nodeSel = g.append('g').selectAll('.sg-node')
    .data(nodes).enter().append('g')
    .attr('class', 'sg-node')
    .style('cursor', 'pointer')
    .call(d3.drag()
      .on('start', function(event, d) { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag', function(event, d) { d.fx = event.x; d.fy = event.y; })
      .on('end', function(event, d) { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
    );

  nodeSel.append('circle')
    .attr('r', function(d) { return radiusMap[d.id]; })
    .attr('fill', function(d) { return getCategoryColor(d.category, categories); })
    .attr('fill-opacity', 0.45)
    .attr('stroke', 'none');

  nodeSel.append('image')
    .attr('href', function(d) { return '/data/object/' + d.id + '.jpg'; })
    .attr('x', function(d) { return -radiusMap[d.id]; })
    .attr('y', function(d) { return -radiusMap[d.id]; })
    .attr('width', function(d) { return radiusMap[d.id] * 2; })
    .attr('height', function(d) { return radiusMap[d.id] * 2; })
    .attr('preserveAspectRatio', 'xMidYMid slice')
    .attr('clip-path', function(d) { return 'url(#nclip-' + d.id + ')'; })
    .style('pointer-events', 'none');

  nodeSel.append('circle')
    .attr('class', 'node-ring')
    .attr('r', function(d) { return radiusMap[d.id]; })
    .attr('fill', 'none')
    .attr('stroke', function(d) { return getCategoryColor(d.category, categories); })
    .attr('stroke-width', 2);

  svg.on('click', function() { hideNodeCard(); });

  var tooltip = document.getElementById('sg-tooltip');
  nodeSel
    .on('click', function(event, d) {
      event.stopPropagation();
      showNodeCard(d);
    })
    .on('mouseover', function(event, d) {
      tooltip.style.display = 'block';
      tooltip.innerHTML = '<div class="tt-cat" style="color:' + getCategoryColor(d.category, categories) + '">' + d.category + ' #' + d.id + '</div>'
        + '<div class="tt-desc">' + (d.description || '') + '</div>'
        + (d.center ? '<div style="color:#777;font-size:10px;margin-top:4px;">位置: (' + d.center.map(function(v){return v.toFixed(2);}).join(', ') + ')</div>' : '');
    })
    .on('mousemove', function(event) {
      var rect = container.getBoundingClientRect();
      var tx = event.clientX - rect.left + 12;
      var ty = event.clientY - rect.top - 20;
      if (tx + 230 > rect.width) tx = event.clientX - rect.left - 230;
      tooltip.style.left = tx + 'px';
      tooltip.style.top = ty + 'px';
    })
    .on('mouseout', function() { tooltip.style.display = 'none'; });

  var simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(validEdges).id(function(d) { return d.id; }).distance(70).strength(0.4))
    .force('charge', d3.forceManyBody().strength(-200))
    .force('center', d3.forceCenter(W / 2, H / 2))
    .force('collision', d3.forceCollide().radius(function(d) { return radiusMap[d.id] + 4; }))
    .on('tick', function() {
      linkSel
        .attr('x1', function(d) { return d.source.x; })
        .attr('y1', function(d) { return d.source.y; })
        .attr('x2', function(d) { return d.target.x; })
        .attr('y2', function(d) { return d.target.y; });
      nodeSel.attr('transform', function(d) { return 'translate(' + d.x + ',' + d.y + ')'; });
    });

  sgSimulation = simulation;
}