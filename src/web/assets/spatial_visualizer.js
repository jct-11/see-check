// spatial_visualizer.js - Three.js 3D 可视化模块
// 职责：Three.js 场景管理、点云渲染、相机控制、相机轨迹可视化

// Three.js 模块对象
let THREE = null;
let OrbitControls = null;

// Three.js 场景对象
let scene = null;
let camera = null;
let renderer = null;
let animationId = null;

// 点云对象
let pointCloud = null;
let trajectoryTube = null;
let cameraFrustums = [];

// 渲染状态
let stats = { fps: 0, vertices: 0 };
let frameTime = 0;
let frameCount = 0;

// GUI 控制参数
let guiPointSize = 0;
let guiDownsample = 1;
let guiShowCamera = true;
let guiCamSize = 0.05;
let guiConfThreshold = 50.0;

// 回调函数
let onStatsUpdate = null;

// 设置回调
function setVisualizerCallbacks(callbacks) {
  if (callbacks.onStatsUpdate) onStatsUpdate = callbacks.onStatsUpdate;
}

// 加载 Three.js
async function loadThreeJS() {
  if (THREE) return THREE;
  try {
    THREE = await import('three');
    const { OrbitControls: OC } = await import('three/addons/controls/OrbitControls.js');
    OrbitControls = OC;
    console.log('[Spatial Visualizer] Three.js and OrbitControls loaded successfully');
    return THREE;
  } catch (err) {
    console.error('[Spatial Visualizer] Failed to load Three.js:', err);
    throw err;
  }
}

// 初始化 3D 场景
async function init3DScene() {
  if (!THREE) await loadThreeJS();

  const container = document.getElementById('spatialCanvasContainer');
  const canvas = document.getElementById('spatialCanvas');

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf0f0f0);

  camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.001, 10000);
  camera.position.set(3, 2, 3);
  camera.lookAt(0, 0, 0);
  camera.up.set(0, -1, 0);

  renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: false });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const gridHelper = new THREE.GridHelper(20, 20, 0xcccccc, 0xdddddd);
  scene.add(gridHelper);

  const axesHelper = new THREE.AxesHelper(1);
  scene.add(axesHelper);

  const pointCloudGeometry = new THREE.BufferGeometry();
  const pointCloudMaterial = new THREE.PointsMaterial({
    size: 0.005,
    vertexColors: true,
    sizeAttenuation: true,
    transparent: false,
    opacity: 1.0,
    depthWrite: true,
    depthTest: true
  });
  pointCloud = new THREE.Points(pointCloudGeometry, pointCloudMaterial);
  window.pointCloud = pointCloud;
  scene.add(pointCloud);

  window.controls = new OrbitControls(camera, renderer.domElement);
  window.controls.enableDamping = true;
  window.controls.dampingFactor = 0.08;
  window.controls.minDistance = 0.01;
  window.controls.maxDistance = 5000;
  window.controls.target.set(0, 0, 0);

  animate();
  window.addEventListener('resize', onWindowResize);

  console.log('[Spatial Visualizer] 3D scene initialized');
}

// 窗口大小调整
function onWindowResize() {
  const container = document.getElementById('spatialCanvasContainer');
  if (!container || !camera || !renderer) return;

  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}

// 动画循环
function animate() {
  animationId = requestAnimationFrame(animate);

  const currentTime = performance.now();
  frameCount++;
  if (currentTime - frameTime >= 1000) {
    stats.fps = frameCount;
    if (onStatsUpdate) {
      onStatsUpdate({ fps: stats.fps });
    }
    frameCount = 0;
    frameTime = currentTime;
  }

  if (window.controls) {
    window.controls.update();
  }

  renderer.render(scene, camera);
}

// 更新点云场景
function updateScene(data) {
  if (!THREE || !scene || !pointCloud) {
    console.warn('[Spatial Visualizer] Scene not initialized');
    return;
  }

  const { points, colors, cameraPoses, sceneCenter, batchId } = data;

  if (!points || points.length === 0) {
    pointCloud.visible = false;
    return;
  }

  const numPoints = points.length / 3;
  let positions;
  if (points instanceof Float32Array) {
    positions = points;
  } else {
    positions = new Float32Array(points);
  }

  // 处理颜色
  const colorsOut = new Float32Array(points.length);

  if (colors && colors.length === points.length) {
    if (colors instanceof Uint8Array) {
      for (let i = 0; i < colors.length; i++) {
        colorsOut[i] = colors[i] / 255.0;
      }
    } else {
      for (let i = 0; i < colors.length; i++) {
        colorsOut[i] = Math.max(0, Math.min(1, colors[i] / 255.0));
      }
    }
  } else if (colors && colors.length > 0) {
    const colorLen = Math.min(colors.length, points.length);
    for (let i = 0; i < colorLen; i++) {
      colorsOut[i] = Math.max(0, Math.min(1, colors[i] / 255.0));
    }
    for (let i = colorLen; i < colorsOut.length; i += 3) {
      colorsOut[i] = 0.5;
      colorsOut[i + 1] = 0.5;
      colorsOut[i + 2] = 0.5;
    }
  } else {
    for (let i = 0; i < colorsOut.length; i += 3) {
      colorsOut[i] = 0.5;
      colorsOut[i + 1] = 0.5;
      colorsOut[i + 2] = 0.5;
    }
  }

  // 下采样
  let dsPositions = positions;
  let dsColors = colorsOut;
  if (guiDownsample > 1) {
    const dsLen = Math.floor(numPoints / guiDownsample) * 3;
    dsPositions = new Float32Array(dsLen);
    dsColors = new Float32Array(dsLen);
    for (let i = 0, j = 0; i < numPoints && j < dsLen; i += guiDownsample, j += 3) {
      const si = i * 3;
      dsPositions[j] = positions[si];
      dsPositions[j + 1] = positions[si + 1];
      dsPositions[j + 2] = positions[si + 2];
      dsColors[j] = colorsOut[si];
      dsColors[j + 1] = colorsOut[si + 1];
      dsColors[j + 2] = colorsOut[si + 2];
    }
  }

  // 创建几何体
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(dsPositions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(dsColors, 3));

  if (pointCloud.geometry) {
    pointCloud.geometry.dispose();
  }
  pointCloud.geometry = geometry;

  // 计算场景包围盒
  geometry.computeBoundingBox();
  const bbox = geometry.boundingBox;
  const center = new THREE.Vector3();
  bbox.getCenter(center);
  const size = new THREE.Vector3();
  bbox.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z, 0.001);

  // 自动计算点大小
  const autoPointSize = maxDim * 0.002;
  const finalPointSize = guiPointSize > 0 ? guiPointSize : autoPointSize;
  pointCloud.material.size = finalPointSize;
  pointCloud.material.vertexColors = true;
  pointCloud.material.sizeAttenuation = true;
  pointCloud.material.transparent = false;
  pointCloud.material.opacity = 1.0;
  pointCloud.material.depthWrite = true;
  pointCloud.material.depthTest = true;
  pointCloud.visible = true;

  // 调整相机位置（仅第一批数据）
  if (batchId <= 1 && camera && window.controls) {
    camera.position.set(
      center.x + maxDim * 0.7,
      center.y + maxDim * 0.5,
      center.z + maxDim * 0.7
    );
    camera.lookAt(center);
    window.controls.target.copy(center);
    window.controls.update();
  }

  // 更新统计
  stats.vertices = numPoints;
  if (onStatsUpdate) {
    onStatsUpdate({ vertices: stats.vertices });
  }

  // 更新相机可视化
  if (guiShowCamera && cameraPoses && cameraPoses.length > 0) {
    updateCameraTrajectory(cameraPoses);
    updateCameraFrustums(cameraPoses);
  }

  // 隐藏加载提示
  const overlay = document.getElementById('canvasOverlay');
  if (overlay) overlay.style.display = 'none';
}

// 更新相机轨迹管
function updateCameraTrajectory(cameraPoses) {
  if (!THREE || !scene || !cameraPoses || cameraPoses.length < 2) return;

  // 清除旧的轨迹管
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

  // 创建 CatmullRom 样条曲线
  const curve = new THREE.CatmullRomCurve3(points);
  const numSegments = Math.max(20, points.length * 2);
  const tubeRadius = Math.max(0.005, guiCamSize * 0.1);
  const geometry = new THREE.TubeGeometry(curve, numSegments, tubeRadius, 8, false);

  // 创建渐变颜色
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

// 更新相机视锥体
function updateCameraFrustums(cameraPoses) {
  if (!THREE || !scene) return;

  // 清除旧的视锥体
  cameraFrustums.forEach(obj => {
    scene.remove(obj);
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

  if (!guiShowCamera || !cameraPoses || cameraPoses.length === 0) return;

  const step = Math.max(1, Math.floor(cameraPoses.length / 30));

  cameraPoses.forEach((pose, index) => {
    if (index % step !== 0 && index !== cameraPoses.length - 1) return;

    const pos = pose.position;
    const R = pose.R;
    const focal = pose.focal;
    const pp = pose.pp;

    const hue = (index / cameraPoses.length) * 0.8 + 0.15;
    const frustumColor = new THREE.Color().setHSL(hue, 0.8, 0.6);

    // 创建相机位置标记球
    const markerGeom = new THREE.SphereGeometry(guiCamSize * 0.3, 12, 12);
    const markerMat = new THREE.MeshBasicMaterial({ color: frustumColor, transparent: true, opacity: 0.9 });
    const marker = new THREE.Mesh(markerGeom, markerMat);
    marker.position.set(pos[0], pos[1], pos[2]);
    scene.add(marker);
    cameraFrustums.push(marker);

    // 创建视锥体线框
    if (R && focal && pp) {
      const frustumScale = guiCamSize;
      const fov = 2 * Math.atan2(pp[0] || 256, focal || 500);
      const aspect = (pp[0] || 256) / (pp[1] || 256);

      const h = Math.tan(fov / 2) * frustumScale;
      const w = h * aspect;
      const d = frustumScale;

      const frustumVerts = new Float32Array([
        0, 0, 0,    -w, -h, d,
        0, 0, 0,     w, -h, d,
        0, 0, 0,     w,  h, d,
        0, 0, 0,    -w,  h, d,
        -w, -h, d,   w, -h, d,
        w, -h, d,    w,  h, d,
        w,  h, d,   -w,  h, d,
        -w,  h, d,  -w, -h, d
      ]);

      const frustumGeom = new THREE.BufferGeometry();
      frustumGeom.setAttribute('position', new THREE.BufferAttribute(frustumVerts, 3));
      const frustumMat = new THREE.LineBasicMaterial({ color: frustumColor, transparent: true, opacity: 0.7 });
      const frustumLines = new THREE.LineSegments(frustumGeom, frustumMat);

      const rotMatrix = new THREE.Matrix4();
      if (R && R.length === 3) {
        rotMatrix.set(
          R[0][0], R[0][1], R[0][2], 0,
          R[1][0], R[1][1], R[1][2], 0,
          R[2][0], R[2][1], R[2][2], 0,
          0, 0, 0, 1
        );
      }
      frustumLines.setRotationFromMatrix(rotMatrix);
      frustumLines.position.set(pos[0], pos[1], pos[2]);

      scene.add(frustumLines);
      cameraFrustums.push(frustumLines);
    }
  });
}

// 重置 3D 相机
function reset3DCamera() {
  if (!camera || !window.controls) return;

  if (pointCloud && pointCloud.geometry && pointCloud.geometry.attributes.position) {
    const bbox = new THREE.Box3().setFromObject(pointCloud);
    const center = new THREE.Vector3();
    bbox.getCenter(center);
    const size = new THREE.Vector3();
    bbox.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);

    camera.position.set(
      center.x + maxDim * 0.7,
      center.y + maxDim * 0.5,
      center.z + maxDim * 0.7
    );
    camera.lookAt(center);
    window.controls.target.copy(center);
  } else {
    camera.position.set(3, 2, 3);
    camera.lookAt(0, 0, 0);
    window.controls.target.set(0, 0, 0);
  }
  window.controls.update();
}

// 设置视角方向
function setViewDirection(direction) {
  if (!camera || !window.controls) return;

  let center = new THREE.Vector3(0, 0, 0);
  let scale = 3;

  if (pointCloud && pointCloud.geometry && pointCloud.geometry.attributes.position) {
    const bbox = new THREE.Box3().setFromObject(pointCloud);
    bbox.getCenter(center);
    const size = new THREE.Vector3();
    bbox.getSize(size);
    scale = Math.max(size.x, size.y, size.z) * 1.5;
  }

  camera.position.set(
    center.x + direction[0] * scale,
    center.y + direction[1] * scale,
    center.z + direction[2] * scale
  );
  camera.lookAt(center);
  window.controls.target.copy(center);
  window.controls.update();
}

// 设置视角到第一个相机
function setViewToFirstCamera() {
  if (cameraFrustums.length === 0) return;

  const firstFrustum = cameraFrustums.find(obj => obj instanceof THREE.Mesh);
  if (firstFrustum) {
    camera.position.set(
      firstFrustum.position.x + 0.5,
      firstFrustum.position.y + 0.5,
      firstFrustum.position.z + 0.5
    );
    camera.lookAt(firstFrustum.position);
    window.controls.target.copy(firstFrustum.position);
    window.controls.update();
  }
}

// 切换点云可见性
function togglePointCloud() {
  if (pointCloud) {
    pointCloud.visible = !pointCloud.visible;
    const btn = document.getElementById('togglePointCloudBtn');
    if (btn) btn.textContent = pointCloud.visible ? '☁️ 点云' : '☁️ 点云(隐藏)';
  }
}

// 设置 GUI 参数
function setGuiPointSize(value) {
  guiPointSize = parseFloat(value);
}

function setGuiDownsample(value) {
  guiDownsample = parseInt(value);
}

function setGuiShowCamera(value) {
  guiShowCamera = value;
}

function setGuiCamSize(value) {
  guiCamSize = parseFloat(value);
}

// 获取渲染器
function getRenderer() {
  return renderer;
}

// 获取场景
function getScene() {
  return scene;
}

// 获取相机
function getCamera() {
  return camera;
}

function getCurrentSceneData() {
  var data = {
    points: [],
    colors: [],
    cameraPoses: []
  };

  if (pointCloud && pointCloud.geometry && pointCloud.geometry.attributes.position) {
    var posAttr = pointCloud.geometry.attributes.position;
    var colAttr = pointCloud.geometry.attributes.color;
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

// 导出模块
window.SpatialVisualizer = {
  init: init3DScene,
  updateScene: updateScene,
  resetCamera: reset3DCamera,
  setViewDirection: setViewDirection,
  setViewToFirstCamera: setViewToFirstCamera,
  togglePointCloud: togglePointCloud,
  setCallbacks: setVisualizerCallbacks,
  setPointSize: setGuiPointSize,
  setDownsample: setGuiDownsample,
  setShowCamera: setGuiShowCamera,
  setCamSize: setGuiCamSize,
  getRenderer: getRenderer,
  getScene: getScene,
  getCamera: getCamera,
  getCurrentSceneData: getCurrentSceneData
};
