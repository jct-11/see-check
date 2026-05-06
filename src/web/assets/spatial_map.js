// spatial_map.js - 2D鸟瞰地图模块
// 职责：将3D点云投影到XZ平面、绘制轨迹、支持缩放/平移/交互

let mapCanvas = null;
let mapCtx = null;
let mapWidth = 0;
let mapHeight = 0;

let mapPoints = [];
let mapColors = [];
let mapTrajectory = [];
let mapCameraPoses = [];

let mapScale = 1.0;
let mapOffsetX = 0;
let mapOffsetY = 0;
let mapMinX = -5;
let mapMaxX = 5;
let mapMinZ = -5;
let mapMaxZ = 5;

let showGrid = true;
let showHeatmap = false;
let showTrajectory = true;
let showPointCloud = true;
let showCameraMarkers = true;

let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;
let dragStartOffsetX = 0;
let dragStartOffsetY = 0;

let heatmapCanvas = null;
let heatmapCtx = null;
let heatmapDirty = true;

let onMapClick = null;
let onMapHover = null;

const HEATMAP_CELL_SIZE = 8;
const POINT_RENDER_SIZE = 2;

function initMap() {
  mapCanvas = document.getElementById('spatial2DMapCanvas');
  if (!mapCanvas) {
    console.error('[SpatialMap] Canvas element not found');
    return;
  }

  mapCtx = mapCanvas.getContext('2d');
  resizeMapCanvas();

  mapCanvas.addEventListener('mousedown', onMouseDown);
  mapCanvas.addEventListener('mousemove', onMouseMove);
  mapCanvas.addEventListener('mouseup', onMouseUp);
  mapCanvas.addEventListener('mouseleave', onMouseUp);
  mapCanvas.addEventListener('wheel', onWheel, { passive: false });

  window.addEventListener('resize', resizeMapCanvas);

  console.log('[SpatialMap] 2D map initialized');
}

function resizeMapCanvas() {
  const container = document.getElementById('spatial2DMapContainer');
  if (!container || !mapCanvas) return;

  mapWidth = container.clientWidth;
  mapHeight = container.clientHeight;
  mapCanvas.width = mapWidth * window.devicePixelRatio;
  mapCanvas.height = mapHeight * window.devicePixelRatio;
  mapCanvas.style.width = mapWidth + 'px';
  mapCanvas.style.height = mapHeight + 'px';
  mapCtx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);

  if (heatmapCanvas) {
    heatmapCanvas.width = mapWidth;
    heatmapCanvas.height = mapHeight;
  }

  heatmapDirty = true;
  renderMap();
}

function worldToScreen(wx, wz) {
  const rangeX = mapMaxX - mapMinX || 1;
  const rangeZ = mapMaxZ - mapMinZ || 1;
  const padding = 40;
  const drawW = mapWidth - padding * 2;
  const drawH = mapHeight - padding * 2;
  const baseScale = Math.min(drawW / rangeX, drawH / rangeZ);

  const sx = padding + (wx - mapMinX) / rangeX * drawW * mapScale + mapOffsetX;
  const sy = padding + (wz - mapMinZ) / rangeZ * drawH * mapScale + mapOffsetY;

  return { x: sx, y: sy };
}

function screenToWorld(sx, sy) {
  const rangeX = mapMaxX - mapMinX || 1;
  const rangeZ = mapMaxZ - mapMinZ || 1;
  const padding = 40;
  const drawW = mapWidth - padding * 2;
  const drawH = mapHeight - padding * 2;

  const wx = ((sx - mapOffsetX - padding) / drawW) * rangeX / mapScale + mapMinX;
  const wz = ((sy - mapOffsetY - padding) / drawH) * rangeZ / mapScale + mapMinZ;

  return { x: wx, z: wz };
}

function computeBounds() {
  if (mapPoints.length === 0 && mapTrajectory.length === 0) {
    mapMinX = -5; mapMaxX = 5;
    mapMinZ = -5; mapMaxZ = 5;
    return;
  }

  let minX = Infinity, maxX = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;

  for (let i = 0; i < mapPoints.length; i += 3) {
    const x = mapPoints[i];
    const z = mapPoints[i + 2];
    if (isFinite(x) && isFinite(z)) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    }
  }

  for (let i = 0; i < mapTrajectory.length; i++) {
    const p = mapTrajectory[i];
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.z < minZ) minZ = p.z;
    if (p.z > maxZ) maxZ = p.z;
  }

  const padX = Math.max((maxX - minX) * 0.15, 0.5);
  const padZ = Math.max((maxZ - minZ) * 0.15, 0.5);
  mapMinX = minX - padX;
  mapMaxX = maxX + padX;
  mapMinZ = minZ - padZ;
  mapMaxZ = maxZ + padZ;
}

function updateMapData(data) {
  if (!data) return;

  if (data.points && data.points.length > 0) {
    mapPoints = data.points instanceof Float32Array ? Array.from(data.points) : data.points;
  }
  if (data.colors && data.colors.length > 0) {
    mapColors = data.colors instanceof Uint8Array ? Array.from(data.colors) : data.colors;
  }
  if (data.cameraPoses && data.cameraPoses.length > 0) {
    mapCameraPoses = data.cameraPoses;
    mapTrajectory = data.cameraPoses.map(function(pose) {
      if (pose.position) {
        return { x: pose.position[0], y: pose.position[1], z: pose.position[2], frameId: pose.frame_id };
      }
      return null;
    }).filter(Boolean);
  }

  computeBounds();
  heatmapDirty = true;
  renderMap();

  const overlay = document.getElementById('mapOverlay');
  if (overlay && mapPoints.length > 0) {
    overlay.style.display = 'none';
  }
}

function renderMap() {
  if (!mapCtx || mapWidth === 0 || mapHeight === 0) return;

  mapCtx.clearRect(0, 0, mapWidth, mapHeight);

  drawBackground();

  if (showGrid) drawGrid();

  if (showHeatmap && mapPoints.length > 0) drawHeatmap();

  if (showPointCloud && mapPoints.length > 0) drawPoints();

  if (showTrajectory && mapTrajectory.length > 0) drawTrajectory();

  if (showCameraMarkers && mapTrajectory.length > 0) drawCameraMarkers();

  drawAxisLabels();
}

function drawBackground() {
  mapCtx.fillStyle = '#0d1117';
  mapCtx.fillRect(0, 0, mapWidth, mapHeight);

  const gradient = mapCtx.createRadialGradient(
    mapWidth / 2, mapHeight / 2, 0,
    mapWidth / 2, mapHeight / 2, Math.max(mapWidth, mapHeight) * 0.6
  );
  gradient.addColorStop(0, 'rgba(33, 150, 243, 0.03)');
  gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
  mapCtx.fillStyle = gradient;
  mapCtx.fillRect(0, 0, mapWidth, mapHeight);
}

function drawGrid() {
  const rangeX = mapMaxX - mapMinX;
  const rangeZ = mapMaxZ - mapMinZ;
  const maxRange = Math.max(rangeX, rangeZ);

  let step = 1;
  if (maxRange > 50) step = 10;
  else if (maxRange > 20) step = 5;
  else if (maxRange > 10) step = 2;
  else if (maxRange > 2) step = 0.5;
  else step = 0.1;

  mapCtx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
  mapCtx.lineWidth = 0.5;
  mapCtx.font = '10px monospace';
  mapCtx.fillStyle = 'rgba(255, 255, 255, 0.25)';

  const startX = Math.floor(mapMinX / step) * step;
  const startZ = Math.floor(mapMinZ / step) * step;

  for (let wx = startX; wx <= mapMaxX; wx += step) {
    const s = worldToScreen(wx, 0);
    mapCtx.beginPath();
    mapCtx.moveTo(s.x, 0);
    mapCtx.lineTo(s.x, mapHeight);
    mapCtx.stroke();

    if (Math.abs(wx) > step * 0.01) {
      mapCtx.fillText(wx.toFixed(step < 1 ? 1 : 0), s.x + 2, mapHeight - 5);
    }
  }

  for (let wz = startZ; wz <= mapMaxZ; wz += step) {
    const s = worldToScreen(0, wz);
    mapCtx.beginPath();
    mapCtx.moveTo(0, s.y);
    mapCtx.lineTo(mapWidth, s.y);
    mapCtx.stroke();

    if (Math.abs(wz) > step * 0.01) {
      mapCtx.fillText(wz.toFixed(step < 1 ? 1 : 0), 5, s.y - 2);
    }
  }

  const origin = worldToScreen(0, 0);
  mapCtx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
  mapCtx.lineWidth = 1;
  mapCtx.beginPath();
  mapCtx.moveTo(origin.x, 0);
  mapCtx.lineTo(origin.x, mapHeight);
  mapCtx.stroke();
  mapCtx.beginPath();
  mapCtx.moveTo(0, origin.y);
  mapCtx.lineTo(mapWidth, origin.y);
  mapCtx.stroke();
}

function drawPoints() {
  const numPoints = mapPoints.length / 3;
  const hasColors = mapColors.length >= numPoints * 3;

  for (let i = 0; i < numPoints; i++) {
    const wx = mapPoints[i * 3];
    const wz = mapPoints[i * 3 + 2];

    if (!isFinite(wx) || !isFinite(wz)) continue;

    const s = worldToScreen(wx, wz);

    if (s.x < -10 || s.x > mapWidth + 10 || s.y < -10 || s.y > mapHeight + 10) continue;

    let r = 100, g = 180, b = 255;
    if (hasColors) {
      r = mapColors[i * 3];
      g = mapColors[i * 3 + 1];
      b = mapColors[i * 3 + 2];
      if (r > 1 || g > 1 || b > 1) {
        r = Math.round(r);
        g = Math.round(g);
        b = Math.round(b);
      } else {
        r = Math.round(r * 255);
        g = Math.round(g * 255);
        b = Math.round(b * 255);
      }
    }

    const height = mapPoints[i * 3 + 1];
    if (!hasColors && isFinite(height)) {
      const range = mapMaxZ - mapMinZ || 1;
      const t = Math.max(0, Math.min(1, (height - mapMinZ) / range));
      r = Math.round(30 + t * 200);
      g = Math.round(100 + (1 - t) * 155);
      b = Math.round(255 - t * 100);
    }

    mapCtx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.7)';
    mapCtx.fillRect(s.x - POINT_RENDER_SIZE / 2, s.y - POINT_RENDER_SIZE / 2, POINT_RENDER_SIZE, POINT_RENDER_SIZE);
  }
}

function drawHeatmap() {
  if (heatmapDirty || !heatmapCanvas) {
    if (!heatmapCanvas) {
      heatmapCanvas = document.createElement('canvas');
      heatmapCtx = heatmapCanvas.getContext('2d');
    }
    heatmapCanvas.width = mapWidth;
    heatmapCanvas.height = mapHeight;
    heatmapCtx.clearRect(0, 0, mapWidth, mapHeight);

    const cellW = HEATMAP_CELL_SIZE;
    const cellH = HEATMAP_CELL_SIZE;
    const cols = Math.ceil(mapWidth / cellW);
    const rows = Math.ceil(mapHeight / cellH);
    const counts = new Float32Array(cols * rows);

    const numPoints = mapPoints.length / 3;
    let maxCount = 0;

    for (let i = 0; i < numPoints; i++) {
      const wx = mapPoints[i * 3];
      const wz = mapPoints[i * 3 + 2];
      if (!isFinite(wx) || !isFinite(wz)) continue;

      const s = worldToScreen(wx, wz);
      const col = Math.floor(s.x / cellW);
      const row = Math.floor(s.y / cellH);

      if (col >= 0 && col < cols && row >= 0 && row < rows) {
        counts[row * cols + col]++;
        if (counts[row * cols + col] > maxCount) {
          maxCount = counts[row * cols + col];
        }
      }
    }

    if (maxCount > 0) {
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          const count = counts[row * cols + col];
          if (count === 0) continue;

          const t = count / maxCount;
          let r, g, b;
          if (t < 0.25) {
            r = 0; g = Math.round(t * 4 * 255); b = 255;
          } else if (t < 0.5) {
            r = 0; g = 255; b = Math.round((1 - (t - 0.25) * 4) * 255);
          } else if (t < 0.75) {
            r = Math.round((t - 0.5) * 4 * 255); g = 255; b = 0;
          } else {
            r = 255; g = Math.round((1 - (t - 0.75) * 4) * 255); b = 0;
          }

          heatmapCtx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + (0.15 + t * 0.35) + ')';
          heatmapCtx.fillRect(col * cellW, row * cellH, cellW, cellH);
        }
      }
    }

    heatmapDirty = false;
  }

  mapCtx.drawImage(heatmapCanvas, 0, 0);
}

function drawTrajectory() {
  if (mapTrajectory.length < 2) return;

  mapCtx.lineWidth = 2;
  mapCtx.lineCap = 'round';
  mapCtx.lineJoin = 'round';

  for (let i = 1; i < mapTrajectory.length; i++) {
    const p0 = worldToScreen(mapTrajectory[i - 1].x, mapTrajectory[i - 1].z);
    const p1 = worldToScreen(mapTrajectory[i].x, mapTrajectory[i].z);

    const t = i / mapTrajectory.length;
    const hue = t * 0.8 + 0.15;
    const color = hslToRgb(hue, 0.9, 0.6);

    mapCtx.strokeStyle = 'rgba(' + color.r + ',' + color.g + ',' + color.b + ',0.8)';
    mapCtx.beginPath();
    mapCtx.moveTo(p0.x, p0.y);
    mapCtx.lineTo(p1.x, p1.y);
    mapCtx.stroke();
  }

  mapCtx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
  mapCtx.lineWidth = 6;
  mapCtx.beginPath();
  const first = worldToScreen(mapTrajectory[0].x, mapTrajectory[0].z);
  mapCtx.moveTo(first.x, first.y);
  for (let i = 1; i < mapTrajectory.length; i++) {
    const p = worldToScreen(mapTrajectory[i].x, mapTrajectory[i].z);
    mapCtx.lineTo(p.x, p.y);
  }
  mapCtx.stroke();
}

function drawCameraMarkers() {
  const step = Math.max(1, Math.floor(mapTrajectory.length / 30));

  for (let i = 0; i < mapTrajectory.length; i++) {
    if (i % step !== 0 && i !== 0 && i !== mapTrajectory.length - 1) continue;

    const pt = mapTrajectory[i];
    const s = worldToScreen(pt.x, pt.z);
    const isStart = i === 0;
    const isEnd = i === mapTrajectory.length - 1;

    if (isStart || isEnd) {
      mapCtx.beginPath();
      mapCtx.arc(s.x, s.y, 6, 0, Math.PI * 2);
      mapCtx.fillStyle = isStart ? '#4CAF50' : '#f44336';
      mapCtx.fill();
      mapCtx.strokeStyle = 'rgba(255,255,255,0.8)';
      mapCtx.lineWidth = 2;
      mapCtx.stroke();

      mapCtx.font = 'bold 10px sans-serif';
      mapCtx.fillStyle = '#fff';
      mapCtx.textAlign = 'center';
      mapCtx.fillText(isStart ? 'S' : 'E', s.x, s.y + 3.5);
    } else {
      const t = i / mapTrajectory.length;
      const hue = t * 0.8 + 0.15;
      const color = hslToRgb(hue, 0.9, 0.6);

      mapCtx.beginPath();
      mapCtx.arc(s.x, s.y, 3, 0, Math.PI * 2);
      mapCtx.fillStyle = 'rgba(' + color.r + ',' + color.g + ',' + color.b + ',0.8)';
      mapCtx.fill();
    }

    if (mapCameraPoses[i] && mapCameraPoses[i].R) {
      drawCameraFrustum(s, mapCameraPoses[i], i);
    }
  }
}

function drawCameraFrustum(screenPos, pose, index) {
  const R = pose.R;
  if (!R || R.length < 3) return;

  const frustumSize = 8;
  let fwdX = 0, fwdZ = 1;
  if (R[0] && R[2]) {
    fwdX = -R[2][0];
    fwdZ = -R[2][2];
  }
  const len = Math.sqrt(fwdX * fwdX + fwdZ * fwdZ) || 1;
  fwdX /= len;
  fwdZ /= len;

  const perpX = -fwdZ;
  const perpZ = fwdX;

  const tipX = screenPos.x;
  const tipY = screenPos.y;
  const baseCX = tipX + fwdX * frustumSize;
  const baseCY = tipY + fwdZ * frustumSize;
  const halfW = frustumSize * 0.4;

  const t = index / Math.max(mapCameraPoses.length, 1);
  const hue = t * 0.8 + 0.15;
  const color = hslToRgb(hue, 0.8, 0.6);

  mapCtx.strokeStyle = 'rgba(' + color.r + ',' + color.g + ',' + color.b + ',0.5)';
  mapCtx.lineWidth = 1;
  mapCtx.beginPath();
  mapCtx.moveTo(tipX, tipY);
  mapCtx.lineTo(baseCX + perpX * halfW, baseCY + perpZ * halfW);
  mapCtx.moveTo(tipX, tipY);
  mapCtx.lineTo(baseCX - perpX * halfW, baseCY - perpZ * halfW);
  mapCtx.stroke();
}

function drawAxisLabels() {
  const origin = worldToScreen(0, 0);

  mapCtx.font = 'bold 12px sans-serif';
  mapCtx.textAlign = 'center';

  mapCtx.fillStyle = 'rgba(244, 67, 54, 0.7)';
  mapCtx.fillText('X+', Math.min(mapWidth - 15, origin.x + 40), origin.y - 5);

  mapCtx.fillStyle = 'rgba(76, 175, 80, 0.7)';
  mapCtx.fillText('Z+', origin.x + 12, Math.max(15, origin.y - 30));

  mapCtx.fillStyle = 'rgba(255,255,255,0.3)';
  mapCtx.font = '10px sans-serif';
  mapCtx.fillText('O', origin.x + 10, origin.y + 12);
}

function hslToRgb(h, s, l) {
  let r, g, b;
  if (s === 0) {
    r = g = b = l;
  } else {
    function hue2rgb(p, q, t) {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1/6) return p + (q - p) * 6 * t;
      if (t < 1/2) return q;
      if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
      return p;
    }
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1/3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1/3);
  }
  return { r: Math.round(r * 255), g: Math.round(g * 255), b: Math.round(b * 255) };
}

// ============== 交互 ==============

function onMouseDown(e) {
  isDragging = true;
  dragStartX = e.clientX;
  dragStartY = e.clientY;
  dragStartOffsetX = mapOffsetX;
  dragStartOffsetY = mapOffsetY;
  mapCanvas.style.cursor = 'grabbing';
}

function onMouseMove(e) {
  const rect = mapCanvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;
  const world = screenToWorld(sx, sy);

  const coordInfo = document.getElementById('mapCoordInfo');
  if (coordInfo) {
    coordInfo.textContent = 'X: ' + world.x.toFixed(2) + '  Z: ' + world.z.toFixed(2);
  }

  if (isDragging) {
    mapOffsetX = dragStartOffsetX + (e.clientX - dragStartX);
    mapOffsetY = dragStartOffsetY + (e.clientY - dragStartY);
    heatmapDirty = true;
    renderMap();
  }

  if (onMapHover) {
    onMapHover(world.x, world.z, sx, sy);
  }
}

function onMouseUp(e) {
  if (isDragging) {
    isDragging = false;
    mapCanvas.style.cursor = 'grab';

    if (Math.abs(e.clientX - dragStartX) < 3 && Math.abs(e.clientY - dragStartY) < 3) {
      const rect = mapCanvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const world = screenToWorld(sx, sy);
      if (onMapClick) {
        onMapClick(world.x, world.z, sx, sy);
      }
    }
  }
}

function onWheel(e) {
  e.preventDefault();

  const rect = mapCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  const zoomFactor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  const newScale = Math.max(0.1, Math.min(50, mapScale * zoomFactor));

  const worldBefore = screenToWorld(mx, my);

  mapScale = newScale;

  const worldAfter = screenToWorld(mx, my);
  mapOffsetX += (worldAfter.x - worldBefore.x) * 0;
  mapOffsetY += (worldAfter.z - worldBefore.z) * 0;

  mapOffsetX += (mx - mapWidth / 2) * (1 - zoomFactor);
  mapOffsetY += (my - mapHeight / 2) * (1 - zoomFactor);

  updateZoomInfo();
  heatmapDirty = true;
  renderMap();
}

function updateZoomInfo() {
  const zoomInfo = document.getElementById('mapZoomInfo');
  if (zoomInfo) {
    zoomInfo.textContent = '缩放: ' + mapScale.toFixed(1) + 'x';
  }
}

// ============== 公开 API ==============

function zoomIn() {
  mapScale = Math.min(50, mapScale * 1.3);
  updateZoomInfo();
  heatmapDirty = true;
  renderMap();
}

function zoomOut() {
  mapScale = Math.max(0.1, mapScale / 1.3);
  updateZoomInfo();
  heatmapDirty = true;
  renderMap();
}

function resetView() {
  mapScale = 1.0;
  mapOffsetX = 0;
  mapOffsetY = 0;
  computeBounds();
  updateZoomInfo();
  heatmapDirty = true;
  renderMap();
}

function toggleGrid() {
  showGrid = !showGrid;
  renderMap();
}

function toggleHeatmap() {
  showHeatmap = !showHeatmap;
  heatmapDirty = true;
  renderMap();
}

function setCallbacks(callbacks) {
  if (callbacks.onMapClick) onMapClick = callbacks.onMapClick;
  if (callbacks.onMapHover) onMapHover = callbacks.onMapHover;
}

function getMapState() {
  return {
    scale: mapScale,
    offsetX: mapOffsetX,
    offsetY: mapOffsetY,
    pointCount: mapPoints.length / 3,
    trajectoryLength: mapTrajectory.length,
    showGrid: showGrid,
    showHeatmap: showHeatmap,
    showTrajectory: showTrajectory,
    showPointCloud: showPointCloud,
    showCameraMarkers: showCameraMarkers
  };
}

function clearMap() {
  mapPoints = [];
  mapColors = [];
  mapTrajectory = [];
  mapCameraPoses = [];
  mapScale = 1.0;
  mapOffsetX = 0;
  mapOffsetY = 0;
  heatmapDirty = true;
  renderMap();

  const overlay = document.getElementById('mapOverlay');
  if (overlay) overlay.style.display = 'flex';
}

window.SpatialMap = {
  init: initMap,
  updateData: updateMapData,
  render: renderMap,
  zoomIn: zoomIn,
  zoomOut: zoomOut,
  resetView: resetView,
  toggleGrid: toggleGrid,
  toggleHeatmap: toggleHeatmap,
  setCallbacks: setCallbacks,
  getState: getMapState,
  clear: clearMap
};
