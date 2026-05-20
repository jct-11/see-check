# Camera Flight Controls Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace simple quaternion flight controls with smoothed Euler-angle rotation, dual movement modes, pitch clamping, and eased camera reset.

**Architecture:** Replace `flightQuat` with `currentEuler`/`targetEuler` Euler pair for smoothing. Replace `flightKeys` boolean map with `moveState` 3-axis velocity system with damping. All changes in `spatial.js` (~60 lines modified).

**Tech Stack:** Three.js (Vector3, Euler, Quaternion, MathUtils), vanilla JS

---

### Task 1: Replace state variables and constants

**Files:** Modify `src/web/assets/spatial.js:764-766,927-929`

- [ ] **Step 1: Replace flight state variables**

At line 764-766, replace:
```javascript
let flightQuat = null;
let flightKeys = {};
let flightLeftDown = false, flightRightDown = false;
```
With:
```javascript
// Camera rotation — Euler smoothing
let currentEuler = null;
let targetEuler = null;
// Movement state — 3-axis velocity with damping
let moveState = { x: 0, y: 0, z: 0 };  // -1/0/+1 per axis
let currentVelocity = new THREE.Vector3();
// Mouse state
let flightLeftDown = false, flightRightDown = false;
let flightLastMouseX = 0, flightLastMouseY = 0;
// Movement mode
let flightModeHorizontal = false;  // false = View-Relative (default), true = Horizontal
let isResetting = false;
let resetStartTime = 0;
let resetStartPos = new THREE.Vector3();
let resetStartTarget = new THREE.Vector3();
const RESET_DURATION = 0.8;
```

- [ ] **Step 2: Replace constants**

At line 927-929, replace:
```javascript
const FLIGHT_MOVE_SPEED = 0.08;
const FLIGHT_ZOOM_SPEED = 0.15;
const FLIGHT_SENSITIVITY = 0.003;
```
With:
```javascript
const FLIGHT_BASE_SPEED = 2.0;
const FLIGHT_DAMPING = 25.0;
const FLIGHT_SENSITIVITY = 0.003;
const FLIGHT_SCROLL_SPEED = 0.005;
```

- [ ] **Step 3: Commit**

```bash
git add src/web/assets/spatial.js
git commit -m "feat: replace flight state variables and constants for new camera controls"
```

---

### Task 2: Update init3DScene to init new variables

**Files:** Modify `src/web/assets/spatial.js:872-886`

- [ ] **Step 1: Replace flightQuat init**

At line 872, replace:
```javascript
  flightQuat = new THREE.Quaternion();
  flightKeys = {};
```
With:
```javascript
  currentEuler = new THREE.Euler(0, 0, 0, 'YXZ');
  targetEuler = new THREE.Euler(0, 0, 0, 'YXZ');
  moveState = { x: 0, y: 0, z: 0 };
  currentVelocity.set(0, 0, 0);
```

- [ ] **Step 2: Add F key handler for mode toggle**

After the keydown/keyup listeners (after line 886), add:
```javascript
  window.addEventListener('keydown', function(e) {
    if (e.key.toLowerCase() === 'f' && !cameraFollowEnabled && !e.repeat) {
      flightModeHorizontal = !flightModeHorizontal;
      console.log('[Spatial] Flight mode:', flightModeHorizontal ? 'Horizontal' : 'View-Relative');
    }
    if (e.key.toLowerCase() === 'r' && !cameraFollowEnabled) {
      const cx = sceneCenter[0], cy = sceneCenter[1], cz = sceneCenter[2];
      const dist = sceneScale * 0.8;
      resetStartPos.copy(camera3d.position);
      resetStartTarget.set(cx + dist * 0.5, cy + dist * 0.5, cz + dist);
      isResetting = true;
      resetStartTime = performance.now() / 1000;
    }
  });
```

Note: F and R only work when NOT in follow mode.

- [ ] **Step 3: Commit**

```bash
git add src/web/assets/spatial.js
git commit -m "feat: init Euler state, add F/R key handlers for camera controls"
```

---

### Task 3: Rewrite rotation handler (onFlightMouseMove)

**Files:** Modify `src/web/assets/spatial.js:944-965`

- [ ] **Step 1: Replace the rotation logic**

Replace lines 944-953 (the flightLeftDown block) with:

```javascript
function onFlightMouseMove(e) {
  const dx = e.clientX - flightLastMouseX;
  const dy = e.clientY - flightLastMouseY;
  if (flightLeftDown) {
    targetEuler.y -= dx * FLIGHT_SENSITIVITY;
    targetEuler.x -= dy * FLIGHT_SENSITIVITY;
    targetEuler.x = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, targetEuler.x));  // clamp ±89°
  }
  if (flightRightDown) {
    // Right-drag pan (unchanged)
    if (!camera3d) return;
    const dir = camera3d.getWorldDirection(new THREE.Vector3());
    const right = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(0, 1, 0)).normalize();
    const up = new THREE.Vector3().crossVectors(right, dir).normalize();
    const scale = FLIGHT_BASE_SPEED * 0.005;
    camera3d.position.addScaledVector(right, -dx * scale);
    camera3d.position.addScaledVector(up, dy * scale);
  }
  flightLastMouseX = e.clientX;
  flightLastMouseY = e.clientY;
}
```

- [ ] **Step 2: Update scroll handler constant**

In `onFlightWheel` (~line 969), replace `FLIGHT_ZOOM_SPEED` with `FLIGHT_SCROLL_SPEED`:
```javascript
function onFlightWheel(e) {
  e.preventDefault();
  if (!camera3d || cameraFollowEnabled) return;
  const dir = camera3d.getWorldDirection(new THREE.Vector3());
  camera3d.position.addScaledVector(dir, -e.deltaY * FLIGHT_SCROLL_SPEED);
}
```

- [ ] **Step 3: Commit**

```bash
git add src/web/assets/spatial.js
git commit -m "feat: Euler rotation with pitch clamp, update scroll speed constant"
```

---

### Task 4: Rewrite keyboard handler and movement system

**Files:** Modify `src/web/assets/spatial.js:976-999`

- [ ] **Step 1: Replace onFlightKeyDown/Up**

Replace lines 976-991:
```javascript
function onFlightKeyDown(e) {
  flightKeys[e.key.toLowerCase()] = true;
  if (['w','a','s','d','q','e'].includes(e.key.toLowerCase())) {
    e.preventDefault();
  }
}

function onFlightKeyUp(e) {
  flightKeys[e.key.toLowerCase()] = false;
}
```
With:
```javascript
function onFlightKeyDown(e) {
  const key = e.key.toLowerCase();
  if (['w','a','s','d','q','e','f','r'].includes(key)) {
    e.preventDefault();
  }
  switch (key) {
    case 'w': moveState.z = 1; break;
    case 's': moveState.z = -1; break;
    case 'a': moveState.x = -1; break;
    case 'd': moveState.x = 1; break;
    case 'q': moveState.y = -1; break;
    case 'e': moveState.y = 1; break;
  }
}

function onFlightKeyUp(e) {
  const key = e.key.toLowerCase();
  switch (key) {
    case 'w': case 's': moveState.z = 0; break;
    case 'a': case 'd': moveState.x = 0; break;
    case 'q': case 'e': moveState.y = 0; break;
  }
}
```

Note: F and R are handled in the dedicated keydown listener added in Task 2.

- [ ] **Step 2: Replace updateFlightMovement**

Replace lines 993-999 (`updateFlightMovement` function):
```javascript
function updateFlightMovement(delta) {
  if (!camera3d || cameraFollowEnabled) return;

  const targetVel = new THREE.Vector3(moveState.x, moveState.y, moveState.z);
  if (targetVel.length() > 1) targetVel.normalize();
  targetVel.multiplyScalar(FLIGHT_BASE_SPEED);

  const damping = Math.min(1.0, FLIGHT_DAMPING * delta);
  currentVelocity.lerp(targetVel, damping);

  if (flightModeHorizontal) {
    // Horizontal mode: WS along horizontal projection, AD horizontal, QE pure vertical
    const dir = camera3d.getWorldDirection(new THREE.Vector3());
    dir.y = 0;
    if (dir.length() < 0.001) dir.set(0, 0, 1);
    dir.normalize();
    const right = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(0, 1, 0)).normalize();
    camera3d.position.addScaledVector(dir, currentVelocity.z * delta);
    camera3d.position.addScaledVector(right, currentVelocity.x * delta);
    camera3d.position.y += currentVelocity.y * delta;
  } else {
    // View-Relative mode: all directions based on camera orientation
    const dir = camera3d.getWorldDirection(new THREE.Vector3());
    const right = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(0, 1, 0)).normalize();
    const up = new THREE.Vector3().crossVectors(right, dir).normalize();
    camera3d.position.addScaledVector(dir, currentVelocity.z * delta);
    camera3d.position.addScaledVector(right, currentVelocity.x * delta);
    camera3d.position.addScaledVector(up, currentVelocity.y * delta);
  }
}

function updateCameraRotation(delta) {
  if (cameraFollowEnabled) return;
  const smooth = 1.0 - Math.pow(0.001, delta);
  currentEuler.x += (targetEuler.x - currentEuler.x) * smooth;
  currentEuler.y += (targetEuler.y - currentEuler.y) * smooth;
  currentEuler.z += (targetEuler.z - currentEuler.z) * smooth;
  camera3d.quaternion.setFromEuler(currentEuler);
}

function updateReset(delta) {
  if (!isResetting) return;
  const elapsed = (performance.now() / 1000) - resetStartTime;
  if (elapsed >= RESET_DURATION) {
    camera3d.position.copy(resetStartTarget);
    targetEuler.set(0, 0, 0);
    currentEuler.set(0, 0, 0);
    isResetting = false;
    return;
  }
  const t = elapsed / RESET_DURATION;
  // Cubic ease-in-out
  const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  camera3d.position.lerpVectors(resetStartPos, resetStartTarget, ease);
}
```

- [ ] **Step 3: Commit**

```bash
git add src/web/assets/spatial.js
git commit -m "feat: velocity+damping movement, dual mode, rotation smoothing, R-key reset"
```

---

### Task 5: Update animate loop

**Files:** Modify `src/web/assets/spatial.js:1004-1040`

- [ ] **Step 1: Replace the animate function body**

Replace the animate function (lines 1004-1040) with:

```javascript
function animate() {
  animationId = requestAnimationFrame(animate);

  const now = performance.now();
  const maxFps = cameraFollowEnabled ? 15 : 30;
  if (now - lastRenderTime < (1000 / maxFps)) return;

  // Delta time for damping
  const delta = Math.min((now - lastRenderTime) / 1000, 0.1);

  // Camera follow mode
  if (cameraFollowEnabled && followSmoothedPos && followLookTarget) {
    camera3d.position.copy(followSmoothedPos);
    camera3d.up.set(0, -1, 0);
    camera3d.lookAt(followLookTarget);
    // Sync Euler from camera so rotation is continuous on exit
    currentEuler.setFromQuaternion(camera3d.quaternion, 'YXZ');
    targetEuler.copy(currentEuler);
  } else if (!cameraFollowEnabled && camera3d) {
    // Free-flight: update reset, rotation smoothing, movement
    updateReset(delta);
    updateCameraRotation(delta);
    updateFlightMovement(delta);
  }

  if (renderer && scene && camera3d) {
    // ... FPS stats unchanged ...
    frameCount++;
    const now2 = performance.now();
    if (now2 - frameTime >= 1000) {
      visualizerStats.fps = Math.round(frameCount / ((now2 - frameTime) / 1000));
      frameCount = 0;
      frameTime = now2;
    }
    renderer.render(scene, camera3d);
    lastRenderTime = now;
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/web/assets/spatial.js
git commit -m "feat: update animate loop with new movement, rotation, and reset systems"
```

---

### Task 6: Update disableCameraFollow to sync Euler state

**Files:** Modify `src/web/assets/spatial.js:1262-1277`

- [ ] **Step 1: Replace flightQuat copy with Euler sync**

In `disableCameraFollow`, replace:
```javascript
  if (camera3d) {
    flightQuat.copy(camera3d.quaternion);
    camera3d.up.set(0, 1, 0);
  }
```
With:
```javascript
  if (camera3d) {
    currentEuler.setFromQuaternion(camera3d.quaternion, 'YXZ');
    targetEuler.copy(currentEuler);
    camera3d.up.set(0, 1, 0);
  }
```

- [ ] **Step 2: Commit**

```bash
git add src/web/assets/spatial.js
git commit -m "fix: sync Euler state on camera follow exit instead of flightQuat"
```

---

### Task 7: Update remaining flightQuat references

**Files:** Modify `src/web/assets/spatial.js` — fitCameraToScene, reset3DCamera, setViewDirection, forceStopProcessing

- [ ] **Step 1: Replace all remaining flightQuat assignments**

For each occurrence of `flightQuat =` or `flightQuat.`, replace with Euler equivalents:

`fitCameraToScene`:
```javascript
// Old:
flightQuat.setFromUnitVectors(new THREE.Vector3(0, 0, -1), lookDir);
// New:
camera3d.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, -1), lookDir);
currentEuler.setFromQuaternion(camera3d.quaternion, 'YXZ');
targetEuler.copy(currentEuler);
```

`reset3DCamera`:
```javascript
// Old:
flightQuat = new THREE.Quaternion();
// New:
currentEuler.set(0, 0, 0);
targetEuler.set(0, 0, 0);
```

`setViewDirection`:
```javascript
// Old:
flightQuat.setFromUnitVectors(new THREE.Vector3(0, 0, -1), lookDir);
// New:
camera3d.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, -1), lookDir);
currentEuler.setFromQuaternion(camera3d.quaternion, 'YXZ');
targetEuler.copy(currentEuler);
```

`forceStopProcessing`:
```javascript
// Old:
if (camera3d) { flightQuat.copy(camera3d.quaternion); }
// New:
if (camera3d) { currentEuler.setFromQuaternion(camera3d.quaternion, 'YXZ'); targetEuler.copy(currentEuler); }
```

`enableCameraFollow`:
```javascript
// Old:
if (camera3d) { flightQuat.copy(camera3d.quaternion); }
// New:
if (camera3d) { currentEuler.setFromQuaternion(camera3d.quaternion, 'YXZ'); targetEuler.copy(currentEuler); }
```

- [ ] **Step 2: Commit**

```bash
git add src/web/assets/spatial.js
git commit -m "fix: update all remaining flightQuat references to Euler state"
```
