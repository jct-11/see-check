# Camera Flight Controls Redesign

## Problem

当前自由移动模式使用简单 quaternion 旋转 + 固定速度 View-Relative 移动，缺少旋转平滑、移动模式切换、pitch 限制和缓动重置。参考 `3D场景图点云Viewer方案.md` 的 FPS 相机控制逻辑进行改版。

## Design

### Movement System

Two modes, F key toggles:

| Mode | Behavior | Default |
|------|----------|---------|
| View-Relative | WS = view direction, AD = camera right/left, QE = camera up/down | yes |
| Horizontal | WS = horizontal projection of view, AD = horizontal left/right, QE = pure vertical | |

Speed system:
- Base speed 2.0 m/s
- Damping coefficient 25.0
- Per-frame velocity smoothing: `v = targetVelocity + (v - targetVelocity) * exp(-damping * delta)`

Implementation: `moveState` object tracks each axis key state (-1/0/+1 for w/s, a/d, q/e), compute target velocity vector each frame, apply exponential smoothing.

Double-tap acceleration: not included (user confirmed not needed).

### Camera Rotation

Replace raw quaternion multiplication with Euler angle smoothing:
- Left-drag rotation, sensitivity 0.003 rad/pixel
- `targetEuler` (target) + `currentEuler` (displayed)
- Smoothing: `currentEuler += (targetEuler - currentEuler) * (1 - pow(0.001, delta))`
- Pitch clamped to ±89 degrees
- Right-drag pan preserved

### Other Controls

| Input | Action |
|-------|--------|
| Scroll wheel | Move along view direction, `-deltaY * 0.005` |
| R key | Reset camera to scene center, 0.8s cubic ease-in-out animation |
| F key | Toggle Horizontal / View-Relative movement mode |

## Implementation Notes

- Replace `flightQuat` quaternion tracking with `currentEuler`/`targetEuler` Euler angle pair
- Replace `FLIGHT_MOVE_SPEED` constant with speed + damping system
- Replace `FLIGHT_ZOOM_SPEED` (0.15) with 0.005 coefficient
- Add pitch clamping in rotation update
- Add reset animation (cubic ease-in-out tween)
- No changes to mouse event handlers (same left/right button behavior)
