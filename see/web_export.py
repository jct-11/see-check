"""Web visualization data export for LingBot-Map.

Exports the complete inference results as a self-contained web-viewable directory:
- cameras.json (per-frame camera extrinsics/intrinsics)
- metadata.json (scene info, defaults)
- frame_*.jpg (source images for frustum overlays)
- merged_confidence.bin (per-point confidence for merged cloud)
- viewer.html (copied static Three.js viewer)
"""

import os
import json
import shutil
import numpy as np


def export_web_data(predictions, images_cpu, output_dir, conf_threshold=0.5):
    """Export all data needed for web-based 3D visualization.

    Args:
        predictions: dict with keys 'extrinsic', 'intrinsic', 'world_points',
                    'world_points_conf', etc. (already on CPU, numpy or tensor)
        images_cpu: CPU tensor of shape (S, 3, H, W) in float [0,1] range
        output_dir: directory to write output files
        conf_threshold: confidence threshold used for display defaults
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert tensors to numpy as needed
    ext = _to_numpy(predictions.get("extrinsic"))    # (S, 3, 4) c2w
    intr = _to_numpy(predictions.get("intrinsic"))    # (S, 3, 3)
    imgs = _to_numpy(images_cpu)                      # (S, 3, H, W)
    world_points = _to_numpy(predictions.get("world_points"))   # (S, H, W, 3)
    world_conf = _to_numpy(predictions.get("world_points_conf")) # (S, H, W)

    if ext is None or intr is None or imgs is None:
        print("[web_export] Missing essential data (extrinsic/intrinsic/images), skipping")
        return

    S = ext.shape[0]
    H, W = imgs.shape[-2], imgs.shape[-1]
    print(f"[web_export] Exporting {S} frames ({W}x{H}) to {output_dir}")

    # ── 1. cameras.json ─────────────────────────────────────────────────
    # Export w2c R/t (what viser uses in cam_dict via closed_form_inverse_se3)
    cam_list = []
    for i in range(S):
        ext_4x4 = np.eye(4, dtype=np.float64)
        ext_4x4[:3, :3] = ext[i, :3, :3]
        ext_4x4[:3, 3] = ext[i, :3, 3]
        w2c = np.linalg.inv(ext_4x4)
        R_w2c = w2c[:3, :3].tolist()
        t_w2c = w2c[:3, 3].tolist()
        cam_list.append({
            "focal": [float(intr[i, 0, 0]), float(intr[i, 1, 1])],
            "pp": [float(intr[i, 0, 2]), float(intr[i, 1, 2])],
            "R_w2c": R_w2c,
            "t_w2c": t_w2c,
            "image_w": W,
            "image_h": H,
        })
    with open(os.path.join(output_dir, "cameras.json"), "w") as f:
        json.dump(cam_list, f, indent=2)
    print(f"[web_export] ✓ cameras.json ({S} frames)")

    # ── 2. metadata.json ────────────────────────────────────────────────
    # viser computes scene center from t_w2c (not t_c2w)
    t_w2c_list = []
    for i in range(S):
        ext_4x4 = np.eye(4, dtype=np.float64)
        ext_4x4[:3, :3] = ext[i, :3, :3]
        ext_4x4[:3, 3] = ext[i, :3, 3]
        w2c = np.linalg.inv(ext_4x4)
        t_w2c_list.append(w2c[:3, 3])
    cam_positions = np.array(t_w2c_list)
    scene_center = cam_positions.mean(axis=0).tolist()
    scene_scale = float(np.linalg.norm(np.ptp(cam_positions, axis=0))) if S > 1 else 1.0

    metadata = {
        "num_frames": S,
        "image_width": W,
        "image_height": H,
        "scene_center": scene_center,
        "scene_scale": max(scene_scale, 0.1),
        "default_conf_threshold": conf_threshold,
        "default_downsample": 2,
        "has_world_points": world_points is not None,
        "has_confidence": world_conf is not None,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[web_export] ✓ metadata.json")

    # ── 3. Frame images (JPEG) ──────────────────────────────────────────
    # Use OpenCV if available, otherwise PIL
    try:
        import cv2
        _has_cv2 = True
    except ImportError:
        _has_cv2 = False

    for i in range(S):
        img = imgs[i]  # (3, H, W)
        img = (img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        jpg_path = os.path.join(output_dir, f"frame_{i:06d}.jpg")
        if _has_cv2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(jpg_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        else:
            from PIL import Image
            Image.fromarray(img).save(jpg_path, quality=85)
        if i % 50 == 0:
            print(f"[web_export]   frame images {i}/{S}")

    print(f"[web_export] ✓ {S} frame images saved as JPEG")

    # ── 4. Merged confidence binary ─────────────────────────────────────
    # If inference_web.py already saved correctly-ordered confidence, skip
    merged_conf_path = os.path.join(output_dir, "merged_confidence.bin")
    if os.path.exists(merged_conf_path):
        print("[web_export] ✓ merged_confidence.bin already exists, skipping")
    else:
        conf_source = world_conf
        if conf_source is None:
            conf_source = _to_numpy(predictions.get("depth_conf"))
            if conf_source is not None:
                print("[web_export] Using depth_conf for confidence data")

        if world_points is not None or predictions.get("depth") is not None:
            all_conf = []
            for i in range(S):
                if conf_source is not None:
                    c = conf_source[i].reshape(-1)
                else:
                    c = np.ones(int(H * W), dtype=np.float32)
                all_conf.append(c)
            merged_conf = np.concatenate(all_conf).astype(np.float32)
            merged_conf.tofile(merged_conf_path)
            print(f"[web_export] ✓ merged_confidence.bin ({len(merged_conf)} values)")
        else:
            print("[web_export] ⚠ No point data or confidence available, skipping merged_confidence.bin")

    # ── 5. Copy static viewer HTML ──────────────────────────────────────
    viewer_src = os.path.join(os.path.dirname(__file__), "static_viewer.html")
    viewer_dst = os.path.join(output_dir, "viewer.html")
    if os.path.exists(viewer_src):
        shutil.copy2(viewer_src, viewer_dst)
        print(f"[web_export] ✓ viewer.html copied to output dir")
    else:
        print(f"[web_export] ⚠ static_viewer.html not found at {viewer_src}")

    print(f"[web_export] Done! View with: cd {output_dir} && python3 -m http.server 8080")


def _to_numpy(t):
    """Convert torch tensor or numpy array to numpy."""
    if t is None:
        return None
    if hasattr(t, "cpu"):
        return t.cpu().numpy()
    if hasattr(t, "numpy"):
        return t.numpy()
    return np.asarray(t)
