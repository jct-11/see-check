#!/usr/bin/env python3
"""Scale calibrator: compute metric scale factor s from DAv2 Small depth,
then scale params_with_idx.npz means3D to real-world meters.

Usage: python scripts/scale_calibrate.py <batch_id> <scene_name>

batch_id:  used to locate conf/depth/frames dirs under DATA_DIR
scene_name: used to locate params_with_idx.npz under DGSG_EXP_DIR
"""

import numpy as np
import os
import sys
import time
import json
import traceback
import gc
import pathlib
import cv2
from PIL import Image

# ── Paths ──
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
DAV2_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DATA_DIR = "/home/sscy/lingbot-map/stmem-main/data"
DGSG_EXP_DIR = "/home/liangjiahua/dgsg-orin/experiments/mydata"


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[scale {ts}] {msg}", flush=True)


def find_best_frames(batch_dir, k=None):
    """Find top-K frames with highest mean depth_conf.
    K is auto-computed from total frame count if not specified:
      N < 50  → K=3
      N < 200 → K=5
      N < 500 → K=7
      N >=500 → K=10
    """
    conf_dir = batch_dir / "conf"
    conf_files = sorted(conf_dir.glob("frame_*.npy"))
    if not conf_files:
        log("WARNING: no conf files found, using frame 0 only")
        return [(0, 0.0)]

    if k is None:
        n = len(conf_files)
        if n < 50:
            k = 3
        elif n < 200:
            k = 5
        elif n < 500:
            k = 7
        else:
            k = 10

    scored = []
    for cf in conf_files:
        try:
            c = np.load(str(cf))
            m = float(c.mean())
            idx = int(cf.stem.split("_")[1])
            scored.append((idx, m))
        except Exception:
            continue

    scored.sort(key=lambda x: -x[1])
    top = scored[:k]
    log(f"Top-{k} frames (from {len(conf_files)} total): {[(idx, f'{conf:.3f}') for idx, conf in top]}")
    return top


def compute_scale(depth_metric, depth_pred, rgb=None):
    """Hybrid scale computation: log-space trimmed mean → percentile fallback."""
    # Align sizes: resize metric depth to match pred depth shape
    import cv2
    dh, dw = depth_pred.shape[:2]
    if depth_metric.shape[:2] != (dh, dw):
        depth_metric = cv2.resize(depth_metric, (dw, dh), interpolation=cv2.INTER_LINEAR)

    # Valid mask
    valid = (depth_metric > 0.1) & (depth_metric < 19.0) & (depth_pred > 0.1) & (depth_pred < 50.0)

    if valid.sum() < 100:
        log("ERROR: too few valid pixels, using s=1.0")
        return 1.0, "fallback_empty", 0.0

    ratios = depth_metric[valid] / (depth_pred[valid] + 1e-8)
    ratios = ratios[ratios > 0.001]  # filter extreme outliers

    # ── Method A: log-space trimmed mean ──
    log_ratios = np.log(np.clip(ratios, 1e-4, 1e4))
    trim_pct = 0.2
    n_trim = int(len(log_ratios) * trim_pct)
    if n_trim > 0:
        sorted_lr = np.sort(log_ratios)
        trimmed = sorted_lr[n_trim:-n_trim]
    else:
        trimmed = log_ratios

    if len(trimmed) < 50:
        log("trimmed sample too small, using s=1.0")
        return 1.0, "fallback_trim", 0.0

    s_log = float(np.mean(trimmed))
    s_std = float(np.std(trimmed))
    s = float(np.exp(s_log))

    # Check reliability: if std > threshold, fall back to percentile method
    log_ratio_std = s_std
    if log_ratio_std > 0.5:  # high variance → fallback
        log(f"High log-ratio std ({log_ratio_std:.3f}), falling back to percentile method")
        percentiles = [10, 25, 50, 75, 90]
        s_candidates = []
        for p in percentiles:
            p_metric = np.percentile(depth_metric[valid], p)
            p_pred = np.percentile(depth_pred[valid], p)
            if p_pred > 0.01:
                s_candidates.append(p_metric / p_pred)
        if s_candidates:
            s = float(np.median(s_candidates))
            method = "percentile"
        else:
            s = 1.0
            method = "fallback_zero"
    else:
        method = "log_trimmed_mean"

    confidence = max(0.0, min(1.0, 1.0 - log_ratio_std / 1.0))

    log(f"s = {s:.6f} (method={method}, log_std={log_ratio_std:.3f}, conf={confidence:.2f})")
    return s, method, confidence


def main():
    t0 = time.time()

    if len(sys.argv) < 3:
        log("ERROR: missing arguments")
        log("Usage: python scale_calibrate.py <batch_id> <scene_name> [--k N]")
        sys.exit(1)

    batch_id = sys.argv[1]
    scene_name = sys.argv[2]
    k = None  # auto: derived from total frame count
    for i, arg in enumerate(sys.argv):
        if arg == "--k" and i + 1 < len(sys.argv):
            k = int(sys.argv[i + 1])

    batch_dir = pathlib.Path(DATA_DIR) / batch_id
    npz_path = pathlib.Path(DGSG_EXP_DIR) / scene_name / "params_with_idx.npz"

    if not npz_path.exists():
        log(f"ERROR: npz not found: {npz_path}")
        sys.exit(1)

    k_str = "auto" if k is None else str(k)
    log(f"batch_id={batch_id} scene={scene_name} k={k_str}")
    log(f"npz_path={npz_path}")

    # ── 1. Find top-K frames by confidence ──
    top_frames = find_best_frames(batch_dir, k)

    # ── 2. Load DAv2 model once, run on all K frames ──
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    import torch
    import cv2
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Loading DAv2 model (device={device.type})...")
    model = AutoModelForDepthEstimation.from_pretrained(DAV2_MODEL).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(DAV2_MODEL)

    s_values = []
    frame_results = []

    for frame_idx, conf_mean in top_frames:
        log(f"Frame {frame_idx} (conf={conf_mean:.3f})...")

        # Load RGB and model depth
        rgb_path = batch_dir / "frames" / f"frame_{frame_idx:06d}.jpg"
        if not rgb_path.exists():
            log(f"  SKIP: no RGB for frame {frame_idx}")
            continue
        bgr = cv2.imread(str(rgb_path))
        if bgr is None:
            log(f"  SKIP: failed to read {rgb_path}")
            continue
        rgb_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        depth_path = batch_dir / "depth" / f"frame_{frame_idx:06d}.png"
        if not depth_path.exists():
            log(f"  SKIP: no depth for frame {frame_idx}")
            continue
        depth_pred = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        # DAv2 inference
        t_inf = time.time()
        inputs = processor(images=rgb_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            d_raw = outputs.predicted_depth
        d_raw = torch.nn.functional.interpolate(
            d_raw.unsqueeze(1), size=rgb_pil.size[::-1],
            mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy().astype(np.float32)
        log(f"  DAv2: {((time.time()-t_inf)*1000):.0f}ms")

        s_i, method_i, conf_i = compute_scale(d_raw, depth_pred)
        if 0.01 < s_i < 100 and np.isfinite(s_i):
            s_values.append(s_i)
            frame_results.append({
                "frame": int(frame_idx), "conf_mean": float(conf_mean),
                "s": float(s_i), "method": method_i, "s_confidence": float(conf_i),
            })
        else:
            log(f"  SKIP: unreasonable s={s_i}")

    # ── Clean up DAv2 ──
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log("DAv2 model unloaded")

    if len(s_values) == 0:
        log("ERROR: no valid s from any frame, using s=1.0")
        s = 1.0
        method = "fallback_none"
        conf = 0.0
    elif len(s_values) == 1:
        s = s_values[0]
        method = "single_frame"
        conf = frame_results[0]["s_confidence"]
    else:
        s = float(np.median(s_values))
        method = f"median_of_{len(s_values)}"
        conf = float(1.0 - np.std(s_values) / (np.mean(s_values) + 1e-8))
        conf = max(0.0, min(1.0, conf))

    log(f"Final s = {s:.6f} (method={method}, conf={conf:.2f}, from {len(s_values)} frames: {[f'{v:.3f}' for v in s_values]})")

    # ── 3. Scale means3D and save ──
    log("Loading npz for scaling...")
    data = np.load(npz_path)
    means3D = data["means3D"].astype(np.float64)
    means3D *= s
    means3D = means3D.astype(np.float32)

    log(f"Scaling means3D by s={s:.6f}, saving...")
    np.savez_compressed(
        npz_path,
        means3D=means3D,
        rgb_colors=data["rgb_colors"],
        object_idx=data["object_idx"]
    )

    # ── 4. Write scale metadata ──
    meta = {
        "scale_factor": float(s),
        "method": method,
        "confidence": float(conf),
        "num_frames": len(s_values),
        "frames": frame_results,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    meta_path = npz_path.parent / "scale_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log(f"Done. s={s:.6f}, method={method}, conf={conf:.2f}, elapsed={meta['elapsed_sec']}s")


if __name__ == "__main__":
    main()
