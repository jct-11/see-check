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

# ── Paths ──
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
DAV2_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DATA_DIR = "/home/sscy/lingbot-map/stmem-main/data"
DGSG_EXP_DIR = "/home/liangjiahua/dgsg-orin/experiments/mydata"


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[scale {ts}] {msg}", flush=True)


def find_best_frame(batch_dir):
    """Find frame with highest mean depth_conf."""
    conf_dir = batch_dir / "conf"
    conf_files = sorted(conf_dir.glob("frame_*.npy"))
    if not conf_files:
        log("WARNING: no conf files found, using frame 0")
        return 0, 0.0

    best_idx = 0
    best_mean = -1.0
    for cf in conf_files:
        try:
            c = np.load(str(cf))
            m = float(c.mean())
            if m > best_mean:
                best_mean = m
                best_idx = int(cf.stem.split("_")[1])
        except Exception:
            continue

    log(f"Best frame: {best_idx} (conf_mean={best_mean:.4f})")
    return best_idx, best_mean


def load_model_depths(batch_dir, frame_idx):
    """Load DAv2 metric depth and lingbot-map predicted depth for a frame."""
    import cv2
    from PIL import Image

    # ── Load RGB image ──
    rgb_path = batch_dir / "frames" / f"frame_{frame_idx:06d}.jpg"
    if not rgb_path.exists():
        rgb_path = next(batch_dir.glob("frames/frame_*.jpg"), None)
        if rgb_path is None:
            raise FileNotFoundError(f"No frame image found in {batch_dir}/frames/")
    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        raise ValueError(f"Failed to read {rgb_path}")
    h, w = rgb.shape[:2]
    log(f"Using frame: {rgb_path.name} ({w}x{h})")

    # ── DAv2 Small metric depth (transformers pipeline) ──
    log("Running DAv2 Small metric depth inference...")
    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    t0 = time.time()
    pipe = pipeline("depth-estimation", model=DAV2_MODEL, device=device)
    # pipeline expects PIL Image, not cv2 numpy
    rgb_pil = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    result = pipe(rgb_pil)
    depth_metric = np.array(result["depth"], dtype=np.float32)  # HxW meters
    model_ms = (time.time() - t0) * 1000
    log(f"DAv2 inference: {model_ms:.0f}ms (device={'cuda' if device==0 else 'cpu'})")

    # ── Clean up ──
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log("DAv2 model unloaded")

    # ── Load lingbot-map predicted depth ──
    depth_path = batch_dir / "depth" / f"frame_{frame_idx:06d}.png"
    if not depth_path.exists():
        depth_list = sorted(batch_dir.glob("depth/frame_*.png"))
        if depth_list:
            depth_path = depth_list[0]
        else:
            raise FileNotFoundError(f"No depth image found for frame {frame_idx}")
    depth_pred = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    log(f"Loaded depth: {depth_pred.shape[1]}x{depth_pred.shape[0]}")

    return depth_metric, depth_pred, rgb


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
        log("Usage: python scale_calibrate.py <batch_id> <scene_name>")
        sys.exit(1)

    batch_id = sys.argv[1]
    scene_name = sys.argv[2]
    batch_dir = pathlib.Path(DATA_DIR) / batch_id
    npz_path = pathlib.Path(DGSG_EXP_DIR) / scene_name / "params_with_idx.npz"

    if not npz_path.exists():
        log(f"ERROR: npz not found: {npz_path}")
        sys.exit(1)

    log(f"batch_id={batch_id} scene={scene_name}")
    log(f"npz_path={npz_path}")

    # ── 1. Find best frame by confidence ──
    best_frame, best_conf = find_best_frame(batch_dir)

    # ── 2. Compute scale factor s ──
    try:
        depth_metric, depth_pred, rgb = load_model_depths(batch_dir, best_frame)
    except FileNotFoundError as e:
        log(f"ERROR: missing data for scale calibration: {e}")
        log("Pipeline continues with s=1.0 (unscaled)")
        return

    s, method, conf = compute_scale(depth_metric, depth_pred)

    if s < 0.01 or s > 100 or not np.isfinite(s):
        log(f"WARNING: unreasonable s={s}, clamping to 1.0")
        s = 1.0

    # ── 3. Scale means3D and save ──
    log(f"Loading npz for scaling...")
    data = np.load(npz_path)
    means3D = data["means3D"].astype(np.float64)
    means3D *= s
    means3D = means3D.astype(np.float32)

    # Overwrite original npz
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
        "best_frame": int(best_frame),
        "best_frame_conf_mean": float(best_conf),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    meta_path = npz_path.parent / "scale_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log(f"Done. s={s:.6f}, method={method}, conf={conf:.2f}, elapsed={meta['elapsed_sec']}s")


if __name__ == "__main__":
    main()
