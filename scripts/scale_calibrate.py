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
DAV2_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
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
            k = 5
        elif n < 200:
            k = 10
        elif n < 500:
            k = 15
        else:
            k = 20

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

    # Fallback: if batch data missing (overwritten by next session), use latest symlink
    if not batch_dir.exists() or not (batch_dir / "conf").exists():
        latest_link = pathlib.Path(DATA_DIR) / "latest"
        if latest_link.is_symlink():
            fallback_dir = latest_link.resolve()
            log(f"batch {batch_id} data missing, fallback to latest: {fallback_dir.name}")
            batch_dir = fallback_dir
        else:
            log(f"ERROR: batch {batch_id} not found and no latest symlink")
            sys.exit(1)

    exp_dir = pathlib.Path(DGSG_EXP_DIR) / scene_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    k_str = "auto" if k is None else str(k)
    log(f"batch_id={batch_id} scene={scene_name} k={k_str}")
    log(f"batch_dir={batch_dir}")

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

    # Helper: run DAv2 on one frame, return metric depth
    def dav2_infer(frame_idx):
        rgb_path = batch_dir / "frames" / f"frame_{frame_idx:06d}.jpg"
        if not rgb_path.exists():
            return None
        bgr = cv2.imread(str(rgb_path))
        if bgr is None:
            return None
        rgb_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        t_inf = time.time()
        inputs = processor(images=rgb_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            d_raw = model(**inputs).predicted_depth
        d_raw = torch.nn.functional.interpolate(
            d_raw.unsqueeze(1), size=rgb_pil.size[::-1],
            mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy().astype(np.float32)
        log(f"    DAv2 frame {frame_idx}: {((time.time()-t_inf)*1000):.0f}ms")
        return d_raw

    # Helper: compute s for a frame (with caching)
    _s_cache = {}
    def get_s(frame_idx):
        if frame_idx in _s_cache:
            return _s_cache[frame_idx]
        d_metric = dav2_infer(frame_idx)
        if d_metric is None:
            return None
        depth_path = batch_dir / "depth" / f"frame_{frame_idx:06d}.png"
        if not depth_path.exists():
            return None
        d_pred = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        s_i, _, _ = compute_scale(d_metric, d_pred)
        if 0.01 < s_i < 100 and np.isfinite(s_i):
            _s_cache[frame_idx] = s_i
            return s_i
        return None

    TOLERANCE = 0.3
    s_values = []
    frame_results = []
    discarded = 0

    for frame_idx, conf_mean in top_frames:
        log(f"Frame {frame_idx} (conf={conf_mean:.3f})...")

        s_f = get_s(frame_idx)
        if s_f is None:
            log(f"  SKIP: cannot compute s for frame {frame_idx}")
            discarded += 1
            continue

        s_prev = get_s(frame_idx - 1)
        s_next = get_s(frame_idx + 1)

        if s_prev is None or s_next is None:
            log(f"  SKIP: missing neighbor ({'prev' if s_prev is None else 'next'})")
            discarded += 1
            continue

        dp = abs(s_f - s_prev) / s_f
        dn = abs(s_f - s_next) / s_f
        if dp > TOLERANCE or dn > TOLERANCE:
            log(f"  DISCARD: s_f={s_f:.3f} s_prev={s_prev:.3f} s_next={s_next:.3f} (dp={dp:.2f} dn={dn:.2f})")
            discarded += 1
            continue

        log(f"  ACCEPT: s={s_f:.3f} (prev={s_prev:.3f} next={s_next:.3f})")
        s_values.append(s_f)
        frame_results.append({
            "frame": int(frame_idx), "conf_mean": float(conf_mean),
            "s": float(s_f), "neighbors_ok": True,
        })

    log(f"Temporal filter: {len(s_values)} accepted, {discarded} discarded (cache={len(_s_cache)} inferences)")

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

    # ── 3. Save scale result JSON (npz scaling deferred to batch_service after DGSG) ──
    meta = {
        "scale_factor": float(s),
        "method": method,
        "confidence": float(conf),
        "num_frames": len(s_values),
        "frames": frame_results,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    result_path = exp_dir / "scale_result.json"
    with open(result_path, "w") as f:
        json.dump(meta, f, indent=2)

    log(f"Done. s={s:.6f}, method={method}, conf={conf:.2f}, elapsed={meta['elapsed_sec']}s")


if __name__ == "__main__":
    main()
