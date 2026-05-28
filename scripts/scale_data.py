#!/usr/bin/env python3
"""Scale lingbot inference data (depth/poses/point) by s, archive originals.

Usage: python scripts/scale_data.py <batch_id> <s>
"""

import numpy as np
import os
import sys
import time
import shutil
import pathlib

DATA_DIR = "/home/sscy/lingbot-map/stmem-main/data"
DATASAVE_DIR = "/home/sscy/lingbot-map/stmem-main/datasave"


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[scale_data {ts}] {msg}", flush=True)


def main():
    if len(sys.argv) < 3:
        log("ERROR: missing arguments")
        log("Usage: python scale_data.py <batch_id> <s>")
        sys.exit(1)

    batch_id = sys.argv[1]
    s = float(sys.argv[2])
    batch_dir = pathlib.Path(DATA_DIR) / batch_id

    if not batch_dir.is_dir():
        log(f"ERROR: batch dir not found: {batch_dir}")
        sys.exit(1)

    log(f"batch_id={batch_id} s={s:.6f}")

    # ── 1. Archive original data ──
    ts = time.strftime("%Y%m%d_%H%M%S")
    archive_dir = pathlib.Path(DATASAVE_DIR) / ts
    archive_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["poses", "depth", "rgb", "point"]:
        src = batch_dir / sub
        if src.is_dir():
            shutil.copytree(str(src), str(archive_dir / sub))
            log(f"Archived {sub} → {archive_dir / sub}")
    log(f"Original data archived: {archive_dir}")

    # ── 2. Scale depth PNGs ──
    depth_dir = batch_dir / "depth"
    if depth_dir.is_dir():
        import cv2
        for dp in sorted(depth_dir.glob("frame_*.png")):
            d = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED).astype(np.float32)
            d_scaled = (d * s).clip(0, 65535).astype(np.uint16)
            cv2.imwrite(str(dp), d_scaled)
        log(f"Scaled {len(list(depth_dir.glob('frame_*.png')))} depth images ×{s:.4f}")

    # ── 3. Scale poses ──
    poses_dir = batch_dir / "poses"
    if poses_dir.is_dir():
        count = 0
        for pp in sorted(poses_dir.glob("frame_*.txt")):
            c2w = np.loadtxt(str(pp))
            if c2w.shape == (4, 4):
                c2w[:3, 3] *= s  # scale translation only
                np.savetxt(str(pp), c2w, "%.15e")
                count += 1
        log(f"Scaled {count} poses ×{s:.4f}")

    # ── 4. Scale point clouds ──
    point_dir = batch_dir / "point"
    if point_dir.is_dir():
        count = 0
        for np_path in sorted(point_dir.glob("frame_*.npy")):
            pts = np.load(str(np_path)).astype(np.float64)
            pts *= s
            np.save(str(np_path), pts.astype(np.float32))
            count += 1
        log(f"Scaled {count} point clouds ×{s:.4f}")

    log("Done.")


if __name__ == "__main__":
    main()
