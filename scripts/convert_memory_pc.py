"""Convert params_with_idx.npz to web-friendly binary format.

Usage: python scripts/convert_memory_pc.py <batch_id>

Input:  /home/liangjiahua/dgsg-orin/experiments/mydata/<batch_id>/params_with_idx.npz
        /home/liangjiahua/dgsg-orin/experiments/mydata/<batch_id>/scene_graph.json
Output: src/web/assets/memory_pc.bin
        src/web/assets/memory_scene_graph.json
"""

import numpy as np
import shutil
import struct
import os
import sys
import time
import traceback

DGSG_EXP_DIR = "/home/liangjiahua/dgsg-orin/experiments/mydata"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ASSETS_DIR = os.path.join(PROJECT_DIR, "src", "web", "assets")
BIN_OUT = os.path.join(ASSETS_DIR, "memory_pc.bin")
JSON_OUT = os.path.join(ASSETS_DIR, "memory_scene_graph.json")


def log(msg):
    """Timestamped log to stdout (captured by batch_service subprocess)."""
    ts = time.strftime("%H:%M:%S")
    print(f"[convert {ts}] {msg}", flush=True)


def main():
    t0 = time.time()

    if len(sys.argv) < 2:
        log("ERROR: missing batch_id argument")
        log("Usage: python scripts/convert_memory_pc.py <batch_id>")
        sys.exit(1)

    batch_id = sys.argv[1]
    exp_dir = os.path.join(DGSG_EXP_DIR, batch_id)
    npz_path = os.path.join(exp_dir, "params_with_idx.npz")
    sg_path = os.path.join(exp_dir, "scene_graph.json")

    log(f"batch_id={batch_id}")
    log(f"npz_path={npz_path}")
    log(f"sg_path={sg_path}")

    # --- Load NPZ ---
    t_load = time.time()
    try:
        data = np.load(npz_path)
        means3D = data["means3D"].astype(np.float32)
        rgb_colors = data["rgb_colors"].astype(np.float32)
        object_idx = data["object_idx"].astype(np.uint16)
    except FileNotFoundError as e:
        log(f"ERROR: NPZ not found: {npz_path}")
        sys.exit(1)
    except KeyError as e:
        log(f"ERROR: missing key {e} in NPZ. Available keys: {list(data.keys())}")
        sys.exit(1)
    except Exception as e:
        log(f"ERROR: failed to load NPZ: {e}")
        traceback.print_exc()
        sys.exit(1)

    n = min(len(means3D), len(rgb_colors), len(object_idx))
    means3D = means3D[:n]
    rgb_colors = rgb_colors[:n]
    object_idx = object_idx[:n]
    load_ms = (time.time() - t_load) * 1000
    unique_ids = len(set(int(x) for x in object_idx[:10000]))  # sample first 10k
    log(f"loaded {n} points, ~{unique_ids} unique object IDs (sampled), load_time={load_ms:.0f}ms")

    # --- Write binary ---
    t_write = time.time()
    pos_bytes = means3D.tobytes()
    col_bytes = rgb_colors.tobytes()
    idx_bytes = object_idx.tobytes()

    os.makedirs(ASSETS_DIR, exist_ok=True)

    with open(BIN_OUT, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(pos_bytes)
        f.write(col_bytes)
        f.write(idx_bytes)

    size_mb = os.path.getsize(BIN_OUT) / (1024 * 1024)
    write_ms = (time.time() - t_write) * 1000
    log(f"wrote {BIN_OUT} ({size_mb:.1f} MB, write_time={write_ms:.0f}ms)")

    # --- Copy scene_graph ---
    if os.path.exists(sg_path):
        shutil.copy2(sg_path, JSON_OUT)
        sg_size = os.path.getsize(JSON_OUT)
        log(f"copied {sg_path} -> {JSON_OUT} ({sg_size} bytes)")
    else:
        log(f"WARNING: scene_graph.json not found at {sg_path}, labels will be missing")

    # --- Verify ---
    with open(BIN_OUT, "rb") as f:
        n_read = struct.unpack("<I", f.read(4))[0]
        if n_read != n:
            log(f"ERROR: verification failed: header N={n_read} != expected {n}")
            sys.exit(1)

    total_ms = (time.time() - t0) * 1000
    log(f"DONE in {total_ms:.0f}ms — {BIN_OUT}")


if __name__ == "__main__":
    main()
