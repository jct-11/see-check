"""Convert params_with_idx.npz to web-friendly binary format.

Usage: python scripts/convert_memory_pc.py

Input:  /home/liangjiahua/dgsg-orin/experiments/mydata/lingbot/params_with_idx.npz
        /home/liangjiahua/dgsg-orin/experiments/mydata/lingbot/scene_graph.json
Output: src/web/assets/memory_pc.bin
        src/web/assets/memory_scene_graph.json
"""

import numpy as np
import shutil
import struct
import os
import sys

NPZ_PATH = "/home/liangjiahua/dgsg-orin/experiments/mydata/lingbot/params_with_idx.npz"
SG_PATH = "/home/liangjiahua/dgsg-orin/experiments/mydata/lingbot/scene_graph.json"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ASSETS_DIR = os.path.join(PROJECT_DIR, "src", "web", "assets")
BIN_OUT = os.path.join(ASSETS_DIR, "memory_pc.bin")
JSON_OUT = os.path.join(ASSETS_DIR, "memory_scene_graph.json")


def main():
    try:
        print(f"Loading {NPZ_PATH} ...")
        data = np.load(NPZ_PATH)
        means3D = data["means3D"].astype(np.float32)
        rgb_colors = data["rgb_colors"].astype(np.float32)
        object_idx = data["object_idx"].astype(np.uint16)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: missing key {e} in {NPZ_PATH}", file=sys.stderr)
        sys.exit(1)

    n = len(means3D)
    print(f"Total points: {n}")

    # Binary layout: [N: uint32][positions: float32[N*3]][colors: float32[N*3]][object_idx: uint16[N]]
    # Offsets: N at 0, positions at 4, colors at 4+12N, idx at 4+24N
    pos_bytes = means3D.tobytes()
    col_bytes = rgb_colors.tobytes()
    idx_bytes = object_idx.tobytes()

    os.makedirs(ASSETS_DIR, exist_ok=True)

    with open(BIN_OUT, "wb") as f:
        f.write(struct.pack("<I", n))           # point count (4 bytes)
        f.write(pos_bytes)                       # positions (12N bytes)
        f.write(col_bytes)                       # colors (12N bytes)
        f.write(idx_bytes)                       # object_idx (2N bytes)

    size_mb = os.path.getsize(BIN_OUT) / (1024 * 1024)
    print(f"Wrote {BIN_OUT} ({size_mb:.1f} MB)")

    shutil.copy2(SG_PATH, JSON_OUT)
    print(f"Copied {SG_PATH} -> {JSON_OUT}")

    # Verify
    with open(BIN_OUT, "rb") as f:
        n_read = struct.unpack("<I", f.read(4))[0]
        assert n_read == n, f"Verification failed: {n_read} != {n}"

    print("Done.")


if __name__ == "__main__":
    main()
