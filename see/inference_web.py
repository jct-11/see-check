#!/usr/bin/env python3
"""Complete inference + web export pipeline for LingBot-Map.

This script mirrors the inference logic from inference_api.py but additionally
exports camera parameters, frame images, and confidence data so that the
results can be viewed in a browser via the static Three.js viewer.

Usage (command line):
    python inference_web.py \
        --image_folder /path/to/images \
        --model_path checkpoints/robbyant/lingbot-map/lingbot-map-long.pt \
        --output_dir data/my_batch/output \
        --num_frames 200

Usage (import):
    from inference_web import run_inference_web
    result = run_inference_web(image_folder="...", model_path="...", output_dir="...")
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general, closed_form_inverse_se3
from lingbot_map.utils.load_fn import load_and_preprocess_images

_BATCHED_NDIMS = {
    "pose_enc": 3,
    "depth": 5,
    "depth_conf": 4,
    "world_points": 5,
    "world_points_conf": 4,
    "extrinsic": 4,
    "intrinsic": 4,
    "chunk_scales": 2,
    "chunk_transforms": 4,
    "images": 5,
}


def _squeeze_single_batch(key, value):
    batched_ndim = _BATCHED_NDIMS.get(key)
    if batched_ndim is None or not hasattr(value, "ndim"):
        return value
    if value.ndim == batched_ndim and value.shape[0] == 1:
        return value[0]
    return value


def _postprocess(predictions, images):
    """Convert pose encoding to extrinsics (c2w) and move to CPU.

    Same as inference_api.postprocess() but does NOT delete 'images' from
    predictions so that web export can access it.
    """
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )

    extrinsic_4x4 = torch.zeros(
        (*extrinsic.shape[:-2], 4, 4),
        device=extrinsic.device,
        dtype=extrinsic.dtype,
    )
    extrinsic_4x4[..., :3, :4] = extrinsic
    extrinsic_4x4[..., 3, 3] = 1.0
    extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
    extrinsic = extrinsic_4x4[..., :3, :4]

    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions.pop("pose_enc_list", None)

    print("Moving results to CPU...")
    for k in list(predictions.keys()):
        if isinstance(predictions[k], torch.Tensor):
            predictions[k] = _squeeze_single_batch(
                k, predictions[k].to("cpu", non_blocking=True)
            )
    images_cpu = images.to("cpu", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return predictions, images_cpu


def _write_ply(file_path, points, colors, use_binary=True):
    num_points = len(points)
    if num_points == 0:
        return
    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {num_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(file_path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(num_points):
            f.write(points[i].tobytes())
            f.write(colors[i].tobytes())


def run_inference_web(
    image_folder,
    model_path,
    output_dir,
    num_frames=200,
    image_size=518,
    patch_size=14,
    num_scale_frames=8,
    keyframe_interval=None,
    mode="streaming",
    conf_threshold=0.1,
    downsample_factor=4,
):
    """Run inference and export complete web-viewable results.

    Returns:
        dict with keys: success, num_frames, total_points, frame_files,
                        merged_ply_path, output_dir, viewer_url_hint
    """
    t_start = time.time()

    # ── Device ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference_web] Device: {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # ── Load images ────────────────────────────────────────────────────
    print(f"[inference_web] Loading images from {image_folder}...")
    paths = []
    for ext in [".jpg", ".png", ".jpeg"]:
        paths.extend(
            sorted(
                [f for f in os.listdir(image_folder) if f.endswith(ext)]
            )
        )
    paths = paths[:num_frames]
    paths = [os.path.join(image_folder, p) for p in paths]

    if len(paths) == 0:
        return {"success": False, "error": f"No images found in {image_folder}"}

    print(f"[inference_web] Found {len(paths)} images")
    images = load_and_preprocess_images(
        paths, mode="crop", image_size=image_size, patch_size=patch_size
    )
    print(f"[inference_web] Images loaded: shape={images.shape}")

    # ── Load model ─────────────────────────────────────────────────────
    print(f"[inference_web] Loading model from {model_path}...")
    model = GCTStream(
        img_size=image_size,
        patch_size=patch_size,
        enable_3d_rope=True,
        max_frame_num=1024,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=False,
        camera_num_iterations=4,
    )

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    print("[inference_web] Model loaded")

    # ── Run inference ──────────────────────────────────────────────────
    if torch.cuda.is_available():
        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
    else:
        dtype = torch.float32

    if dtype != torch.float32 and hasattr(model, "aggregator"):
        model.aggregator = model.aggregator.to(dtype=dtype)

    images = images.to(device)

    if keyframe_interval is None:
        if mode == "streaming" and images.shape[0] > 320:
            keyframe_interval = (images.shape[0] + 319) // 320
            print(
                f"[inference_web] Auto keyframe_interval={keyframe_interval} "
                f"(frames={images.shape[0]} > 320)"
            )
        else:
            keyframe_interval = 1

    print(f"[inference_web] Running {mode} inference (dtype={dtype})...")
    t_inf = time.time()

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if mode == "streaming":
            predictions = model.inference_streaming(
                images,
                num_scale_frames=num_scale_frames,
                keyframe_interval=keyframe_interval,
                output_device=None,
            )
        else:
            predictions = model.inference_windowed(
                images,
                window_size=64,
                overlap_size=16,
                num_scale_frames=num_scale_frames,
                keyframe_interval=keyframe_interval,
                output_device=None,
            )

    print(f"[inference_web] Inference done in {time.time() - t_inf:.1f}s")
    if torch.cuda.is_available():
        print(
            f"[inference_web] GPU peak: "
            f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        )

    # ── Postprocess ────────────────────────────────────────────────────
    predictions, images_cpu = _postprocess(predictions, images)
    print(f"[inference_web] Predictions keys: {list(predictions.keys())}")

    # ── Save PLY files ─────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    print(f"[inference_web] Saving point clouds to {output_dir}...")

    frame_files = []
    all_points_list = []
    all_colors_list = []
    all_conf_list = []

    if "world_points" in predictions and images_cpu is not None:
        point_data = predictions["world_points"]  # (S, H, W, 3)
        conf_data = predictions.get("world_points_conf")  # (S, H, W)
        S_result = point_data.shape[0]

        for i in range(S_result):
            points = point_data[i].reshape(-1, 3)

            # Apply confidence filter
            if conf_data is not None:
                conf = conf_data[i].reshape(-1)
                mask = conf >= conf_threshold
                points = points[mask]
            else:
                mask = None

            if len(points) == 0:
                continue

            # Get colors
            img = images_cpu[i]  # (3, H, W)
            img_hwc = img.permute(1, 2, 0)  # (H, W, 3)
            img_colors = img_hwc.reshape(-1, 3)
            if mask is not None:
                img_colors = img_colors[mask]

            # Track confidence for this frame
            if conf_data is not None:
                frame_conf = conf_data[i].reshape(-1)
                if mask is not None:
                    frame_conf = frame_conf[mask]
            else:
                frame_conf = np.ones(len(points), dtype=np.float32)
            colors = (img_colors.clamp(0, 1) * 255).byte()

            # Downsample
            if downsample_factor > 1 and len(points) > downsample_factor:
                idx = np.random.choice(
                    len(points),
                    len(points) // downsample_factor,
                    replace=False,
                )
                points = points[idx]
                colors = colors[idx]
                frame_conf = frame_conf[idx]

            # Write PLY
            ply_path = os.path.join(output_dir, f"frame_{i:06d}.ply")
            _write_ply(ply_path, points.numpy(), colors.numpy())
            frame_files.append(ply_path)
            all_points_list.append(points.numpy())
            all_colors_list.append(colors.numpy())
            all_conf_list.append(frame_conf.astype(np.float32))

            if i % 50 == 0:
                print(f"[inference_web]   frame {i}/{S_result}: {len(points_world)} pts")

        print(f"[inference_web] Saved {len(frame_files)} frame PLY files")

        # Merged PLY
        if all_points_list:
            merged_points = np.vstack(all_points_list)
            merged_colors = np.vstack(all_colors_list)
            merged_ply_path = os.path.join(output_dir, "merged_pointcloud.ply")
            _write_ply(merged_ply_path, merged_points, merged_colors)
            print(
                f"[inference_web] Saved merged PLY: "
                f"{len(merged_points)} points"
            )
        else:
            merged_ply_path = None

    elif "depth" in predictions and "intrinsic" in predictions and images_cpu is not None:
        # Fallback: reconstruct 3D points from depth map + intrinsics
        print("[inference_web] Reconstructing 3D points from depth maps...")
        intrinsic = predictions["intrinsic"]  # (S, 3, 3)
        depth_data = predictions["depth"]     # (S, H, W) or (S, 1, H, W)
        conf_data = predictions.get("depth_conf")

        S_result = depth_data.shape[0]
        H_img, W_img = images_cpu.shape[-2], images_cpu.shape[-1]

        for i in range(S_result):
            depth = depth_data[i].squeeze()  # (H, W)
            img = images_cpu[i].permute(1, 2, 0)  # (H, W, 3)

            # Valid depth mask
            valid_mask = depth > 0
            if conf_data is not None:
                conf = conf_data[i].squeeze()
                valid_mask = valid_mask & (conf >= conf_threshold)

            if valid_mask.sum() == 0:
                continue

            # Get intrinsics for this frame
            fx = float(intrinsic[i, 0, 0])
            fy = float(intrinsic[i, 1, 1])
            cx = float(intrinsic[i, 0, 2])
            cy = float(intrinsic[i, 1, 2])

            # Unproject depth to 3D (camera-local coordinates)
            u_coords = np.arange(W_img)
            v_coords = np.arange(H_img)
            u_grid, v_grid = np.meshgrid(u_coords, v_coords)

            z = depth[valid_mask].numpy()
            u_v = u_grid[valid_mask]
            v_v = v_grid[valid_mask]

            x = (u_v - cx) * z / fx
            y = (v_v - cy) * z / fy

            # Camera-local points: shape (N, 3)
            points_cam = np.stack([x, y, z], axis=1)

            # ── Transform to world coords (viser's depth_to_world_coords_points) ──
            ext_i = predictions["extrinsic"][i].numpy()  # (3, 4) c2w
            # viser inverts c2w→w2c, then: world = cam @ R_w2c.T + t_w2c
            ext_4x4 = np.eye(4, dtype=np.float64)
            ext_4x4[:3, :3] = ext_i[:3, :3]
            ext_4x4[:3, 3] = ext_i[:3, 3]
            w2c = np.linalg.inv(ext_4x4)
            R_w2c = w2c[:3, :3].astype(np.float32)
            t_w2c = w2c[:3, 3].astype(np.float32)

            points_world = points_cam @ R_w2c.T + t_w2c  # (N, 3)

            img_np = (img[valid_mask].numpy().clip(0, 1) * 255).astype(np.uint8)

            # Collect confidence for this frame (before downsampling)
            if conf_data is not None:
                frame_conf_full = conf_data[i].squeeze().numpy()[valid_mask]
            else:
                frame_conf_full = np.ones(len(points_world), dtype=np.float32)

            # Downsample
            if downsample_factor > 1 and len(points_world) > downsample_factor:
                idx = np.random.choice(
                    len(points_world),
                    len(points_world) // downsample_factor,
                    replace=False,
                )
                points_world = points_world[idx]
                img_np = img_np[idx]
                frame_conf_full = frame_conf_full[idx]

            # Write PLY
            ply_path = os.path.join(output_dir, f"frame_{i:06d}.ply")
            _write_ply(ply_path, points_world, img_np)
            frame_files.append(ply_path)
            all_points_list.append(points_world)
            all_colors_list.append(img_np.astype(np.float32))
            all_conf_list.append(frame_conf_full.astype(np.float32))

            if i % 50 == 0:
                print(f"[inference_web]   frame {i}/{S_result}: {len(points_world)} pts")

        print(f"[inference_web] Saved {len(frame_files)} frame PLY files (from depth)")

        # Merged PLY
        if all_points_list:
            merged_points = np.vstack(all_points_list)
            merged_colors = np.vstack(all_colors_list).astype(np.uint8)
            merged_ply_path = os.path.join(output_dir, "merged_pointcloud.ply")
            _write_ply(merged_ply_path, merged_points, merged_colors)
            print(f"[inference_web] Saved merged PLY: {len(merged_points)} points")
        else:
            merged_ply_path = None

    else:
        print("[inference_web] No point cloud data (world_points or depth+intrinsic) found")
        print(f"[inference_web] Available keys: {list(predictions.keys())}")
        merged_ply_path = None

    # ── Save merged confidence binary (matches PLY point ordering) ──────
    if all_conf_list:
        merged_conf = np.concatenate(all_conf_list).astype(np.float32)
        conf_path = os.path.join(output_dir, "merged_confidence.bin")
        merged_conf.tofile(conf_path)
        print(f"[inference_web] Saved merged confidence: {len(merged_conf)} values")

    # ── Export web visualization data ──────────────────────────────────
    from lingbot_map.vis.web_export import export_web_data

    export_web_data(
        predictions=predictions,
        images_cpu=images_cpu,
        output_dir=output_dir,
        conf_threshold=conf_threshold,
    )

    # ── Cleanup ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    total_points = sum(len(p) for p in all_points_list)

    result = {
        "success": True,
        "num_frames": len(frame_files),
        "total_points": total_points,
        "frame_files": frame_files,
        "merged_ply_path": merged_ply_path,
        "output_dir": os.path.abspath(output_dir),
        "elapsed_seconds": elapsed,
    }
    print(f"[inference_web] Done in {elapsed:.1f}s. "
          f"Viewer: {output_dir}/viewer.html")
    return result


# ── CLI entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LingBot-Map inference + web export"
    )
    parser.add_argument(
        "--image_folder", type=str, required=True, help="Path to input images"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/robbyant/lingbot-map/lingbot-map-long.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: pointcloud/pointcloud_{timestamp}/)",
    )
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument("--keyframe_interval", type=int, default=None)
    parser.add_argument(
        "--mode", type=str, default="streaming",
        choices=["streaming", "windowed"]
    )
    parser.add_argument("--conf_threshold", type=float, default=0.1)
    parser.add_argument("--downsample_factor", type=int, default=4)

    args = parser.parse_args()

    if args.output_dir is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "pointcloud",
            f"pointcloud_{ts}",
        )

    result = run_inference_web(
        image_folder=args.image_folder,
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_scale_frames=args.num_scale_frames,
        keyframe_interval=args.keyframe_interval,
        mode=args.mode,
        conf_threshold=args.conf_threshold,
        downsample_factor=args.downsample_factor,
    )

    if result["success"]:
        print(f"\n{'='*60}")
        print(f"Inference + web export complete!")
        print(f"Output: {result['output_dir']}")
        print(f"Frames: {result['num_frames']}")
        print(f"Total points: {result['total_points']:,}")
        print(f"Elapsed: {result['elapsed_seconds']:.1f}s")
        print(f"\nTo view in browser:")
        print(f"  cd {result['output_dir']}")
        print(f"  python3 -m http.server 8080")
        print(f"  # Then open http://<host>:8080/viewer.html")
        print(f"{'='*60}")
    else:
        print(f"ERROR: {result.get('error', 'Unknown error')}")
        sys.exit(1)
