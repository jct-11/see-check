"""LingBot-MAP live camera capture + streaming 3D reconstruction.

Captures frames from an IP camera and saves them to a local folder, while
simultaneously running LingBot-MAP in live streaming mode on the same folder.
A live 3D viewer shows the point cloud and camera trajectory growing in real time.

Usage:
    python live_camera.py --model_path /path/to/checkpoint.pt \
        --camera_url https://192.168.0.28:8080 --capture_fps 5 --max_images 1000
"""

import argparse
import os
import sys
import time
import ssl
import shutil
import threading
from urllib.request import urlopen, Request

import cv2
import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import (
    unproject_depth_map_to_point_map,
    closed_form_inverse_se3_general,
)

from stream import (
    inference_live,
    load_images,
    load_model,
)


def camera_capture_loop(camera_url, output_dir, fps, max_images, stop_event):
    """Background thread: capture frames from IP camera (MJPEG stream) and save as JPEG.

    Uses raw HTTP stream reading instead of cv2.VideoCapture to avoid
    FFmpeg MJPEG compatibility issues (e.g. "Stream ends prematurely").

    Saves images as 000000.jpg, 000001.jpg, ... in output_dir.
    Stops when stop_event is set or max_images is reached.

    Args:
        camera_url: URL of the IP camera MJPEG stream.
        output_dir: Directory to save captured frames.
        fps: Target capture frames per second.
        max_images: Maximum number of images to capture.
        stop_event: threading.Event to signal capture stop.
    """
    interval = 1.0 / fps
    saved = 0
    
    # Auto-append /video if missing (common for IP cameras)
    if not camera_url.endswith("/video"):
        camera_url = camera_url.rstrip("/") + "/video"
        print(f"[Camera] Auto-corrected URL to: {camera_url}")
    
    print(f"[Camera] Connecting to MJPEG stream at {camera_url} ...")

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    req = Request(camera_url)

    try:
        stream = urlopen(req, context=ssl_context, timeout=10)
    except Exception as e:
        print(f"[Camera] Failed to open stream at {camera_url}: {e}")
        return

    content_type = stream.headers.get("Content-Type", "")
    if "multipart" not in content_type:
        print(f"[Camera] Warning: unexpected Content-Type '{content_type}', "
              f"trying as MJPEG anyway...")

    boundry = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundry = part.split("=", 1)[1].strip().encode()
            break

    if boundry is None:
        boundry = b"--boundarydonotcross"

    buf = b""
    print(f"[Camera] Streaming MJPEG at {fps} fps, max {max_images} images")

    try:
        while not stop_event.is_set() and saved < max_images:
            loop_start = time.time()

            chunk = stream.read(65536)
            if not chunk:
                print("[Camera] Stream ended.")
                break
            buf += chunk

            if b"\xff\xd8" not in buf or b"\xff\xd9" not in buf:
                if len(buf) > 10 * 1024 * 1024:
                    buf = buf[-1024 * 1024:]
                continue

            while True:
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9")
                if start == -1 or end == -1 or end < start:
                    break

                jpeg_data = buf[start:end + 2]
                buf = buf[end + 2:]

                if len(jpeg_data) > 1024:
                    frame = cv2.imdecode(
                        np.frombuffer(jpeg_data, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if frame is not None and frame.size > 0:
                        path = os.path.join(output_dir, f"{saved:06d}.jpg")
                        cv2.imwrite(path, frame)
                        saved += 1

                        elapsed = time.time() - loop_start
                        if elapsed < interval:
                            time.sleep(interval - elapsed)

                        if saved % 50 == 0:
                            print(f"[Camera] Captured {saved}/{max_images} images")

                        if saved >= max_images:
                            break

                if saved >= max_images:
                    break

            if saved >= max_images:
                break

    except Exception as e:
        print(f"[Camera] Stream error: {e}")

    try:
        stream.close()
    except Exception:
        pass

    if saved >= max_images:
        print(f"[Camera] Reached max_images ({max_images}). Capture stopped.")
        stop_event.set()
    else:
        print(f"[Camera] Capture ended. Saved {saved} images.")


def main():
    parser = argparse.ArgumentParser(
        description="LingBot-MAP: Live Camera Capture + Streaming 3D Reconstruction"
    )

    # Camera
    parser.add_argument("--camera_url", type=str, default="https://192.168.0.28:8080",
                        help="IP camera stream URL")
    parser.add_argument("--capture_fps", type=float, default=5.0,
                        help="Camera capture speed in fps (default: 5.0)")
    parser.add_argument("--max_images", type=int, default=1000,
                        help="Maximum images to capture before pausing (default: 1000)")
    parser.add_argument("--output_dir", type=str, default="live_capture",
                        help="Directory for captured images (default: live_capture/)")

    # Model (same as stream.py)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)

    # Streaming options
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument("--keyframe_interval", type=int, default=None,
                        help="Keyframe interval for KV cache. If unset, auto-selected based on max_images "
                             "(1 when max_images <= 320, else ceil(max_images / 320)).")
    parser.add_argument("--enable_3d_rope", action="store_true", default=True)
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--camera_num_iterations", type=int, default=4)
    parser.add_argument("--use_sdpa", action="store_true", default=False)
    parser.add_argument("--offload_to_cpu", type=bool, default=True)

    # Live polling
    parser.add_argument("--poll_interval", type=float, default=1.0,
                        help="Seconds between folder scans in live mode (default: 1.0)")

    # Visualization
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--conf_threshold", type=float, default=0.7,
                        help="Confidence threshold filter (default: 0.7). Points with confidence below this are filtered out.")
    parser.add_argument("--downsample_factor", type=int, default=10)
    parser.add_argument("--point_size", type=float, default=0.00001)
    parser.add_argument("--mask_sky", action="store_true")
    parser.add_argument("--vis_interval", type=int, default=5,
                        help="Update 3D viewer every N frames (default: 5 to reduce callback overhead).")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Create output directory -------------------------------------------------
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Camera output directory: {os.path.abspath(args.output_dir)}")

    # ---- Start camera capture in background thread --------------------------------
    stop_event = threading.Event()
    camera_thread = threading.Thread(
        target=camera_capture_loop,
        args=(args.camera_url, args.output_dir, args.capture_fps,
              args.max_images, stop_event),
        daemon=True,
    )
    camera_thread.start()

    # ---- Wait for initial images to accumulate -----------------------------------
    print(f"Waiting for at least {args.num_scale_frames} images for scale phase...")
    while True:
        jpg_files = sorted(
            f for f in os.listdir(args.output_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if len(jpg_files) >= args.num_scale_frames:
            break
        time.sleep(0.5)

    # ---- Load initial images -----------------------------------------------------
    t0 = time.time()
    print(f"Loading {len(jpg_files)} initial images...")
    images, paths, resolved_image_folder = load_images(
        image_folder=args.output_dir,
        image_size=args.image_size,
        patch_size=args.patch_size,
    )
    print(f"Initial images loaded: {images.shape[0]} frames, {images.shape[-2]}x{images.shape[-1]}")

    # ---- Build model --------------------------------------------------------------
    model = load_model(args, device)
    print(f"Total load time: {time.time() - t0:.1f}s")

    # Pick inference dtype
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        print(f"Casting aggregator to {dtype} (heads kept in fp32)")
        model.aggregator = model.aggregator.to(dtype=dtype)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            f"GPU mem after load: "
            f"alloc={torch.cuda.memory_allocated()/1e9:.2f} GB, "
            f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB"
        )

    output_device = torch.device("cpu") if args.offload_to_cpu else None

    # ---- Auto-select keyframe_interval based on max_images (same as demo.py) -------
    if args.keyframe_interval is None:
        if args.max_images > 320:
            args.keyframe_interval = (args.max_images + 319) // 320
            print(f"Auto-selected --keyframe_interval={args.keyframe_interval} "
                  f"(max_images={args.max_images} > 320, KV cache ~320 keyframes)")
        else:
            args.keyframe_interval = 1
            print(f"Auto-selected --keyframe_interval=1 (max_images={args.max_images} <= 320)")

        kv_cache_size = args.num_scale_frames + \
                        max(0, args.window_size - args.num_scale_frames) * args.keyframe_interval \
                        if hasattr(args, 'window_size') else args.num_scale_frames + 312 * args.keyframe_interval
        print(f"Estimated KV cache capacity: ~{kv_cache_size} frames")

    # ---- Start live viewer BEFORE inference --------------------------------------
    from lingbot_map.vis.live_viewer import LivePointCloudViewer
    viewer = LivePointCloudViewer(
        port=args.port,
        conf_threshold=args.conf_threshold,
        downsample_factor=args.downsample_factor,
        point_size=args.point_size,
        max_visible_frames=300,
    )
    print(f"Live 3D viewer started at http://localhost:{args.port}")

    # ---- Build per-frame callback for live visualization -------------------------
    _vis_counter = 0
    _vis_sent = 0

    def _on_live_frame(frame_idx, image_np, frame_output):
        """Callback: postprocess a single frame and add to live viewer.

        IMPORTANT: pose_encoding_to_extri_intri returns C2W (the model predicts
        camera-from-world, i.e. camera position in world frame).  We pass C2W
        directly to the viewer — no inversion needed.

        The matching demo.py postprocess() does: C2W → inv(C2W) → W2C (store),
        then PointCloudViewer does: inv(W2C) → C2W (use).  Same end result.

        Updates visualization only every args.vis_interval frames to reduce
        WebSocket overhead and browser rendering load.
        """
        nonlocal _vis_counter, _vis_sent

        if frame_idx % args.vis_interval != 0:
            _vis_counter += 1

        _vis_sent += 1
        if _vis_sent % 10 == 0:
            print(f"  [vis] Sent frame {frame_idx} to viewer (#{_vis_sent})")

        pose_enc = frame_output["pose_enc"]
        depth = frame_output.get("depth")
        depth_conf = frame_output.get("depth_conf")
        world_points = frame_output.get("world_points")

        H, W = image_np.shape[:2]

        if pose_enc is None or depth is None or depth_conf is None:
            return

        # Unbatch: model outputs [B=1, S=1, ...]
        if pose_enc.dim() == 3:
            _pose_enc = pose_enc[0, 0]
        elif pose_enc.dim() == 2:
            _pose_enc = pose_enc[0]
        else:
            _pose_enc = pose_enc

        if depth.dim() == 5:
            _depth = depth[0, 0]
            _depth_conf = depth_conf[0, 0]
        elif depth.dim() == 4:
            _depth = depth[0]
            _depth_conf = depth_conf[0]
        else:
            _depth = depth
            _depth_conf = depth_conf

        # Step 1: Decode pose encoding → C2W extrinsic [3x4] + intrinsic [3x3]
        _pose_enc_batch = _pose_enc.unsqueeze(0).unsqueeze(0)
        extrinsic_c2w, intrinsic = pose_encoding_to_extri_intri(
            _pose_enc_batch, image_size_hw=(H, W)
        )
        extrinsic_c2w = extrinsic_c2w[0, 0]
        intrinsic = intrinsic[0, 0]

        intrinsic_np = intrinsic.cpu().numpy()
        _depth_np = _depth.cpu().numpy()
        _depth_conf_np = _depth_conf.cpu().numpy()

        # Step 2: Get world points (model output or fallback unproject)
        if world_points is not None:
            if world_points.dim() == 5:
                _wp = world_points[0, 0].cpu().numpy()
            elif world_points.dim() == 4:
                _wp = world_points[0].cpu().numpy()
            else:
                _wp = world_points.cpu().numpy()
            _src = "model"
        else:
            # unproject depth → world: need W2C = inv(C2W)
            c2w_4x4 = torch.eye(4, device=extrinsic_c2w.device, dtype=extrinsic_c2w.dtype)
            c2w_4x4[:3, :4] = extrinsic_c2w
            w2c = closed_form_inverse_se3_general(c2w_4x4.unsqueeze(0)).squeeze(0)
            w2c_np = w2c[:3, :4].cpu().numpy()
            wp_t = unproject_depth_map_to_point_map(
                _depth_np[None],
                w2c_np[np.newaxis],
                intrinsic_np[np.newaxis],
            )
            _wp = wp_t[0]
            _src = "unproj"

        extrinsic_c2w_np = extrinsic_c2w.cpu().numpy()

        if frame_idx % 20 == 0:
            valid = (np.isfinite(_wp).all(axis=-1) & (_depth_conf_np > 0)).sum()
            print(f"  [vis] frame {frame_idx}: world_pts={_wp.shape} "
                  f"src={_src} conf>0={valid} "
                  f"conf_mean={_depth_conf_np.mean():.3f} "
                  f"depth_median={np.median(_depth_np[_depth_conf_np > 0]):.3f}"
                  if _depth_conf_np.sum() > 0 else
                  f"  [vis] frame {frame_idx}: world_pts={_wp.shape} src={_src} all_zero_conf")

        viewer.add_frame(
            frame_idx, image_np, _wp, _depth_conf_np,
            extrinsic_c2w_np, intrinsic_np,
        )

    # ---- Live inference -----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Live inference: camera={args.camera_url}, "
          f"capture_fps={args.capture_fps}, max_images={args.max_images}")
    print(f"keyframe_interval={args.keyframe_interval}, poll_interval={args.poll_interval}s")
    print(f"Press Ctrl+C to stop inference (viewer stays open).")
    print(f"{'='*60}\n")

    t0 = time.time()
    try:
        predictions = inference_live(
            model,
            initial_images=images,
            initial_paths=paths,
            image_folder=resolved_image_folder,
            num_scale_frames=args.num_scale_frames,
            keyframe_interval=args.keyframe_interval,
            output_device=output_device,
            model_device=device,
            dtype=dtype,
            poll_interval=args.poll_interval,
            on_frame=_on_live_frame,
            on_frame_interval=1,
        )
    except KeyboardInterrupt:
        print("\nInference interrupted by user.")
    finally:
        del images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nInference done in {time.time() - t0:.1f}s")
    if torch.cuda.is_available():
        print(
            f"GPU peak during inference: "
            f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB "
            f"(reserved peak {torch.cuda.max_memory_reserved()/1e9:.2f} GB)"
        )

    # ---- Keep viewer running -----------------------------------------------------
    print(f"\nViewer still running at http://localhost:{args.port}")
    print("Press Ctrl+C to exit.")
    viewer.run_forever()

    # Signal camera thread to stop
    stop_event.set()
    camera_thread.join(timeout=5)


if __name__ == "__main__":
    main()
