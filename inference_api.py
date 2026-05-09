"""可被调用的推理接口模块

这个模块提供可以被batch_service.py调用的推理函数
严格参考demo.py的推理流程
"""

import os
import time
import traceback
import torch
import numpy as np
from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general
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
    """Drop the leading batch dimension for single-sequence demo outputs."""
    batched_ndim = _BATCHED_NDIMS.get(key)
    if batched_ndim is None or not hasattr(value, "ndim"):
        return value
    if value.ndim == batched_ndim and value.shape[0] == 1:
        return value[0]
    return value


def postprocess(predictions, images):
    """Convert pose encoding to extrinsics (c2w) and move to CPU."""
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    extrinsic_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
    extrinsic_4x4[..., :3, :4] = extrinsic
    extrinsic_4x4[..., 3, 3] = 1.0
    extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
    extrinsic = extrinsic_4x4[..., :3, :4]

    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions.pop("pose_enc_list", None)
    predictions.pop("images", None)

    print("Moving results to CPU...")
    for k in list(predictions.keys()):
        if isinstance(predictions[k], torch.Tensor):
            predictions[k] = _squeeze_single_batch(k, predictions[k].to("cpu", non_blocking=True))
    images_cpu = images.to("cpu", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return predictions, images_cpu


def write_ply(file_path, points, colors, use_binary=True):
    """保存点云为PLY文件"""
    num_points = len(points)
    if num_points == 0:
        return
    
    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)
    
    if use_binary:
        header = f"ply\nformat binary_little_endian 1.0\nelement vertex {num_points}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        with open(file_path, 'wb') as f:
            f.write(header.encode('ascii'))
            for i in range(num_points):
                f.write(points[i].tobytes())
                f.write(colors[i].tobytes())
    else:
        header = f"ply\nformat ascii 1.0\nelement vertex {num_points}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        with open(file_path, 'w') as f:
            f.write(header)
            for i in range(num_points):
                x, y, z = points[i]
                r, g, b = colors[i]
                f.write(f"{x} {y} {z} {r} {g} {b}\n")

def save_npy(file_path, data):
    """保存numpy数组为.npy文件"""
    np.save(file_path, data)
    print(f"Saved {file_path} (shape: {data.shape})")

def save_json(file_path, data):
    """保存数据为JSON文件"""
    import json
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {file_path}")

def save_depth_png(file_path, depth):
    """保存深度图为PNG文件"""
    import cv2
    # 归一化到0-255
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255
    depth_uint8 = depth_normalized.astype(np.uint8)
    cv2.imwrite(file_path, depth_uint8)
    print(f"Saved depth map {file_path}")

  
def run_inference(image_folder, model_path, output_dir, num_frames=200, image_size=518, patch_size=14, num_scale_frames=8, keyframe_interval=1, mode="streaming"):
    """可被调用的推理函数接口

    Args:
        image_folder: 图片文件夹路径
        model_path: 模型路径
        output_dir: 输出目录
        num_frames: 帧数量
        image_size: 图片大小
        patch_size: patch大小
        num_scale_frames: scale帧数
        keyframe_interval: 关键帧间隔
        mode: 推理模式 ("streaming" 或 "windowed")

    Returns:
        dict: 包含点云数据的字典
    """
    try:
        # 设置内存优化环境变量
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # 清理 GPU 内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"GPU mem before load: alloc={torch.cuda.memory_allocated()/1e9:.2f} GB, reserved={torch.cuda.memory_reserved()/1e9:.2f} GB")

        # 加载图片
        print(f"Loading images from {image_folder}...")
        paths = []
        for ext in [".jpg", ".png", ".jpeg"]:
            paths.extend(sorted([f for f in os.listdir(image_folder) if f.endswith(ext)]))
        paths = paths[:num_frames]
        paths = [os.path.join(image_folder, p) for p in paths]

        if len(paths) == 0:
            return {"success": False, "error": f"No images found in {image_folder}"}

        print(f"Found {len(paths)} images")
        images = load_and_preprocess_images(paths, mode="crop", image_size=image_size, patch_size=patch_size)
        print(f"Images loaded: shape={images.shape}")

        # 加载模型
        print(f"Loading model from {model_path}...")
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
        print(f"Model loaded successfully")

        # 推理
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dtype = torch.float32

        if dtype != torch.float32 and hasattr(model, "aggregator"):
            model.aggregator = model.aggregator.to(dtype=dtype)

        images = images.to(device)
        print(f"Input: {images.shape[0]} frames, shape {tuple(images.shape)}")
        print(f"Mode: {mode}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU mem after load: alloc={torch.cuda.memory_allocated()/1e9:.2f} GB")

        # keyframe_interval 自动选择（参考demo.py）
        if keyframe_interval is None:
            if mode == "streaming" and images.shape[0] > 320:
                keyframe_interval = (images.shape[0] + 319) // 320
                print(f"Auto-selected keyframe_interval={keyframe_interval} (num_frames={images.shape[0]} > 320)")
            else:
                keyframe_interval = 1

        print(f"Running {mode} inference (dtype={dtype})...")
        t0 = time.time()

        # 参考demo.py: output_device=None (保持结果在GPU上)
        output_device = None

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            if mode == "streaming":
                predictions = model.inference_streaming(
                    images,
                    num_scale_frames=num_scale_frames,
                    keyframe_interval=keyframe_interval,
                    output_device=output_device,
                )
            else:
                predictions = model.inference_windowed(
                    images,
                    window_size=64,
                    overlap_size=16,
                    num_scale_frames=num_scale_frames,
                    keyframe_interval=keyframe_interval,
                    output_device=output_device
                )

        print(f"Inference done in {time.time() - t0:.1f}s")
        if torch.cuda.is_available():
            print(f"GPU peak during inference: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

        # 后处理（参考demo.py: 不删除images，直接使用）
        predictions, images_cpu = postprocess(predictions, images)

        # 保存点云
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving point clouds to {output_dir}...")

        frame_points = []
        frame_colors = []
        frame_files = []

        # 参考demo.py的点云提取逻辑
        # predictions['images'] 在 postprocess 中已被删除，使用 images_cpu
        if images_cpu is not None:
            num_frames_result = images_cpu.shape[0]
            print(f"Processing {num_frames_result} frames for point cloud extraction")
            
            if 'depth' in predictions and 'intrinsic' in predictions:
                print("Reconstructing 3D points from depth map and transforming to world coords...")
                intrinsic = predictions['intrinsic']
                
                for i in range(num_frames_result):
                    depth = predictions['depth'][i].cpu().numpy()
                    img = images_cpu[i].cpu().numpy().transpose(1, 2, 0)
                    img = (img * 255).astype(np.uint8)
                    h, w = img.shape[:2]
                    
                    if depth.ndim == 3:
                        depth = depth[0] if depth.shape[0] == 1 else depth.squeeze()
                    
                    u_coords = np.arange(w)
                    v_coords = np.arange(h)
                    u_grid, v_grid = np.meshgrid(u_coords, v_coords)
                    
                    fx = float(intrinsic[i, 0, 0]) if intrinsic.ndim == 3 else float(intrinsic[0, 0])
                    fy = float(intrinsic[i, 1, 1]) if intrinsic.ndim == 3 else float(intrinsic[1, 1])
                    cx = float(intrinsic[i, 0, 2]) if intrinsic.ndim == 3 else float(intrinsic[0, 2])
                    cy = float(intrinsic[i, 1, 2]) if intrinsic.ndim == 3 else float(intrinsic[1, 2])
                    
                    valid_mask = depth > 0
                    
                    z = depth[valid_mask]
                    u_valid = u_grid[valid_mask]
                    v_valid = v_grid[valid_mask]
                    
                    x = (u_valid - cx) * z / fx
                    y = (v_valid - cy) * z / fy
                    
                    points_cam = np.stack([x, y, z], axis=1)
                    
                    # ── Transform to world coords ──
                    # postprocess 后 extrinsic 是 w2c，需 invert 到 c2w（与 cameras.json 一致）
                    ext_i = predictions["extrinsic"][i].numpy()
                    ext_4x4 = np.eye(4, dtype=np.float64)
                    ext_4x4[:3, :3] = ext_i[:3, :3]
                    ext_4x4[:3, 3] = ext_i[:3, 3]
                    c2w = np.linalg.inv(ext_4x4)
                    R_c2w = c2w[:3, :3].astype(np.float32)
                    t_c2w = c2w[:3, 3].astype(np.float32)
                    
                    points_world = points_cam @ R_c2w.T + t_c2w  # (N, 3) 世界坐标
                    
                    colors_3d = img[v_valid, u_valid]
                    
                    frame_ply_path = os.path.join(output_dir, f"frame_{i:03d}.ply")
                    write_ply(frame_ply_path, points_world, colors_3d)
                    
                    frame_points.append(points_world.tolist())
                    frame_colors.append(colors_3d.tolist())
                    frame_files.append(frame_ply_path)
                    
                    print(f"Frame {i}: saved {len(points_world)} points (world coords) to {frame_ply_path}")
                    
                print(f"Total frames: {num_frames_result}, total points: {sum(len(p) for p in frame_points)}")
            else:
                raise ValueError(f"depth+intrinsic not found in predictions. Available keys: {list(predictions.keys())}")
        else:
            print("Warning: No images available for point cloud coloring")

        if frame_points:
            # 同时保存合并的点云（可选）
            all_points = []
            all_colors = []
            for points, colors in zip(frame_points, frame_colors):
                all_points.extend(points)
                all_colors.extend(colors)
            
            merged_ply_path = os.path.join(output_dir, "point_cloud_merged.ply")
            write_ply(merged_ply_path, all_points, all_colors)
            print(f"Saved merged point cloud with {len(all_points)} points to {merged_ply_path}")

            # 保存额外数据到磁盘（兼容 see/ 格式）
            saved_files = {
                "point_clouds": frame_files,
                "merged_ply": merged_ply_path
            }

            # ── 保存 cameras.json（c2w 格式，相机真实位姿） ──
            if 'extrinsic' in predictions and 'intrinsic' in predictions:
                extrinsic = predictions['extrinsic']  # postprocess 后是 w2c 格式
                intrinsic = predictions['intrinsic']
                S_cam = extrinsic.shape[0]
                
                if images_cpu is not None:
                    H_img, W_img = images_cpu.shape[-2], images_cpu.shape[-1]
                else:
                    H_img, W_img = 480, 640
                
                # 将 w2c 逆转为 c2w，用于 cameras.json 和场景中心计算
                extrinsic_4x4 = torch.zeros((S_cam, 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
                extrinsic_4x4[:, :3, :4] = extrinsic
                extrinsic_4x4[:, 3, 3] = 1.0
                extrinsic_c2w = closed_form_inverse_se3_general(extrinsic_4x4)[:, :3, :4]
                
                cam_list = []
                t_c2w_list = []
                for i in range(S_cam):
                    R_c2w = extrinsic_c2w[i, :3, :3].tolist()
                    t_c2w = extrinsic_c2w[i, :3, 3].tolist()
                    t_c2w_list.append(t_c2w)
                    cam_list.append({
                        "focal": [float(intrinsic[i, 0, 0]), float(intrinsic[i, 1, 1])],
                        "pp": [float(intrinsic[i, 0, 2]), float(intrinsic[i, 1, 2])],
                        "R_c2w": R_c2w,
                        "t_c2w": t_c2w,
                        "image_w": W_img,
                        "image_h": H_img,
                    })
                
                cameras_json_path = os.path.join(output_dir, "cameras.json")
                save_json(cameras_json_path, cam_list)
                saved_files["cameras"] = cameras_json_path
                
                # ── 保存 metadata.json（scene_center 从 t_c2w 计算） ──
                cam_positions = np.array(t_c2w_list)
                scene_center = cam_positions.mean(axis=0).tolist()
                scene_scale = float(np.linalg.norm(np.ptp(cam_positions, axis=0))) if S_cam > 1 else 1.0
                
                metadata = {
                    "num_frames": S_cam,
                    "image_width": W_img,
                    "image_height": H_img,
                    "scene_center": scene_center,
                    "scene_scale": max(scene_scale, 0.1),
                    "default_conf_threshold": 0.5,
                    "default_downsample": 2,
                    "world_coords_saved": True,  # ✅ 点云已变换到世界坐标
                    "has_world_points": 'world_points' in predictions,
                    "has_confidence": 'world_points_conf' in predictions,
                }
                metadata_path = os.path.join(output_dir, "metadata.json")
                save_json(metadata_path, metadata)
                saved_files["metadata"] = metadata_path
                
                # 保存 extrinsic/intrinsic（保留兼容）
                extrinsic_path = os.path.join(output_dir, "extrinsic.npy")
                save_npy(extrinsic_path, extrinsic)
                saved_files["extrinsic"] = extrinsic_path
                
                intrinsic_path = os.path.join(output_dir, "intrinsic.npy")
                save_npy(intrinsic_path, intrinsic)
                saved_files["intrinsic"] = intrinsic_path
            
            # 保存深度图和置信度 (depth / depth_conf)
            if 'depth' in predictions:
                depth_dir = os.path.join(output_dir, "depth")
                os.makedirs(depth_dir, exist_ok=True)
                depth_files = []
                depth = predictions['depth']
                for i in range(min(num_frames_result, depth.shape[0])):
                    depth_frame = depth[i].cpu().numpy() if hasattr(depth[i], 'cpu') else depth[i]
                    # 去除多余维度
                    if depth_frame.ndim > 2:
                        depth_frame = depth_frame.squeeze()
                    depth_path = os.path.join(depth_dir, f"depth_{i:03d}.npy")
                    save_npy(depth_path, depth_frame)
                    depth_files.append(depth_path)
                saved_files["depth"] = depth_files
            
            if 'depth_conf' in predictions:
                depth_conf_dir = os.path.join(output_dir, "depth_conf")
                os.makedirs(depth_conf_dir, exist_ok=True)
                depth_conf_files = []
                depth_conf = predictions['depth_conf']
                for i in range(min(num_frames_result, depth_conf.shape[0])):
                    conf_frame = depth_conf[i].cpu().numpy() if hasattr(depth_conf[i], 'cpu') else depth_conf[i]
                    if conf_frame.ndim > 2:
                        conf_frame = conf_frame.squeeze()
                    conf_path = os.path.join(depth_conf_dir, f"depth_conf_{i:03d}.npy")
                    save_npy(conf_path, conf_frame)
                    depth_conf_files.append(conf_path)
                saved_files["depth_conf"] = depth_conf_files
            
            # 保存 world_points_conf
            if 'world_points_conf' in predictions:
                world_conf_dir = os.path.join(output_dir, "world_points_conf")
                os.makedirs(world_conf_dir, exist_ok=True)
                world_conf_files = []
                world_conf = predictions['world_points_conf']
                for i in range(min(num_frames_result, world_conf.shape[0])):
                    conf_frame = world_conf[i].cpu().numpy() if hasattr(world_conf[i], 'cpu') else world_conf[i]
                    if conf_frame.ndim > 1:
                        conf_frame = conf_frame.squeeze()
                    conf_path = os.path.join(world_conf_dir, f"world_conf_{i:03d}.npy")
                    save_npy(conf_path, conf_frame)
                    world_conf_files.append(conf_path)
                saved_files["world_points_conf"] = world_conf_files
            
            # ── 保存 merged_confidence.bin（与 see/ 一致） ──
            if 'world_points_conf' in predictions:
                all_conf = []
                world_conf = predictions['world_points_conf']
                for i in range(min(num_frames_result, world_conf.shape[0])):
                    c = world_conf[i].cpu().numpy() if hasattr(world_conf[i], 'cpu') else world_conf[i]
                    all_conf.append(c.reshape(-1))
                merged_conf = np.concatenate(all_conf).astype(np.float32)
                merged_conf_path = os.path.join(output_dir, "merged_confidence.bin")
                merged_conf.tofile(merged_conf_path)
                saved_files["merged_confidence"] = merged_conf_path
            
            # 保存原始图像（与 see/ 一致：frame_{i:06d}.jpg）
            if images_cpu is not None:
                images_dir = os.path.join(output_dir, "images")
                os.makedirs(images_dir, exist_ok=True)
                image_files = []
                for i in range(min(num_frames_result, images_cpu.shape[0])):
                    img = images_cpu[i].cpu().numpy() if hasattr(images_cpu[i], 'cpu') else images_cpu[i]
                    img = (img * 255).astype(np.uint8).transpose(1, 2, 0)
                    import cv2
                    img_path = os.path.join(images_dir, f"frame_{i:06d}.jpg")
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(img_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    image_files.append(img_path)
                saved_files["images"] = image_files

            # 保存文件清单
            manifest_path = os.path.join(output_dir, "manifest.json")
            save_json(manifest_path, saved_files)

            # ✅ 调用 web_export 补充输出（cameras.json / metadata.json / JPEG / viewer.html）
            try:
                from lingbot_map.vis.web_export import export_web_data
                export_web_data(
                    predictions=predictions,
                    images_cpu=images_cpu,
                    output_dir=output_dir,
                )
                print("web_export done")
            except Exception as e:
                print(f"web_export failed (non-fatal): {e}")

            return {
                "success": True,
                "frame_points": frame_points,
                "frame_colors": frame_colors,
                "frame_files": frame_files,
                "num_frames": len(frame_points),
                "total_points": len(all_points),
                "merged_ply_path": merged_ply_path,
                "output_dir": output_dir,
                "saved_files": saved_files
            }
        else:
            return {"success": False, "error": "No point cloud data generated"}

    except Exception as e:
        print(f"❌ 推理失败: {str(e)}")
        traceback.print_exc()
        import sys
        sys.stdout.flush()
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}