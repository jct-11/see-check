"""
LingBot-MAP Real-time Streaming Demo
完全按照 demo.py 的逻辑实现实时推理和点云生成

实现功能：
- 3D点云生成（带颜色和置信度过滤）
- 相机位姿估计
- HTTP API 接口支持图片上传和点云生成
"""

import argparse
import cv2
import numpy as np
import os
import torch
import time
import queue
import threading
import uuid

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3, closed_form_inverse_se3_general, unproject_depth_map_to_point_map
from lingbot_map.utils.load_fn import preprocess_image

# FastAPI dependencies for HTTP API
try:
    from fastapi import FastAPI, UploadFile, File, Response, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# 流式处理状态（用于保持 KV 缓存）
streaming_state = {
    'model': None,
    'device': None,
    'dtype': None,
    'image_size': 518,
    'patch_size': 14,
    'frame_count': 0,
    'last_ply_filename': None,
    'frame_queue': None,
    'is_running': False
}


# =============================================================================
# Post-processing (完全复制 demo.py)
# =============================================================================

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
    """Drop the leading batch dimension for single-sequence outputs."""
    batched_ndim = _BATCHED_NDIMS.get(key)
    if batched_ndim is None or not hasattr(value, "ndim"):
        return value
    if value.ndim == batched_ndim and value.shape[0] == 1:
        return value[0]
    return value


def postprocess(predictions, images):
    """Convert pose encoding to extrinsics (c2w) and move to CPU."""
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    # Convert w2c to c2w
    extrinsic_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
    extrinsic_4x4[..., :3, :4] = extrinsic
    extrinsic_4x4[..., 3, 3] = 1.0
    extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
    extrinsic = extrinsic_4x4[..., :3, :4]

    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions.pop("pose_enc_list", None)

    for k in list(predictions.keys()):
        if isinstance(predictions[k], torch.Tensor):
            predictions[k] = _squeeze_single_batch(
                k, predictions[k].to("cpu", non_blocking=True)
            )
    images_cpu = images.to("cpu", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return predictions, images_cpu


def prepare_for_visualization(predictions, images=None):
    """Convert predictions to the unbatched NumPy format used by vis code."""
    vis_predictions = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            v = _squeeze_single_batch(k, v.detach().cpu())
            vis_predictions[k] = v.numpy()
        elif isinstance(v, np.ndarray):
            vis_predictions[k] = _squeeze_single_batch(k, v)
        else:
            vis_predictions[k] = v

    if images is None:
        images = predictions.get("images")

    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    if isinstance(images, np.ndarray):
        images = _squeeze_single_batch("images", images)
    elif isinstance(images, torch.Tensor):
        images = _squeeze_single_batch("images", images).numpy()

    if isinstance(images, torch.Tensor):
        images = images.numpy()

    if images is not None:
        vis_predictions["images"] = images

    return vis_predictions


# =============================================================================
# Real-time Viewer (模拟 PointCloudViewer 的效果)
# =============================================================================

# =============================================================================
# Real-time processing
# =============================================================================

class FrameBuffer:
    """Thread-safe frame buffer"""
    
    def __init__(self, max_size=64):
        self.buffer = queue.Queue(maxsize=max_size)
    
    def put(self, frame):
        try:
            self.buffer.put(frame, block=False)
            return True
        except queue.Full:
            return False
    
    def get(self, block=True, timeout=None):
        try:
            return self.buffer.get(block=block, timeout=timeout)
        except queue.Empty:
            return None


# =============================================================================
# Frontend Stream Processing (复用流式推理逻辑)
# =============================================================================

def process_frontend_stream(model, device, dtype, frame_queue, result_queue, 
                            image_size=518, patch_size=14, num_scale_frames=10, 
                            keyframe_interval=2, kv_cache_sliding_window=100):
    """
    处理前端流式输入的帧，复用 process_camera_stream 的核心逻辑
    :param model: LingBot-MAP 模型
    :param device: 设备 (cuda/cpu)
    :param dtype: 数据类型
    :param frame_queue: 前端帧队列
    :param result_queue: 结果队列（用于返回点云数据）
    :param image_size: 图像尺寸
    :param patch_size: patch 尺寸
    :param num_scale_frames: 缩放帧数
    :param keyframe_interval: 关键帧间隔
    :param kv_cache_sliding_window: KV缓存滑动窗口大小
    """
    print("🔄 启动前端流式处理...")
    
    # Collect initial scale frames
    scale_frames = []
    scale_images = []
    
    print(f"收集 {num_scale_frames} 个缩放帧...")
    while len(scale_frames) < num_scale_frames:
        try:
            frame = frame_queue.get(block=True, timeout=30.0)
            if frame is None:
                print("收到终止信号")
                return
        except queue.Empty:
            print("等待缩放帧超时")
            return
        
        img_tensor = preprocess_image(frame, mode="crop", 
                                     image_size=image_size, 
                                     patch_size=patch_size)
        scale_frames.append(img_tensor)
        scale_images.append(frame)
        print(f"  已收集 {len(scale_frames)}/{num_scale_frames}")
    
    # Stack scale frames: convert to [B, S, C, H, W] format
    scale_batch = torch.stack(scale_frames, dim=0).unsqueeze(0).to(device)
    
    # Process scale frames using model.inference_streaming
    model.clean_kv_cache()
    
    print("处理缩放帧...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        predictions = model.inference_streaming(
            scale_batch,
            num_scale_frames=num_scale_frames,
            keyframe_interval=keyframe_interval,
        )
    
    # Post-process
    images_for_post = torch.stack([
        torch.from_numpy(cv2.resize(f, (image_size, image_size))).permute(2, 0, 1)
        for f in scale_images
    ]).float() / 255.0
    
    predictions, images_cpu = postprocess(predictions, images_for_post)
    vis_predictions = prepare_for_visualization(predictions, images_cpu)
    
    # Extract data
    world_points = vis_predictions.get("world_points", vis_predictions.get("depth"))
    depth_conf = vis_predictions.get("depth_conf")
    extrinsic = vis_predictions["extrinsic"]
    intrinsic = vis_predictions["intrinsic"]
    
    # Unproject if depth map
    if world_points.ndim == 4 and world_points.shape[-1] == 1:
        world_points = unproject_depth_map_to_point_map(
            torch.from_numpy(world_points),
            torch.from_numpy(extrinsic),
            torch.from_numpy(intrinsic)
        )
    
    # Send scale frame results
    for i in range(num_scale_frames):
        frame_points = world_points[i].reshape(-1, 3)
        valid = np.isfinite(frame_points).all(axis=1)
        pred_pts = frame_points[valid]
        
        # 提取对应点的颜色（使用原始图像颜色）
        frame_color = images_cpu[i].transpose(1, 2, 0)  # (H, W, 3)
        frame_color = cv2.resize(frame_color, (world_points.shape[2], world_points.shape[1]))
        pc_color = frame_color.reshape(-1, 3)[valid]
        
        if depth_conf is not None:
            conf = depth_conf[i]
            conf_flat = conf.reshape(-1)[valid]
            mask = conf_flat > 1.5
            pred_pts = pred_pts[mask]
            pc_color = pc_color[mask]
        
        max_points = 5000  # 增加点数以匹配viser效果
        if len(pred_pts) > max_points:
            indices = np.linspace(0, len(pred_pts) - 1, max_points, dtype=int)
            pred_pts = pred_pts[indices]
            pc_color = pc_color[indices]
        
        points = pred_pts.flatten().tolist()
        colors = (pc_color * 255).flatten().tolist()  # 转换为0-255范围
        
        ply_filename = f"frontend_scale_{i}.ply"
        save_point_cloud_ply_with_color(points, colors, ply_filename)
        
        result_queue.put({
            'type': 'result',
            'frame_id': f'scale_{i}',
            'point_count': len(points) // 3,
            'point_cloud': points,
            'point_colors': colors,
            'point_cloud_url': f'/point_cloud/{ply_filename}',
            'camera_pose': {
                'x': float(extrinsic[i][0, 3]),
                'y': float(extrinsic[i][1, 3]),
                'z': float(extrinsic[i][2, 3])
            }
        })
    
    print(f"缩放帧处理完成，发送 {num_scale_frames} 个结果")
    
    # ========== Periodic batch processing with KV cache reset ==========
    batch_size = 200
    overlap_size = 100
    new_frames_per_batch = batch_size - overlap_size
    
    print(f"开始周期性批处理 (batch={batch_size}, overlap={overlap_size})...")
    
    sliding_window = []
    batch_idx = 0
    frame_count = num_scale_frames
    
    while True:
        # Collect frames for this batch
        needed_frames = batch_size if batch_idx == 0 else batch_size - overlap_size
        
        while len(sliding_window) < batch_size:
            try:
                frame = frame_queue.get(block=True, timeout=10.0)
                if frame is None:
                    print("收到终止信号，退出批处理")
                    return
                sliding_window.append(frame)
                frame_count += 1
            except queue.Empty:
                if len(sliding_window) > 0:
                    break
                continue
        
        if len(sliding_window) < batch_size and batch_idx == 0:
            continue
        
        print(f"\n=== 批处理 {batch_idx + 1} ===")
        print(f"处理 {len(sliding_window)} 帧...")
        
        # Clean KV cache before each batch
        model.clean_kv_cache()
        torch.cuda.empty_cache()
        
        # Preprocess the batch
        batch_tensors = []
        batch_resized = []
        
        for frame in sliding_window:
            img_tensor = preprocess_image(frame, mode="crop",
                                        image_size=image_size,
                                        patch_size=patch_size)
            img_tensor = img_tensor.to(device)
            batch_tensors.append(img_tensor)
            batch_resized.append(cv2.resize(frame, (image_size, image_size)))
        
        batch_tensor = torch.stack(batch_tensors, dim=0).unsqueeze(0)
        del batch_tensors
        
        # Process batch with streaming inference
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            batch_output = model.inference_streaming(
                batch_tensor,
                num_scale_frames=0,
                keyframe_interval=keyframe_interval,
            )
        
        # Post-process batch
        images_for_post = torch.stack([
            torch.from_numpy(f).permute(2, 0, 1) for f in batch_resized
        ]).float() / 255.0
        
        predictions, images_cpu = postprocess(batch_output, images_for_post)
        vis_predictions = prepare_for_visualization(predictions, images_cpu)
        
        world_points = vis_predictions.get("world_points", vis_predictions.get("depth"))
        depth_conf = vis_predictions.get("depth_conf")
        extrinsic = vis_predictions["extrinsic"]
        intrinsic = vis_predictions["intrinsic"]
        
        if world_points.ndim == 4 and world_points.shape[-1] == 1:
            world_points = unproject_depth_map_to_point_map(
                torch.from_numpy(world_points),
                torch.from_numpy(extrinsic),
                torch.from_numpy(intrinsic)
            )
        
        # Send results for new frames (skip overlap)
        start_idx = overlap_size if batch_idx > 0 else 0
        for i in range(start_idx, len(sliding_window)):
            frame_points = world_points[i].reshape(-1, 3)
            valid = np.isfinite(frame_points).all(axis=1)
            pred_pts = frame_points[valid]
            
            if depth_conf is not None:
                conf = depth_conf[i]
                conf_flat = conf.reshape(-1)[valid]
                mask = conf_flat > 1.5
                pred_pts = pred_pts[mask]
            
            max_points = 200
            if len(pred_pts) > max_points:
                indices = np.linspace(0, len(pred_pts) - 1, max_points, dtype=int)
                pred_pts = pred_pts[indices]
            
            points = pred_pts.flatten().tolist()
            ply_filename = f"frontend_batch{batch_idx}_{i}.ply"
            save_point_cloud_ply(points, ply_filename)
            
            result_queue.put({
                'type': 'result',
                'frame_id': frame_count - (len(sliding_window) - i),
                'point_count': len(points) // 3,
                'point_cloud': points,
                'point_cloud_url': f'/point_cloud/{ply_filename}',
                'camera_pose': {
                    'x': float(extrinsic[i][0, 3]),
                    'y': float(extrinsic[i][1, 3]),
                    'z': float(extrinsic[i][2, 3])
                }
            })
        
        # Slide the window
        if batch_idx > 0:
            sliding_window = sliding_window[-overlap_size:]
        
        batch_idx += 1
        print(f"批处理 {batch_idx} 完成")


# =============================================================================
# HTTP API Service (for image upload and point cloud generation)
# =============================================================================

def save_point_cloud_ply(points, filename):
    """将点云保存为 PLY 格式文件（无颜色）"""
    ply_path = f"point_cloud_output/{filename}"
    num_points = len(points) // 3
    
    with open(ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(num_points):
            x = points[i * 3]
            y = points[i * 3 + 1]
            z = points[i * 3 + 2]
            r = int(((i / num_points) * 255))
            g = int((((num_points - i) / num_points) * 255))
            b = 128

def save_point_cloud_ply_with_color(points, colors, filename):
    """将带颜色的点云保存为 PLY 格式文件（与viser一致）"""
    ply_path = f"point_cloud_output/{filename}"
    num_points = len(points) // 3
    
    with open(ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(num_points):
            x = points[i * 3]
            y = points[i * 3 + 1]
            z = points[i * 3 + 2]
            r = int(colors[i * 3])
            g = int(colors[i * 3 + 1])
            b = int(colors[i * 3 + 2])
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
    
    return ply_path

def start_http_server(model, device, dtype, image_size, patch_size, http_port=8000):
    """启动 HTTP API 服务（包含流式 WebSocket 接口）"""
    if not HAS_FASTAPI:
        raise RuntimeError("FastAPI not installed. Please install with: pip install fastapi uvicorn")
    
    os.makedirs("upload", exist_ok=True)
    os.makedirs("point_cloud_output", exist_ok=True)
    
    # 更新全局流式状态
    streaming_state['model'] = model
    streaming_state['device'] = device
    streaming_state['dtype'] = dtype
    streaming_state['image_size'] = image_size
    streaming_state['patch_size'] = patch_size
    
    app = FastAPI()
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.mount("/point_cloud", StaticFiles(directory="point_cloud_output"), name="point_cloud")
    
    @app.get("/test/info")
    def info():
        return {"status": "running", "model_loaded": model is not None}
    
    @app.get("/api/latest-frame")
    async def latest_frame():
        """返回最新的模拟帧（用于服务器摄像头模式）"""
        # 创建一个简单的模拟图像（1x1像素的PNG）
        png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9c\x63\x00\x01\x00\x00\x05\x00\x01\r\n-\x8a\x00\x00\x00\x00IEND\xaeB`\x82'
        return Response(content=png_data, media_type="image/png")
    
    @app.post("/upload")
    async def upload(file: UploadFile = File(...)):
        try:
            if model is None:
                return {"code": 500, "msg": "模型未加载"}
            
            file_ext = file.filename.split(".")[-1].lower() if "." in file.filename else "jpg"
            task_id = str(uuid.uuid4())
            image_save_path = f"upload/{task_id}.{file_ext}"
            ply_filename = f"{task_id}.ply"
            
            with open(image_save_path, "wb") as f:
                f.write(await file.read())
            
            print(f"🔹 正在处理图片: {image_save_path}")
            
            frame = cv2.imread(image_save_path)
            if frame is None:
                return {"code": 500, "msg": "无法读取图片文件"}
            
            img_tensor = preprocess_image(frame, mode="crop", image_size=image_size, patch_size=patch_size)
            img_tensor = img_tensor.to(device).unsqueeze(0)
            
            images_for_post = torch.stack([
                torch.from_numpy(cv2.resize(frame, (image_size, image_size))).permute(2, 0, 1)
            ]).float() / 255.0
            
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                result = model.inference_streaming(img_tensor)
            
            predictions, images_cpu = postprocess(result, images_for_post)
            vis_predictions = prepare_for_visualization(predictions, images_cpu)
            
            world_points = vis_predictions.get("world_points", vis_predictions.get("depth"))
            depth_conf = vis_predictions.get("depth_conf")
            
            if world_points is None:
                return {"code": 500, "msg": "模型未返回点云数据"}
            
            print(f"   原始点云形状: {world_points.shape}")
            
            if world_points.ndim == 4 and world_points.shape[-1] == 1:
                print("   检测到深度图格式，进行反投影...")
                extrinsic = vis_predictions["extrinsic"]
                intrinsic = vis_predictions["intrinsic"]
                world_points = unproject_depth_map_to_point_map(
                    torch.from_numpy(world_points),
                    torch.from_numpy(extrinsic),
                    torch.from_numpy(intrinsic)
                )
                print(f"   反投影后点云形状: {world_points.shape}")
            
            frame_points = world_points[0] if world_points.ndim > 3 else world_points
            pred_pts = frame_points.reshape(-1, 3)
            
            valid = np.isfinite(pred_pts).all(axis=1)
            pred_pts = pred_pts[valid]
            print(f"   过滤后有效点数量: {len(pred_pts)}")
            
            if depth_conf is not None:
                conf = depth_conf[0] if depth_conf.ndim > 3 else depth_conf
                conf_flat = conf.reshape(-1)[valid]
                mask = conf_flat > 1.5
                pred_pts = pred_pts[mask]
                print(f"   置信度过滤后点数量: {len(pred_pts)}")
            
            max_points = 200
            if len(pred_pts) > max_points:
                indices = np.linspace(0, len(pred_pts) - 1, max_points, dtype=int)
                pred_pts = pred_pts[indices]
            
            if len(pred_pts) == 0:
                return {"code": 500, "msg": "生成的点云为空"}
            
            points = pred_pts.flatten().tolist()
            save_point_cloud_ply(points, ply_filename)
            point_cloud_url = f"http://localhost:{http_port}/point_cloud/{ply_filename}"
            
            print(f"✅ 生成 {len(points)//3} 个点云")
            print(f"✅ 点云文件已保存: {ply_filename}")
            
            camera_pose = None
            if 'extrinsic' in vis_predictions:
                extrinsic = vis_predictions["extrinsic"]
                if extrinsic.ndim >= 2:
                    frame_extrinsic = extrinsic[0] if extrinsic.ndim > 2 else extrinsic
                    camera_position = frame_extrinsic[:3, 3].tolist()
                    camera_pose = {
                        "x": camera_position[0],
                        "y": camera_position[1],
                        "z": camera_position[2]
                    }
            
            return {
                "code": 200,
                "msg": "上传成功，实时点云已生成",
                "task_id": task_id,
                "point_cloud": points,
                "point_cloud_url": point_cloud_url,
                "point_count": len(points) // 3,
                "has_confidence": depth_conf is not None,
                "camera_pose": camera_pose
            }
        
        except Exception as e:
            print(f"❌ 处理失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {"code": 500, "msg": f"处理失败: {str(e)}"}
    
    @app.post("/batch-inference")
    async def batch_inference(request: Request):
        """HTTP 批量推理接口 - 接收base64图片数组，返回所有点云结果"""
        try:
            if model is None:
                return {"code": 500, "msg": "模型未加载"}
            
            body = await request.json()
            frames_base64 = body.get('frames', [])
            batch_id = body.get('batch_id', 0)
            
            if not frames_base64:
                return {"code": 400, "msg": "没有帧数据"}
            
            num_frames = len(frames_base64)
            print(f"🚀 开始批量处理 {num_frames} 帧, batch_id={batch_id}")
            
            # 检查前几帧的大小
            if num_frames > 0:
                for i in range(min(3, num_frames)):
                    frame_len = len(frames_base64[i])
                    print(f"   帧 #{i} Base64长度: {frame_len} 字符")
                    if frame_len < 100:
                        print(f"   ⚠️ 帧 #{i} 可能是空帧或数据不完整")
            
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            # ========== 第一步：解码所有帧并预处理 ==========
            print("📷 解码并预处理所有帧...")
            batch_tensors = []
            batch_resized = []  # 保存调整大小后的原始图像（用于后处理）
            preprocess_image_list = []  # 保存预处理后的张量，用于逐帧回退
            valid_frame_indices = []
            empty_frame_count = 0
            decode_fail_count = 0
            
            for i, frame_b64 in enumerate(frames_base64):
                try:
                    import base64
                    
                    # 检查Base64数据是否为空
                    if not frame_b64 or len(frame_b64) < 100:
                        print(f"⚠️ 帧 #{i} Base64数据为空或过短 (长度={len(frame_b64)})")
                        empty_frame_count += 1
                        continue
                    
                    frame_data = base64.b64decode(frame_b64.split(',')[1] if ',' in frame_b64 else frame_b64)
                    frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                    
                    if frame_array.size == 0:
                        print(f"⚠️ 帧 #{i} 解码后数组为空，跳过")
                        empty_frame_count += 1
                        continue
                    
                    if frame_array.size < 100:
                        print(f"⚠️ 帧 #{i} 数据量过小 ({frame_array.size} 字节)，可能是空图像")
                    
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                    if frame is None:
                        print(f"⚠️ 帧 #{i} 解码失败，跳过")
                        decode_fail_count += 1
                        continue
                    
                    # 检查解码后的图像尺寸
                    h, w = frame.shape[:2]
                    if h < 10 or w < 10:
                        print(f"⚠️ 帧 #{i} 图像尺寸过小 ({w}x{h})，可能是无效图像")
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 预处理图像
                    img_tensor = preprocess_image(frame_rgb, mode="crop",
                                                image_size=image_size,
                                                patch_size=patch_size)
                    batch_tensors.append(img_tensor)
                    preprocess_image_list.append(img_tensor.clone())  # 保存副本用于回退
                    
                    # 保存调整大小后的图像（用于后处理）
                    batch_resized.append(cv2.resize(frame_rgb, (image_size, image_size)))
                    
                    valid_frame_indices.append(i)
                    
                except Exception as e:
                    print(f"⚠️ 帧 #{i} 预处理失败: {type(e).__name__}: {e}")
                    continue
            
            if len(batch_tensors) == 0:
                return {"code": 400, "msg": "没有有效的帧数据"}
            
            print(f"✅ 成功预处理 {len(batch_tensors)} 帧")
            
            # ========== 第二步：真正的批量推理 ==========
            print("🔄 执行批量推理...")
            torch.cuda.empty_cache()
            
            # 堆叠成批量张量 [B, C, H, W] -> [1, B, C, H, W]
            batch_tensor = torch.stack(batch_tensors, dim=0).unsqueeze(0).to(device)
            del batch_tensors
            
            # ========== 详细调试信息 ==========
            print(f"\n📋 批量推理调试信息:")
            print(f"   输入张量形状: {batch_tensor.shape}")
            print(f"   输入张量设备: {batch_tensor.device}")
            print(f"   输入张量元素数: {batch_tensor.numel()}")
            print(f"   输入张量数据类型: {batch_tensor.dtype}")
            print(f"   输入统计: min={batch_tensor.min().item():.4f}, max={batch_tensor.max().item():.4f}, mean={batch_tensor.mean().item():.4f}")
            
            # 检查图像尺寸是否符合模型要求
            B, S, C, H, W = batch_tensor.shape
            print(f"   批量大小(B): {B}, 序列长度(S): {S}, 通道(C): {C}, 高度(H): {H}, 宽度(W): {W}")
            
            # 强制清理模型状态（确保与文件模式一致）
            print("\n🧹 强制清理模型状态...")
            if hasattr(model, 'kv_cache') and model.kv_cache is not None:
                model.kv_cache = None
            model.clean_kv_cache()
            torch.cuda.empty_cache()
            
            # ========== 执行推理 ==========
            try:
                print("\n🚀 开始推理...")
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                    batch_output = model.inference_streaming(
                        batch_tensor,
                        num_scale_frames=0,
                        keyframe_interval=10,
                    )
                
                print(f"✅ 批量推理完成")
                print(f"📤 输出类型: {type(batch_output)}")
                if isinstance(batch_output, dict):
                    print(f"📤 输出键: {list(batch_output.keys())}")
                    for k, v in batch_output.items():
                        if hasattr(v, 'shape'):
                            print(f"   - {k}: shape={v.shape}")
            
            except RuntimeError as e:
                print(f"\n❌ 批量推理失败: {e}")
                
                # 尝试不同的输入格式（去掉batch维度）
                print("🔄 尝试格式2: [B, C, H, W]（去掉外层batch维度）")
                try:
                    batch_tensor_2d = batch_tensor.squeeze(0)  # [B, C, H, W]
                    print(f"   新形状: {batch_tensor_2d.shape}")
                    
                    model.clean_kv_cache()
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                        batch_output = model.inference_streaming(
                            batch_tensor_2d,
                            num_scale_frames=0,
                            keyframe_interval=10,
                        )
                    print("✅ 格式2成功!")
                except Exception as e2:
                    print(f"❌ 格式2也失败: {e2}")
                    
                    # 尝试逐帧处理
                    print("🔄 尝试逐帧处理作为备选方案...")
                    batch_output = None
                    del batch_tensor
                    torch.cuda.empty_cache()
            
            # ========== 第三步：处理推理结果（与文件模式一致）==========
            print("📊 处理推理结果...")
            
            all_points = []
            all_colors = []
            all_camera_poses = []
            scene_center = None
            total_point_count = 0
            
            # 如果批量推理失败，回退到逐帧处理
            if batch_output is None:
                print("   使用逐帧处理模式")
                
                for i in range(len(valid_frame_indices)):
                    frame_idx = valid_frame_indices[i]
                    
                    try:
                        # 重新加载预处理后的图像张量
                        img_tensor = preprocess_image_list[i].to(device).unsqueeze(0)
                        
                        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                            single_output = model.inference_streaming(img_tensor)
                        
                        # 获取单帧结果
                        # 准备后处理图像
                        images_for_post = torch.stack([
                            torch.from_numpy(batch_resized[i]).permute(2, 0, 1)
                        ]).float() / 255.0
                        
                        predictions, images_cpu = postprocess(single_output, images_for_post)
                        vis_predictions = prepare_for_visualization(predictions, images_cpu)
                        
                        # 清理内存
                        del img_tensor, single_output
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"❌ 帧 #{frame_idx} 推理失败: {type(e).__name__}: {e}")
                        continue
            else:
                print("   使用批量处理模式")
                
                # 一次性准备所有后处理图像（与文件模式一致）
                images_for_post = torch.stack([
                    torch.from_numpy(f).permute(2, 0, 1) for f in batch_resized
                ]).float() / 255.0
                
                # 一次性后处理所有结果（与文件模式一致）
                predictions, images_cpu = postprocess(batch_output, images_for_post)
                vis_predictions = prepare_for_visualization(predictions, images_cpu)
                
                world_points = vis_predictions.get("world_points", vis_predictions.get("depth"))
                depth_conf = vis_predictions.get("depth_conf")
                extrinsic = vis_predictions["extrinsic"]
                intrinsic = vis_predictions["intrinsic"]
                
                # 如果是深度图，转换为点云
                if world_points.ndim == 4 and world_points.shape[-1] == 1:
                    world_points = unproject_depth_map_to_point_map(
                        torch.from_numpy(world_points),
                        torch.from_numpy(extrinsic),
                        torch.from_numpy(intrinsic)
                    )
                
                colors = images_cpu.permute(0, 2, 3, 1).cpu().numpy()
                
                # 遍历所有帧
                for i in range(len(valid_frame_indices)):
                    frame_idx = valid_frame_indices[i]
                    
                    try:
                        frame_points = world_points[i].reshape(-1, 3)
                        valid = np.isfinite(frame_points).all(axis=1)
                        points_filtered = frame_points[valid]
                        
                        # 添加调试信息
                        if i == 0:
                            print(f"   帧 #{i}: 原始点数={len(frame_points)}, 有效点数(有限值)={len(points_filtered)}")
                        
                        if depth_conf is not None:
                            conf = depth_conf[i]
                            conf_flat = conf.reshape(-1)[valid]
                            
                            # 添加调试信息
                            if i == 0:
                                print(f"   帧 #{i}: 置信度范围=[{conf_flat.min():.3f}, {conf_flat.max():.3f}], 均值={conf_flat.mean():.3f}")
                            
                            # 降低置信度阈值（从1.5改为0.1）
                            mask = conf_flat > 0.1
                            points_filtered = points_filtered[mask]
                            
                            if i == 0:
                                print(f"   帧 #{i}: 置信度过滤后点数={len(points_filtered)}")
                        else:
                            if i == 0:
                                print(f"   帧 #{i}: 没有置信度数据，跳过置信度过滤")
                        
                        # 获取颜色
                        colors_cropped = colors[i]
                        colors_all = (colors_cropped.reshape(-1, 3) * 255).astype(np.uint8)
                        colors_filtered = colors_all[valid]
                        if depth_conf is not None:
                            colors_filtered = colors_filtered[mask]
                        
                        if scene_center is None and len(points_filtered) > 0:
                            scene_center = np.mean(points_filtered, axis=0)
                        
                        max_points = 5000
                        if len(points_filtered) > max_points:
                            indices = np.linspace(0, len(points_filtered) - 1, max_points, dtype=int)
                            points_filtered = points_filtered[indices]
                            colors_filtered = colors_filtered[indices]
                        
                        all_points.extend(points_filtered.flatten().tolist())
                        all_colors.extend(colors_filtered.flatten().tolist())
                        total_point_count += len(points_filtered)
                        
                        # 获取相机位姿
                        if isinstance(extrinsic, torch.Tensor):
                            ext = extrinsic[i].cpu().numpy()
                        else:
                            ext = extrinsic[i]
                        
                        cam_to_world_mat = closed_form_inverse_se3(ext)
                        cam_to_world = cam_to_world_mat[:3, :]
                        if scene_center is not None:
                            cam_to_world[..., -1] -= scene_center
                        
                        all_camera_poses.append({
                            'position': [
                                float(cam_to_world[0, 3]),
                                float(cam_to_world[1, 3]),
                                float(cam_to_world[2, 3])
                            ],
                            'frame_id': frame_idx
                        })
                        
                        print(f"✅ 帧 #{frame_idx}: {len(points_filtered)} 点")
                        
                    except Exception as e:
                        print(f"❌ 帧 #{frame_idx} 处理失败: {type(e).__name__}: {e}")
                        continue
            
            del batch_output, images_for_post
            torch.cuda.empty_cache()
            gc.collect()
            
            # ========== 第四步：返回结果 ==========
            print(f"\n🎉 批量处理完成! 总点数: {total_point_count}")
            
            return {
                "code": 200,
                "success": True,
                "points": all_points,
                "colors": all_colors,
                "cameraPoses": all_camera_poses,
                "sceneCenter": scene_center.tolist() if scene_center is not None else [],
                "frames_processed": len(valid_frame_indices),
                "total_points": total_point_count,
                "batch_id": batch_id
            }
            
        except Exception as e:
            print(f"❌ 批量推理失败: {type(e).__name__}: {e}")
            return {"code": 500, "success": False, "msg": str(e)}
    
    import uvicorn
    print(f"🚀 HTTP API 服务启动在 http://localhost:{http_port}")
    print(f"   测试接口: GET /test/info")
    print(f"   上传接口: POST /upload")
    print(f"   批量接口: POST /batch-inference")
    print(f"   点云下载: GET /point_cloud/<filename>")
    uvicorn.run(app, host="0.0.0.0", port=http_port, 
                ws_ping_interval=None, ws_ping_timeout=None)

def main():
    parser = argparse.ArgumentParser(description="LingBot-MAP Real-time Streaming Demo")
    
    # Mode selection (for compatibility, only API mode is supported)
    parser.add_argument("--mode", type=str, default="api", choices=["api"], 
                        help="运行模式: api(HTTP接口)")
    
    # Input
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    
    # Output
    parser.add_argument("--http_port", type=int, default=8000, help="HTTP API port")
    
    # Inference
    parser.add_argument("--image_size", type=int, default=518, help="Image size")
    parser.add_argument("--patch_size", type=int, default=14, help="Patch size")
    parser.add_argument("--num_scale_frames", type=int, default=4, help="Number of scale frames")
    parser.add_argument("--keyframe_interval", type=int, default=10, help="Keyframe interval")
    parser.add_argument("--kv_cache_sliding_window", "--window_size", type=int, default=300, help="KV cache sliding window size")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("Loading model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    # Load model from local checkpoint
    model = GCTStream(img_size=args.image_size, patch_size=args.patch_size)
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    print(f"Model loaded. Inference dtype: {dtype}")
    
    # Run HTTP API mode
    print(f"🌐 启动 API 模式，HTTP 服务端口: {args.http_port}")
    start_http_server(model, device, dtype, args.image_size, args.patch_size, args.http_port)


if __name__ == "__main__":
    main()
