"""LingBot-MAP 流式批次管理服务（单用户架构）

参考 live_camera.py 和 stream.py 的设计：
- 模型常驻内存，不重复加载
- 维护KV缓存，逐帧推理
- 后台线程监控帧文件夹，自动处理新帧
- 点云数据缓存在内存中，API直接返回

文件结构：
data/
└── {batch_id}/
    ├── frames/
    │   ├── frame_000.jpg
    │   ├── frame_001.jpg
    │   └── ...
    ├── status.json
    └── logs.json
"""

import os
import json
import time
import glob
import asyncio
import subprocess
import traceback
import threading
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from collections import deque

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF

# 必须在导入torch之前设置
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general, unproject_depth_map_to_point_map
from lingbot_map.utils.load_fn import load_and_preprocess_images

app = FastAPI(title="LingBot-MAP Streaming Service", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置
DATA_DIR = Path("/home/sscy/lingbot-map/stmem-main/data")
MAX_FRAMES_PER_BATCH = 50000
SUPPORTED_FORMATS = {"image/jpeg", "image/png", "image/webp"}
MODEL_PATH = "/home/sscy/lingbot-map/lingbot-map-main/checkpoints/robbyant/lingbot-map/lingbot-map-long.pt"

# 推理参数
IMAGE_SIZE = 518
PATCH_SIZE = 14
NUM_SCALE_FRAMES = 8
# Keyframe interval: auto-selected based on frame count (same as live_camera.py).
# <= 320 frames: interval=1 (every frame is a keyframe, best accuracy).
# > 320 frames: interval=ceil(N/320) to keep KV cache at ~320 keyframes.
# The frontend can override this via the keyframe_interval parameter.
KEYFRAME_INTERVAL_DEFAULT = 1
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# 全局状态
model_state = {
    "model": None,
    "device": None,
    "initialized": False,
    "current_batch_id": None,
    "frame_idx": 0,
    "scale_frames": NUM_SCALE_FRAMES,
    "keyframe_interval": KEYFRAME_INTERVAL_DEFAULT,
    "max_images": None,
    "known_paths": set(),
    "all_predictions": {
        "pose_enc": [],
        "depth": [],
        "depth_conf": [],
        "world_points": [],
        "world_points_conf": [],
        "images": [],
    },
    "is_streaming": False,
    "stop_event": threading.Event(),
    "finish_requested": False,
}

# 点云过滤参数（与 live_camera.py 一致）
CONF_THRESHOLD = 0.7        # 置信度阈值
DOWNSAMPLE_FACTOR = 10     # 下采样倍数
MAX_CACHE_FRAMES = 300     # 内存缓存最大帧数

# 点云帧缓存
frame_cache = {}  # {frame_index: {"points": [...], "colors": [...], "confs": [...], "camera": {...}}}
cache_frame_order = []  # 维护缓存顺序，用于 FIFO 淘汰

def add_to_frame_cache(frame_idx, points, colors, confs, camera):
    """添加点云到缓存，超过 MAX_CACHE_FRAMES 时丢弃最旧帧"""
    global frame_cache, cache_frame_order
    
    frame_cache[frame_idx] = {
        "points": points.tolist(),
        "colors": colors.tolist(),
        "confs": confs.tolist(),
        "camera": camera,
    }
    cache_frame_order.append(frame_idx)
    
    # FIFO 淘汰，保持不超过 MAX_CACHE_FRAMES
    while len(cache_frame_order) > MAX_CACHE_FRAMES:
        oldest = cache_frame_order.pop(0)
        del frame_cache[oldest]

# 批次状态
batch_status = {
    "batch_id": None,
    "status": "idle",
    "total_frames": 0,
    "uploaded_frames": 0,
    "processed_frames": 0,
    "total_points": 0,
    "error_message": "",
    "dgsg_status": "idle",  # idle → building → done → error
}

# 日志
batch_logs = []

def write_log(message, log_type="info"):
    """写入日志"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "message": message,
        "type": log_type
    }
    batch_logs.append(log_entry)
    
    # 保持日志数量在合理范围
    if len(batch_logs) > 1000:
        batch_logs[:] = batch_logs[-500:]
    
    print(f"[{timestamp}] [{log_type}] {message}")

def update_status(**kwargs):
    """更新批次状态"""
    batch_status.update(kwargs)
    
    # 保存到文件
    if batch_status.get("batch_id"):
        status_file = DATA_DIR / batch_status["batch_id"] / "status.json"
        status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(status_file, 'w') as f:
            json.dump(batch_status, f, indent=2)

def load_single_image(image_path, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE):
    """加载并预处理单张图片"""
    img = Image.open(image_path)
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    img = img.convert("RGB")

    width, height = img.size
    new_width = image_size
    new_height = round(height * (new_width / width) / patch_size) * patch_size

    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    img = TF.ToTensor()(img)

    if new_height > image_size:
        start_y = (new_height - image_size) // 2
        img = img[:, start_y: start_y + image_size, :]

    return img

def load_model():
    """加载模型到GPU"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model to {device}...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU mem before load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    model = GCTStream(
        img_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        enable_3d_rope=True,
        enable_point=False,  # Depth-unprojected points (consistent with stream.py/live_camera.py)
        max_frame_num=1024,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=NUM_SCALE_FRAMES,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=False,
        camera_num_iterations=4,
    )

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    
    if isinstance(state_dict, list):
        state_dict = state_dict[0] if state_dict else {}
    elif hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU mem after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    write_log("模型加载成功", "ok")
    return model, device

def on_frame_callback(frame_idx, image_np, frame_output):
    """逐帧回调函数：处理推理结果并缓存点云原始数据（与 live_camera.py 一致）

    后处理仅做数据维度压缩和格式化，过滤全部移到前端 live_viewer.py 执行。
    """
    try:
        H, W = image_np.shape[:2]

        pose_enc = frame_output["pose_enc"]
        depth = frame_output.get("depth")
        depth_conf = frame_output.get("depth_conf")
        world_points = frame_output.get("world_points")

        if pose_enc is None or depth is None or depth_conf is None:
            return

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

        _pose_enc_batch = _pose_enc.unsqueeze(0).unsqueeze(0)
        extrinsic_c2w, intrinsic = pose_encoding_to_extri_intri(
            _pose_enc_batch, image_size_hw=(H, W)
        )
        extrinsic_c2w = extrinsic_c2w[0, 0]
        intrinsic = intrinsic[0, 0]

        intrinsic_np = intrinsic.cpu().numpy()
        _depth_np = _depth.cpu().numpy()
        _depth_conf_np = _depth_conf.cpu().numpy()

        if world_points is not None:
            if world_points.dim() == 5:
                _wp = world_points[0, 0].cpu().numpy()
            elif world_points.dim() == 4:
                _wp = world_points[0].cpu().numpy()
            else:
                _wp = world_points.cpu().numpy()
        else:
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

        extrinsic_c2w_np = extrinsic_c2w.cpu().numpy()

        pred_pts = _wp.reshape(-1, 3)
        color_flat = image_np.reshape(-1, 3) / 255.0
        conf_flat = _depth_conf_np.reshape(-1)

        extr_np = extrinsic_c2w_np
        intr_np = intrinsic_np

        camera = {
            "focal": [float(intr_np[0, 0]), float(intr_np[1, 1])],
            "pp": [float(intr_np[0, 2]), float(intr_np[1, 2])],
            "R_c2w": extr_np[:3, :3].tolist(),
            "t_c2w": extr_np[:3, 3].tolist(),
            "image_w": W,
            "image_h": H,
        }

        # ── 保存 depth、pose、intrinsics 到硬盘（供 dgsg 管线使用）──
        batch_id = model_state.get("current_batch_id")
        if batch_id:
            batch_dir = DATA_DIR / batch_id

            depth_dir = batch_dir / "depth"
            depth_dir.mkdir(exist_ok=True)
            poses_dir = batch_dir / "poses"
            poses_dir.mkdir(exist_ok=True)

            depth_shape = _depth_np.shape
            depth_h, depth_w = depth_shape[0], depth_shape[1]
            if _depth_np.ndim > 2:
                _depth_np = _depth_np.reshape(depth_h, depth_w)

            # 从第一帧原始图片获取原始分辨率（depth 需要和 rgb 同尺寸）
            orig_h, orig_w = H, W  # 默认用模型输出尺寸
            frames_dir_check = batch_dir / "frames"
            if frames_dir_check.exists():
                any_frame = next(frames_dir_check.glob("*.jpg"), None)
                if any_frame is None:
                    any_frame = next(frames_dir_check.glob("*.png"), None)
                if any_frame is not None:
                    from PIL import Image as _Img
                    with _Img.open(any_frame) as _img:
                        orig_w, orig_h = _img.size  # PIL: (width, height)

            # 如果深度图和原始 RGB 尺寸不同，需要 resize + 缩放 intrinsics
            save_depth = _depth_np
            save_intr = intr_np
            if depth_h != orig_h or depth_w != orig_w:
                scale_x = orig_w / depth_w
                scale_y = orig_h / depth_h
                save_intr = intr_np.copy()
                save_intr[0, 0] *= scale_x
                save_intr[1, 1] *= scale_y
                save_intr[0, 2] *= scale_x
                save_intr[1, 2] *= scale_y
                d_img = Image.fromarray((_depth_np * 1000).clip(0, 65535).astype(np.uint16))
                d_img = d_img.resize((orig_w, orig_h), Image.NEAREST)
                save_depth = np.array(d_img).astype(np.float32) / 1000.0

            depth_mm = (save_depth * 1000).clip(0, 65535).astype(np.uint16)
            Image.fromarray(depth_mm).save(depth_dir / f"frame_{frame_idx:06d}.png")

            c2w_4x4 = np.eye(4, dtype=np.float64)
            c2w_4x4[:3, :4] = extrinsic_c2w_np
            np.savetxt(poses_dir / f"frame_{frame_idx:06d}.txt", c2w_4x4, "%.15e")

            intr_path = batch_dir / "intrinsics.json"
            if not intr_path.exists():
                with open(intr_path, "w") as f:
                    json.dump({
                        "fx": float(save_intr[0, 0]),
                        "fy": float(save_intr[1, 1]),
                        "cx": float(save_intr[0, 2]),
                        "cy": float(save_intr[1, 2]),
                        "w": orig_w,
                        "h": orig_h,
                    }, f)

        add_to_frame_cache(frame_idx, pred_pts, color_flat, conf_flat, camera)

        processed = len(frame_cache)
        update_status(processed_frames=processed)

        if processed % 10 == 0:
            write_log(f"已处理 {processed} 帧", "info")

    except Exception as e:
        write_log(f"帧 {frame_idx} 处理失败: {e}", "err")
        traceback.print_exc()

def process_scale_frames(frames_dir, num_scale_frames=NUM_SCALE_FRAMES):
    """处理前N帧作为scale frames（Phase 1）"""
    write_log(f"Phase 1: 处理 {num_scale_frames} 个scale frames...", "info")
    
    # 获取所有帧路径
    exts = (".jpg", ".jpeg", ".png")
    all_paths = []
    for ext in exts:
        all_paths.extend(glob.glob(str(frames_dir / f"*{ext}")))
    all_paths = sorted(set(all_paths))
    
    if len(all_paths) < num_scale_frames:
        write_log(f"帧数不足，需要{num_scale_frames}帧，当前{len(all_paths)}帧", "err")
        return False
    
    # 取前N帧
    scale_paths = all_paths[:num_scale_frames]
    
    # 加载并预处理
    scale_images = []
    for path in scale_paths:
        img = load_single_image(path)
        scale_images.append(img)
    
    scale_tensor = torch.stack(scale_images, dim=0).unsqueeze(0).to(model_state["device"])
    
    # 清理KV缓存
    model_state["model"].clean_kv_cache()
    
    # Phase 1: Scale frames推理
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
        scale_output = model_state["model"].forward(
            scale_tensor,
            num_frame_for_scale=num_scale_frames,
            num_frame_per_block=num_scale_frames,
            causal_inference=True,
        )
    
    # 处理每一帧的输出
    for i in range(num_scale_frames):
        # 获取模型输出尺寸（与 frame_monitor_thread 保持一致）
        depth = scale_output.get("depth")
        if depth is not None and depth.dim() >= 3 and depth.shape[1] > i:
            d = depth[:, i:i + 1]
            while d.ndim > 2:
                d = d[0] if d.shape[0] == 1 else d.squeeze()
            model_h, model_w = d.shape
        else:
            model_h, model_w = IMAGE_SIZE, IMAGE_SIZE
        
        # 将预处理后的图片缩放到模型输出尺寸
        img_tensor = scale_images[i]
        img_resized = img_tensor.permute(1, 2, 0).numpy()
        img_pil = Image.fromarray((img_resized * 255).clip(0, 255).astype(np.uint8))
        img_pil = img_pil.resize((model_w, model_h), Image.Resampling.BICUBIC)
        img_np = np.array(img_pil)
        
        frame_output = {}
        for k, v in scale_output.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[1] > i:
                frame_output[k] = v[:, i:i + 1]
            else:
                frame_output[k] = v
        
        on_frame_callback(i, img_np, frame_output)
        model_state["known_paths"].add(scale_paths[i])
    
    model_state["frame_idx"] = num_scale_frames
    write_log(f"Phase 1 完成，已处理 {num_scale_frames} 帧", "ok")
    
    del scale_output, scale_tensor
    return True

def frame_monitor_thread():
    """后台线程：监控帧文件夹，处理新帧"""
    write_log("帧监控线程启动", "info")
    
    scale_processed = False
    
    while not model_state["stop_event"].is_set():
        try:
            if not model_state["is_streaming"]:
                scale_processed = False
                time.sleep(0.5)
                continue
            
            batch_id = model_state["current_batch_id"]
            if not batch_id:
                time.sleep(0.5)
                continue
            
            frames_dir = DATA_DIR / batch_id / "frames"
            if not frames_dir.exists():
                write_log(f"帧文件夹不存在: {frames_dir}", "info")
                time.sleep(0.5)
                continue
            
            # Phase 1: 处理scale frames（只执行一次）
            if not scale_processed:
                exts = (".jpg", ".jpeg", ".png")
                all_paths = []
                for ext in exts:
                    all_paths.extend(glob.glob(str(frames_dir / f"*{ext}")))
                all_paths = sorted(set(all_paths))
                
                write_log(f"Phase 1 - 扫描帧文件夹: {frames_dir}, 找到 {len(all_paths)} 帧, 需要 {NUM_SCALE_FRAMES} 帧", "info")
                
                if len(all_paths) >= NUM_SCALE_FRAMES:
                    write_log(f"Phase 1 - 开始处理 {NUM_SCALE_FRAMES} 个scale frames...", "info")
                    success = process_scale_frames(frames_dir, NUM_SCALE_FRAMES)
                    if success:
                        scale_processed = True
                        write_log(f"Scale frames处理完成，开始逐帧处理...", "ok")
                    else:
                        write_log(f"Scale frames处理失败，重试...", "err")
                        time.sleep(0.5)
                        continue
                else:
                    write_log(f"Phase 1 - 帧数不足: 需要 {NUM_SCALE_FRAMES} 帧，当前 {len(all_paths)} 帧", "info")
                    time.sleep(0.5)
                    continue
            
            # Phase 2/3: 处理后续帧
            exts = (".jpg", ".jpeg", ".png")
            current_paths = []
            for ext in exts:
                current_paths.extend(glob.glob(str(frames_dir / f"*{ext}")))
            current_paths = sorted(set(current_paths))
            
            write_log(f"Phase 2 - 当前帧总数: {len(current_paths)}, 已处理帧索引: {model_state['frame_idx']}, 已知路径数: {len(model_state['known_paths'])}", "info")
            
            new_paths = [p for p in current_paths if p not in model_state["known_paths"]]
            
            if new_paths:
                write_log(f"Phase 2 - 发现 {len(new_paths)} 个新帧: {[os.path.basename(p) for p in new_paths]}", "info")
                write_log(f"Phase 2 - 开始处理...", "info")
                
                for path in new_paths:
                    if model_state["stop_event"].is_set():
                        break
                    
                    if model_state["max_images"] is not None and model_state["frame_idx"] >= model_state["max_images"]:
                        write_log(f"达到最大图片数 {model_state['max_images']}，停止处理", "info")
                        model_state["is_streaming"] = False
                        update_status(status="completed", processed_frames=model_state["frame_idx"])
                        break
                    
                    frame_idx = model_state["frame_idx"]
                    
                    try:
                        img = load_single_image(path)
                    except Exception as e:
                        write_log(f"加载失败 {path}: {e}", "err")
                        model_state["known_paths"].add(path)
                        continue
                    
                    frame_image = img.unsqueeze(0).unsqueeze(0).to(
                        model_state["device"], non_blocking=True
                    )
                    
                    # 判断是否是关键帧
                    ki = model_state["keyframe_interval"]
                    is_keyframe = (ki <= 1) or \
                                  ((frame_idx - model_state["scale_frames"]) % ki == 0)
                    
                    if not is_keyframe:
                        model_state["model"]._set_skip_append(True)
                    
                    # 推理
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
                        frame_output = model_state["model"].forward(
                            frame_image,
                            num_frame_for_scale=model_state["scale_frames"],
                            num_frame_per_block=1,
                            causal_inference=True,
                        )
                    
                    if not is_keyframe:
                        model_state["model"]._set_skip_append(False)
                    
                    # 回调处理
                    # 获取模型输出尺寸
                    depth = frame_output.get("depth")
                    if depth is not None:
                        d = depth
                        while d.ndim > 2:
                            d = d[0] if d.shape[0] == 1 else d.squeeze()
                        model_h, model_w = d.shape
                    else:
                        model_h, model_w = IMAGE_SIZE, IMAGE_SIZE
                    
                    # 缩放颜色图到模型输出尺寸
                    img_resized = img.permute(1, 2, 0).numpy()
                    img_pil = Image.fromarray((img_resized * 255).clip(0, 255).astype(np.uint8))
                    img_pil = img_pil.resize((model_w, model_h), Image.Resampling.BICUBIC)
                    img_np = np.array(img_pil)
                    
                    on_frame_callback(frame_idx, img_np, frame_output)
                    
                    # 保存预测结果
                    for key in model_state["all_predictions"]:
                        if key in frame_output:
                            model_state["all_predictions"][key].append(
                                frame_output[key].to("cpu")
                            )
                    
                    model_state["known_paths"].add(path)
                    model_state["frame_idx"] += 1
                    
                    del frame_output, frame_image
                
                write_log(f"处理完成，共 {model_state['frame_idx']} 帧", "ok")
            else:
                if model_state.get("finish_requested"):
                    write_log(f"无新帧且已请求结束，进入空闲等待，已处理 {model_state['frame_idx']} 帧", "ok")
                    model_state["is_streaming"] = False
            
            time.sleep(0.5)
            
        except Exception as e:
            write_log(f"监控线程异常: {e}", "err")
            traceback.print_exc()
            time.sleep(1)

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    write_log("服务启动中...", "info")
    
    if not model_state["initialized"]:
        model_state["model"], model_state["device"] = load_model()
        model_state["initialized"] = True
        
        # 启动监控线程
        monitor = threading.Thread(target=frame_monitor_thread, daemon=True)
        monitor.start()
        
        write_log("服务启动完成", "ok")

@app.get("/")
async def root():
    return {"status": "running", "service": "LingBot-MAP Streaming Service"}

@app.post("/batch/{batch_id}/start_inference")
async def start_inference(batch_id: str, body: dict):
    """开始流式推理"""
    batch_dir = DATA_DIR / batch_id
    frames_dir = batch_dir / "frames"

    frames_dir.mkdir(parents=True, exist_ok=True)

    
    # 获取已上传的帧数（保留之前上传的帧计数）
    exts = (".jpg", ".jpeg", ".png")
    all_paths = []
    for ext in exts:
        all_paths.extend(glob.glob(str(frames_dir / f"*{ext}")))
    existing_frames = len(sorted(set(all_paths)))
    
    ki = body.get("keyframe_interval", KEYFRAME_INTERVAL_DEFAULT)
    max_img = body.get("max_images", None)
    
    # 检查是否有推理任务正在运行（单用户架构，不支持并发）
    if model_state["is_streaming"]:
        raise HTTPException(
            status_code=409,
            detail="当前有推理任务正在运行，请先结束当前任务"
        )
    
    model_state["current_batch_id"] = batch_id
    model_state["frame_idx"] = 0
    model_state["keyframe_interval"] = ki
    model_state["max_images"] = max_img
    model_state["known_paths"] = set()
    model_state["is_streaming"] = True
    model_state["stop_event"].clear()
    model_state["finish_requested"] = False
    model_state["all_predictions"] = {
        "pose_enc": [],
        "depth": [],
        "depth_conf": [],
        "world_points": [],
        "world_points_conf": [],
        "images": [],
    }
    
    frame_cache.clear()
    batch_logs.clear()
    
    # 清理KV缓存，准备新批次
    if model_state["model"]:
        model_state["model"].clean_kv_cache()
    
    update_status(
        batch_id=batch_id,
        status="streaming",
        total_frames=existing_frames,      # ✅ 保留已上传的帧数
        uploaded_frames=existing_frames,   # ✅ 保留已上传的帧数
        processed_frames=0,
        total_points=0,
    )
    
    write_log(f"开始流式推理: {batch_id}, 已上传 {existing_frames} 帧", "info")
    
    return {"success": True, "batch_id": batch_id, "message": "流式推理已启动", "existing_frames": existing_frames}

@app.post("/batch/{batch_id}/frames")
async def upload_frames(batch_id: str, files: list[UploadFile] = File(...)):
    """上传帧到批次"""
    batch_dir = DATA_DIR / batch_id
    frames_dir = batch_dir / "frames"
    
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    existing_frames = len(list(frames_dir.glob("frame_*.jpg")))
    
    if existing_frames + len(files) > MAX_FRAMES_PER_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"超过最大帧数量 {MAX_FRAMES_PER_BATCH}"
        )
    
    saved_count = 0
    for i, file in enumerate(files):
        if file.content_type not in SUPPORTED_FORMATS:
            raise HTTPException(status_code=400, detail=f"不支持的格式: {file.content_type}")
        
        frame_data = await file.read()
        frame_index = existing_frames + i
        frame_path = frames_dir / f"frame_{frame_index:03d}.jpg"
        
        with open(frame_path, 'wb') as f:
            f.write(frame_data)
        
        saved_count += 1
    
    total_frames = existing_frames + saved_count
    update_status(
        total_frames=total_frames,
        uploaded_frames=total_frames
    )
    
    return {
        "success": True,
        "batch_id": batch_id,
        "uploaded_count": saved_count,
        "total_frames": total_frames
    }

def _auto_dgsg_pipeline(batch_id: str):
    """后台线程：推理完成后自动跑 dgsg 建图管线 + convert 转换 + 启动 viewer"""
    pipeline_script = "/home/sscy/lingbot-map/stmem-main/run_dgsg_pipeline.sh"
    convert_script = "/home/sscy/lingbot-map/stmem-main/scripts/convert_memory_pc.py"
    scene_name = "lingbot"  # 固定输出到 lingbot 目录
    update_status(dgsg_status="building")
    try:
        write_log(f"[DGSG] 建图管线启动: batch={batch_id} → scene={scene_name}", "info")
        result = subprocess.run(
            ["bash", pipeline_script, batch_id, scene_name],
            capture_output=True, text=True, timeout=3600
        )
        if result.returncode == 0:
            update_status(dgsg_status="done")
            write_log("[DGSG] 建图管线完成", "ok")

            # ── 建图成功后自动 convert ──
            write_log(f"[CONVERT] 开始转换点云数据: scene={scene_name}", "info")
            try:
                cv_result = subprocess.run(
                    ["python3", convert_script, scene_name],
                    capture_output=True, text=True, timeout=300
                )
                for line in cv_result.stdout.strip().split('\n'):
                    if line.strip():
                        write_log(f"[CONVERT] {line.strip()}", "info")
                if cv_result.returncode != 0:
                    write_log(f"[CONVERT] 转换失败 (rc={cv_result.returncode}): {cv_result.stderr[:300]}", "err")
                else:
                    write_log("[CONVERT] 转换完成，前端可切换至空间记忆模式", "ok")
            except subprocess.TimeoutExpired:
                write_log("[CONVERT] 转换超时（>5分钟）", "err")
            except Exception as e:
                write_log(f"[CONVERT] 转换异常: {e}", "err")
        else:
            update_status(dgsg_status="error", dgsg_error=result.stderr[:200])
            write_log(f"[DGSG] 建图管线失败 (rc={result.returncode}): {result.stderr[:200]}", "err")
            if result.stdout:
                for line in result.stdout.strip().split('\n')[-5:]:
                    if line.strip():
                        write_log(f"[DGSG] {line.strip()}", "err")
    except subprocess.TimeoutExpired:
        update_status(dgsg_status="error", dgsg_error="timeout")
        write_log("[DGSG] 建图管线超时（>1小时），已终止", "err")
    except Exception as e:
        update_status(dgsg_status="error", dgsg_error=str(e))
        write_log(f"[DGSG] 建图管线异常: {e}", "err")

@app.post("/batch/{batch_id}/finish_inference")
async def finish_inference(batch_id: str):
    """完成推理：通知监控线程不再处理新帧，等待处理完剩余帧后标记完成"""
    # 设置结束请求标志，让监控线程处理完剩余帧后自行停止
    model_state["finish_requested"] = True
    write_log("收到结束请求，等待监控线程处理完剩余帧...", "info")
    
    # 等待监控线程处理完所有剩余帧（最多等120秒）
    wait_count = 0
    while model_state["is_streaming"] and wait_count < 240:
        await asyncio.sleep(0.5)
        wait_count += 1
    
    # 确保停止
    model_state["is_streaming"] = False
    model_state["finish_requested"] = False
    
    total_processed = len(frame_cache)
    total_points = sum(len(f["points"]) for f in frame_cache.values())
    
    update_status(
        status="completed",
        processed_frames=total_processed,
        total_points=total_points
    )
    
    write_log(f"推理完成，共 {total_processed} 帧，{total_points} 点", "ok")

    # 更新 latest 软链接，始终指向最新 batch
    latest_link = DATA_DIR / "latest"
    batch_dir = DATA_DIR / batch_id
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    os.symlink(batch_dir, latest_link)
    write_log(f"latest → {batch_id}", "ok")

    # 自动触发 dgsg 建图管线
    threading.Thread(target=_auto_dgsg_pipeline, args=(batch_id,), daemon=True).start()
    write_log("dgsg 建图管线已在后台启动", "info")

    return {
        "success": True,
        "batch_id": batch_id,
        "total_frames": total_processed,
        "total_points": total_points
    }

@app.post("/batch/{batch_id}/force_stop")
async def force_stop(batch_id: str):
    """强制停止：立即终止处理，清空所有缓存和状态，删除数据目录"""
    model_state["is_streaming"] = False
    model_state["finish_requested"] = False
    model_state["current_batch_id"] = None
    model_state["frame_idx"] = 0
    model_state["known_paths"] = set()
    model_state["all_predictions"] = {
        "pose_enc": [],
        "depth": [],
        "depth_conf": [],
        "world_points": [],
        "world_points_conf": [],
        "images": [],
    }
    
    frame_cache.clear()
    cache_frame_order.clear()
    batch_logs.clear()
    batch_status.clear()
    
    if model_state["model"]:
        model_state["model"].clean_kv_cache()
    
    batch_dir = DATA_DIR / batch_id
    if batch_dir.exists():
        shutil.rmtree(batch_dir)
    
    write_log(f"强制停止完成: {batch_id}, 所有数据已清空", "info")
    
    return {"success": True, "batch_id": batch_id, "message": "所有处理已终止，数据已清空"}

@app.get("/batch/{batch_id}/status")
async def get_status(batch_id: str):
    """获取批次状态"""
    return batch_status

@app.get("/batch/{batch_id}/logs")
async def get_logs(batch_id: str):
    """获取日志"""
    return {"logs": batch_logs}

@app.get("/batch/{batch_id}/metadata")
async def get_metadata(batch_id: str):
    """获取批次元数据（与 see/ 一致）"""
    batch_dir = DATA_DIR / batch_id
    frames_dir = batch_dir / "frames"
    
    # 获取帧数量
    exts = (".jpg", ".jpeg", ".png")
    all_paths = []
    for ext in exts:
        all_paths.extend(glob.glob(str(frames_dir / f"*{ext}")))
    num_frames = len(sorted(set(all_paths)))
    
    # 获取图片尺寸（从第一张图获取）
    image_width = 518
    image_height = 518
    if all_paths:
        try:
            first_frame = sorted(set(all_paths))[0]
            img = Image.open(first_frame)
            image_width, image_height = img.size
        except:
            pass
    
    # 计算场景中心和尺度（从缓存的点云计算）
    all_points = []
    for frame_idx in frame_cache:
        points = np.array(frame_cache[frame_idx]["points"])
        all_points.append(points)
    
    if all_points:
        all_points_np = np.vstack(all_points)
        scene_center = np.mean(all_points_np, axis=0).tolist()
        scene_min = np.min(all_points_np, axis=0)
        scene_max = np.max(all_points_np, axis=0)
        scene_scale = float(np.max(scene_max - scene_min)) or 1.0
    else:
        scene_center = [0.0, 0.0, 0.0]
        scene_scale = 1.0
    
    metadata = {
        "scene_center": scene_center,
        "scene_scale": scene_scale,
        "num_frames": num_frames,
        "image_width": image_width,
        "image_height": image_height,
        "processed_frames": len(frame_cache),
    }
    
    return {"success": True, "metadata": metadata}

@app.get("/batch/{batch_id}/frame/{frame_index}/point_cloud")
async def get_frame_point_cloud(batch_id: str, frame_index: int):
    """获取单帧点云（从内存缓存）"""
    if frame_index not in frame_cache:
        raise HTTPException(status_code=404, detail=f"帧 {frame_index} 尚未处理完成")
    
    cached = frame_cache[frame_index]
    
    # 返回原始数据（与 live_camera.py 一致，不过滤）
    points_arr = np.array(cached["points"], dtype=np.float32)
    colors_arr = np.array(cached["colors"], dtype=np.float32)
    confs_arr = np.array(cached["confs"], dtype=np.float32)
    
    if len(points_arr) == 0:
        raise HTTPException(status_code=404, detail=f"帧 {frame_index} 无有效点云数据")
    
    # 使用 Base64 编码传输，减少数据量
    import base64
    
    return {
        "success": True,
        "batch_id": batch_id,
        "frame_index": frame_index,
        "points_b64": base64.b64encode(points_arr.flatten()).decode('utf-8'),
        "colors_b64": base64.b64encode(colors_arr.flatten()).decode('utf-8'),
        "confs_b64": base64.b64encode(confs_arr.flatten()).decode('utf-8'),
        "total_points": len(points_arr),
        "source": "memory",
        "encoding": "base64"
    }

@app.get("/batch/{batch_id}/frame/{frame_index}/camera")
async def get_frame_camera(batch_id: str, frame_index: int):
    """获取单帧相机参数"""
    if frame_index not in frame_cache:
        raise HTTPException(status_code=404, detail=f"帧 {frame_index} 尚未处理完成")
    
    cached = frame_cache[frame_index]
    
    return {
        "success": True,
        "batch_id": batch_id,
        "frame_index": frame_index,
        "camera": cached["camera"]
    }

@app.get("/batch/{batch_id}/frame/{frame_index}/image")
async def get_frame_image(batch_id: str, frame_index: int):
    """获取单帧原始图片"""
    batch_dir = DATA_DIR / batch_id
    frames_dir = batch_dir / "frames"
    
    image_path = frames_dir / f"frame_{frame_index:03d}.jpg"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"帧 {frame_index} 图片不存在")
    
    return FileResponse(image_path, media_type="image/jpeg")
