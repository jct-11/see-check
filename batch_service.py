"""LingBot-MAP 批次管理服务

功能：
1. 上传视频帧（按批次号组织）
2. 批量处理帧生成点云
3. 获取点云数据
4. 实时日志流

文件结构：
data/
└── {batch_id}/
    ├── frames/
    │   ├── frame_000.jpg
    │   ├── frame_001.jpg
    │   └── ...
    ├── output/
    │   └── point_cloud.ply
    ├── status.json
    └── logs.json
"""

import os
import json
import traceback
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import threading
from datetime import datetime

sys.path.insert(0, '/home/sscy/lingbot-map/lingbot-map-main')

from inference_api import run_inference

app = FastAPI(title="LingBot-MAP Batch Service", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("/home/sscy/lingbot-map/lingbot-map-main/data")
MAX_FRAMES_PER_BATCH = 5000
SUPPORTED_FORMATS = {"image/jpeg", "image/png", "image/webp"}
MODEL_PATH = "/home/sscy/lingbot-map/lingbot-map-main/checkpoints/robbyant/lingbot-map/lingbot-map-long.pt"

class BatchInfo(BaseModel):
    batch_id: str
    frame_count: int
    status: str
    total_points: int = 0
    error_message: str = ""

class ProcessRequest(BaseModel):
    model_config = {"extra": "allow"}
    batch_id: str = ""

class PointCloudResponse(BaseModel):
    success: bool
    batch_id: str
    points: list = []
    colors: list = []
    total_points: int = 0
    ply_path: str = ""

def write_log(batch_id, message, log_type="info"):
    """写入日志到批次日志文件"""
    batch_dir = DATA_DIR / batch_id
    log_file = batch_dir / "logs.json"
    
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "message": message,
        "type": log_type
    }
    
    logs = []
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except:
            logs = []
    
    logs.append(log_entry)
    
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"[{timestamp}] [{log_type}] {message}")
    sys.stdout.flush()

def update_batch_status(batch_id, status, total_points=0, error_message=""):
    """更新批次状态"""
    batch_dir = DATA_DIR / batch_id
    status_file = batch_dir / "status.json"

    status_data = {
        "batch_id": batch_id,
        "frame_count": len(list((batch_dir / "frames").glob("frame_*.jpg"))) if (batch_dir / "frames").exists() else 0,
        "status": status,
        "total_points": total_points,
        "error_message": error_message
    }

    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2)

@app.get("/")
async def root():
    return {"status": "running", "service": "LingBot-MAP Batch Service"}

@app.post("/batch/{batch_id}/frames")
async def upload_frames(batch_id: str, files: list[UploadFile] = File(...)):
    """上传帧到指定批次"""
    batch_dir = DATA_DIR / batch_id
    frames_dir = batch_dir / "frames"

    # 创建目录
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 获取已有帧数
    existing_frames = len(list(frames_dir.glob("frame_*.jpg")))

    # 检查是否超过最大帧数量
    if existing_frames + len(files) > MAX_FRAMES_PER_BATCH:
        raise HTTPException(status_code=400, detail=f"超过最大帧数量 {MAX_FRAMES_PER_BATCH}，当前已有 {existing_frames} 帧，尝试上传 {len(files)} 帧")

    # 保存文件
    for i, file in enumerate(files):
        if file.content_type not in SUPPORTED_FORMATS:
            raise HTTPException(status_code=400, detail=f"不支持的格式: {file.content_type}")

        frame_index = existing_frames + i
        frame_path = frames_dir / f"frame_{frame_index:03d}.jpg"

        with open(frame_path, 'wb') as f:
            f.write(await file.read())

    # 更新状态
    update_batch_status(batch_id, "pending")

    return {
        "success": True,
        "batch_id": batch_id,
        "uploaded_count": len(files),
        "total_frames": existing_frames + len(files)
    }

@app.post("/batch/{batch_id}/process")
async def process_batch(batch_id: str, request: ProcessRequest = None):
    """处理指定批次"""
    if request is None:
        request = ProcessRequest()

    batch_dir = DATA_DIR / batch_id
    frames_dir = batch_dir / "frames"

    if not batch_dir.exists():
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")

    frame_files = sorted(list(frames_dir.glob("frame_*.jpg")))
    if len(frame_files) == 0:
        raise HTTPException(status_code=400, detail=f"批次 {batch_id} 没有帧数据")

    write_log(batch_id, f"📷 处理批次 {batch_id}: {len(frame_files)} 帧", "info")

    update_batch_status(batch_id, "processing")

    try:
        write_log(batch_id, "🔄 开始调用推理API...", "info")
        output_dir = batch_dir / "output"
        write_log(batch_id, f"   image_folder: {frames_dir}", "info")
        write_log(batch_id, f"   model_path: {MODEL_PATH}", "info")
        write_log(batch_id, f"   num_frames: {len(frame_files)}", "info")
        
        result = run_inference(
            image_folder=str(frames_dir),
            model_path=MODEL_PATH,
            output_dir=str(output_dir),
            num_frames=len(frame_files),
            image_size=518,
            patch_size=14,
            num_scale_frames=8,
            keyframe_interval=1,
            mode="streaming"
        )
        
        write_log(batch_id, f"   推理返回: success={result.get('success')}, frames={result.get('num_frames', 0)}, points={result.get('total_points', 0)}", "info")

        if result["success"]:
            update_batch_status(batch_id, "completed", result["total_points"])
            write_log(batch_id, f"✅ 处理完成，共 {result['num_frames']} 帧，总点数: {result['total_points']}", "ok")

            return {
                "success": True,
                "batch_id": batch_id,
                "num_frames": result["num_frames"],
                "total_points": result["total_points"],
                "frame_files": result["frame_files"],
                "merged_ply_path": result["merged_ply_path"],
                "message": "处理完成，已保存每帧点云和合并点云"
            }
        else:
            update_batch_status(batch_id, "failed", error_message=result.get("error", "Unknown error"))
            write_log(batch_id, f"❌ 处理失败: {result.get('error', 'Unknown error')}", "err")
            raise HTTPException(status_code=500, detail=f"处理失败: {result.get('error', 'Unknown error')}")

    except Exception as e:
        import sys
        write_log(batch_id, f"❌ 处理批次 {batch_id} 失败: {str(e)}", "err")
        traceback.print_exc()
        sys.stdout.flush()
        error_detail = f"处理失败: {str(e)}"
        if "result" in dir() and result.get("traceback"):
            error_detail += f"\n{result.get('traceback')}"
        update_batch_status(batch_id, "failed", error_message=str(e))
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/batch/{batch_id}/logs")
async def get_batch_logs(batch_id: str):
    """获取批次处理日志"""
    log_file = DATA_DIR / batch_id / "logs.json"

    if not log_file.exists():
        return {"logs": []}

    try:
        with open(log_file, 'r') as f:
            logs = json.load(f)
        return {"logs": logs}
    except:
        return {"logs": []}

@app.post("/test/process_local_images")
async def test_process_local_images():
    """测试接口：直接处理本地图片集"""
    test_image_folder = "/home/sscy/lingbot-map/lingbot-map-main/camera/camera_frames_20260509_133039/"
    test_batch_id = "test_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    
    write_log(test_batch_id, "🧪 测试模式：处理本地图片集", "info")
    write_log(test_batch_id, f"   图片路径: {test_image_folder}", "info")
    write_log(test_batch_id, f"   批次号: {test_batch_id}", "info")
    
    batch_dir = DATA_DIR / test_batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    update_batch_status(test_batch_id, "processing")
    
    try:
        write_log(test_batch_id, "🔄 开始调用推理API...", "info")
        output_dir = batch_dir / "output"
        
        result = run_inference(
            image_folder=test_image_folder,
            model_path=MODEL_PATH,
            output_dir=str(output_dir),
            num_frames=5000,
            image_size=518,
            patch_size=14,
            num_scale_frames=8,
            keyframe_interval=1,
            mode="streaming"
        )
        
        write_log(test_batch_id, f"   推理返回: success={result.get('success')}, frames={result.get('num_frames', 0)}, points={result.get('total_points', 0)}", "info")
        
        if result["success"]:
            update_batch_status(test_batch_id, "completed", result["total_points"])
            write_log(test_batch_id, f"✅ 测试处理完成，共 {result['num_frames']} 帧，总点数: {result['total_points']}", "ok")
            
            return {
                "success": True,
                "batch_id": test_batch_id,
                "num_frames": result["num_frames"],
                "total_points": result["total_points"],
                "frame_files": result["frame_files"],
                "merged_ply_path": result["merged_ply_path"],
                "message": "测试处理完成，已保存每帧点云和合并点云"
            }
        else:
            update_batch_status(test_batch_id, "failed", error_message=result.get("error", "Unknown error"))
            write_log(test_batch_id, f"❌ 测试处理失败: {result.get('error', 'Unknown error')}", "err")
            raise HTTPException(status_code=500, detail=f"测试处理失败: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        write_log(test_batch_id, f"❌ 测试处理异常: {str(e)}", "err")
        traceback.print_exc()
        sys.stdout.flush()
        update_batch_status(test_batch_id, "failed", error_message=str(e))
        raise HTTPException(status_code=500, detail=f"测试处理失败: {str(e)}")

@app.get("/batch/{batch_id}/status")
async def get_batch_status(batch_id: str):
    """获取批次状态"""
    status_file = DATA_DIR / batch_id / "status.json"

    if not status_file.exists():
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")

    with open(status_file, 'r') as f:
        return json.load(f)

@app.get("/batch/{batch_id}/point_cloud")
async def get_point_cloud(batch_id: str, max_points: int = 100000):
    """获取批次点云数据
    
    Args:
        batch_id: 批次ID
        max_points: 最大返回点数，默认10万点（用于预览）
    """
    print(f"📥 获取点云请求: batch_id={batch_id}, max_points={max_points}")
    
    batch_dir = DATA_DIR / batch_id
    ply_path = batch_dir / "output" / "point_cloud_merged.ply"

    if not batch_dir.exists():
        print(f"❌ 批次目录不存在: {batch_dir}")
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")

    if not ply_path.exists():
        print(f"❌ PLY文件不存在: {ply_path}")
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 点云未生成")

    print(f"✅ PLY文件存在: {ply_path}")
    
    # 读取 PLY 文件（支持二进制和ASCII格式）
    points = []
    colors = []

    with open(ply_path, 'rb') as f:
        # 读取头部判断格式
        header = b""
        while b"end_header" not in header:
            line = f.readline()
            if not line:
                break
            header += line
        
        # 判断是否为二进制格式
        is_binary = b"binary" in header.lower()
        
        if is_binary:
            # 解析头部获取顶点数量
            import struct
            header_str = header.decode('ascii')
            total_vertices = 0
            for line in header_str.split('\n'):
                if line.startswith('element vertex'):
                    total_vertices = int(line.split()[-1])
                    break
            
            # 计算采样间隔
            sample_interval = max(1, total_vertices // max_points)
            actual_points = min(total_vertices, max_points)
            
            # 读取二进制数据（带采样）
            point_idx = 0
            saved_idx = 0
            while saved_idx < actual_points:
                data = f.read(12 + 3)  # 3个float(4字节) + 3个uchar(1字节)
                if len(data) < 15:
                    break
                if point_idx % sample_interval == 0:
                    x, y, z = struct.unpack('fff', data[:12])
                    r, g, b = struct.unpack('BBB', data[12:15])
                    points.append([x, y, z])
                    colors.append([r / 255.0, g / 255.0, b / 255.0])
                    saved_idx += 1
                point_idx += 1
        else:
            # ASCII格式
            with open(ply_path, 'r') as f:
                lines = f.readlines()
                header_end = lines.index("end_header\n")
                total_vertices = len(lines) - header_end - 1
                
                # 计算采样间隔
                sample_interval = max(1, total_vertices // max_points)
                
                point_idx = 0
                for line in lines[header_end + 1:]:
                    if point_idx % sample_interval == 0 and len(points) < max_points:
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                            colors.append([int(parts[3]) / 255.0, int(parts[4]) / 255.0, int(parts[5]) / 255.0])
                    point_idx += 1

    return {
        "success": True,
        "batch_id": batch_id,
        "points": points,
        "colors": colors,
        "total_points": len(points),
        "original_total": total_vertices if 'total_vertices' in dir() else len(points),
        "ply_path": str(ply_path)
    }

@app.get("/batch/{batch_id}/point_cloud/ply")
async def download_point_cloud(batch_id: str):
    """下载合并的PLY文件"""
    ply_path = DATA_DIR / batch_id / "output" / "point_cloud_merged.ply"

    if not ply_path.exists():
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 点云未生成")

    from fastapi.responses import FileResponse
    return FileResponse(ply_path, media_type="application/octet-stream", filename=f"{batch_id}_merged.ply")

@app.get("/batch/{batch_id}/frame/{frame_index}/point_cloud")
async def get_frame_point_cloud(batch_id: str, frame_index: int):
    """获取单帧点云数据
    
    Args:
        batch_id: 批次ID
        frame_index: 帧索引（0开始）
    """
    print(f"📥 获取单帧点云: batch_id={batch_id}, frame_index={frame_index}")
    
    batch_dir = DATA_DIR / batch_id
    ply_path = batch_dir / "output" / f"frame_{frame_index:03d}.ply"

    if not batch_dir.exists():
        print(f"❌ 批次目录不存在: {batch_dir}")
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")

    if not ply_path.exists():
        print(f"❌ 帧PLY文件不存在: {ply_path}")
        raise HTTPException(status_code=404, detail=f"帧 {frame_index} 点云未生成")

    print(f"✅ 帧PLY文件存在: {ply_path}")
    
    # 读取 PLY 文件
    points = []
    colors = []

    with open(ply_path, 'rb') as f:
        # 读取头部判断格式
        header = b""
        while b"end_header" not in header:
            line = f.readline()
            if not line:
                break
            header += line
        
        # 判断是否为二进制格式
        is_binary = b"binary" in header.lower()
        
        if is_binary:
            # 解析头部获取顶点数量
            import struct
            header_str = header.decode('ascii')
            total_vertices = 0
            for line in header_str.split('\n'):
                if line.startswith('element vertex'):
                    total_vertices = int(line.split()[-1])
                    break
            
            # 读取二进制数据
            for i in range(total_vertices):
                data = f.read(12 + 3)  # 3个float(4字节) + 3个uchar(1字节)
                if len(data) < 15:
                    break
                x, y, z = struct.unpack('fff', data[:12])
                r, g, b = struct.unpack('BBB', data[12:15])
                points.append([x, y, z])
                colors.append([r / 255.0, g / 255.0, b / 255.0])
        else:
            # ASCII格式
            with open(ply_path, 'r') as f:
                lines = f.readlines()
                header_end = lines.index("end_header\n")
                
                for line in lines[header_end + 1:]:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        colors.append([int(parts[3]) / 255.0, int(parts[4]) / 255.0, int(parts[5]) / 255.0])

    # 读取置信度数据
    confs = []
    conf_path = batch_dir / "output" / "world_points_conf" / f"world_conf_{frame_index:03d}.npy"
    if conf_path.exists():
        import numpy as np
        world_conf = np.load(conf_path, allow_pickle=True)
        conf_flat = world_conf.reshape(-1)
        confs = conf_flat.tolist()

    return {
        "success": True,
        "batch_id": batch_id,
        "frame_index": frame_index,
        "points": points,
        "colors": colors,
        "confs": confs,
        "total_points": len(points),
        "ply_path": str(ply_path)
    }

@app.get("/batch/{batch_id}/frame/{frame_index}/point_cloud/ply")
async def download_frame_point_cloud(batch_id: str, frame_index: int):
    """下载单帧PLY文件"""
    ply_path = DATA_DIR / batch_id / "output" / f"frame_{frame_index:03d}.ply"

    if not ply_path.exists():
        raise HTTPException(status_code=404, detail=f"帧 {frame_index} 点云未生成")

    from fastapi.responses import FileResponse
    return FileResponse(ply_path, media_type="application/octet-stream", filename=f"{batch_id}_frame_{frame_index:03d}.ply")

@app.get("/batch/{batch_id}/frames/info")
async def get_batch_frames_info(batch_id: str):
    """获取批次所有帧的信息"""
    batch_dir = DATA_DIR / batch_id
    output_dir = batch_dir / "output"

    if not batch_dir.exists():
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")

    if not output_dir.exists():
        return {"frames": [], "total_frames": 0}

    # 查找所有帧PLY文件
    frame_files = sorted(list(output_dir.glob("frame_*.ply")))
    
    frames_info = []
    for i, ply_file in enumerate(frame_files):
        # 读取PLY文件获取点数
        try:
            with open(ply_file, 'rb') as f:
                header = b""
                while b"end_header" not in header:
                    line = f.readline()
                    if not line:
                        break
                    header += line
                
                header_str = header.decode('ascii')
                num_points = 0
                for line in header_str.split('\n'):
                    if line.startswith('element vertex'):
                        num_points = int(line.split()[-1])
                        break
                
                frames_info.append({
                    "frame_index": i,
                    "ply_file": str(ply_file),
                    "num_points": num_points
                })
        except:
            frames_info.append({
                "frame_index": i,
                "ply_file": str(ply_file),
                "num_points": 0
            })

    return {
        "batch_id": batch_id,
        "frames": frames_info,
        "total_frames": len(frames_info)
    }

@app.get("/batches")
async def list_batches():
    """获取所有批次列表"""
    if not DATA_DIR.exists():
        return {"batches": []}

    batches = []
    for item in DATA_DIR.iterdir():
        if item.is_dir():
            status_file = item / "status.json"
            if status_file.exists():
                with open(status_file, 'r') as f:
                    batches.append(json.load(f))

    return {"batches": batches}

@app.delete("/batch/{batch_id}")
async def delete_batch(batch_id: str):
    """删除批次"""
    import shutil

    batch_dir = DATA_DIR / batch_id
    if not batch_dir.exists():
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")

    shutil.rmtree(batch_dir)
    return {"success": True, "message": f"批次 {batch_id} 已删除"}

@app.get("/batch/{batch_id}/manifest")
async def get_batch_manifest(batch_id: str):
    """获取批次的文件清单"""
    batch_dir = DATA_DIR / batch_id
    manifest_path = batch_dir / "output" / "manifest.json"

    if not batch_dir.exists():
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")

    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="manifest.json 不存在")

    import json
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    return {"success": True, "manifest": manifest}

@app.get("/batch/{batch_id}/cameras")
async def get_cameras(batch_id: str):
    """获取 cameras.json（与 see/ 一致）"""
    batch_dir = DATA_DIR / batch_id
    cameras_path = batch_dir / "output" / "cameras.json"

    if not batch_dir.exists():
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")

    if not cameras_path.exists():
        raise HTTPException(status_code=404, detail="cameras.json 不存在")

    import json
    with open(cameras_path, 'r') as f:
        cameras = json.load(f)
    
    return {"success": True, "cameras": cameras}

@app.get("/batch/{batch_id}/merged_point_cloud")
async def get_merged_point_cloud(batch_id: str):
    """获取合并后的点云数据（与 see/ 一致）"""
    batch_dir = DATA_DIR / batch_id
    ply_path = batch_dir / "output" / "point_cloud_merged.ply"

    if not batch_dir.exists():
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")

    if not ply_path.exists():
        raise HTTPException(status_code=404, detail="point_cloud_merged.ply 不存在")

    import struct
    
    points = []
    colors = []
    
    with open(ply_path, 'rb') as f:
        content = f.read()
        
        # 解析 PLY 头部
        header_end = content.find(b'end_header\n') + len(b'end_header\n')
        header = content[:header_end].decode('ascii')
        
        # 获取顶点数量
        import re
        match = re.search(r'element vertex (\d+)', header)
        if not match:
            raise HTTPException(status_code=500, detail="无法解析 PLY 文件")
        
        num_vertices = int(match.group(1))
        
        # 读取二进制数据
        binary_data = content[header_end:]
        
        # 每个顶点: x(4) + y(4) + z(4) + r(1) + g(1) + b(1) = 15 bytes
        vertex_size = 15
        for i in range(num_vertices):
            offset = i * vertex_size
            if offset + vertex_size > len(binary_data):
                break
            
            x = struct.unpack('<f', binary_data[offset:offset+4])[0]
            y = struct.unpack('<f', binary_data[offset+4:offset+8])[0]
            z = struct.unpack('<f', binary_data[offset+8:offset+12])[0]
            r = binary_data[offset+12]
            g = binary_data[offset+13]
            b = binary_data[offset+14]
            
            points.extend([x, y, z])
            colors.extend([r / 255.0, g / 255.0, b / 255.0])
    
    return {
        "success": True,
        "batch_id": batch_id,
        "num_points": len(points) // 3,
        "points": points,
        "colors": colors
    }

@app.get("/batch/{batch_id}/metadata")
async def get_metadata(batch_id: str):
    """获取 metadata.json（与 see/ 一致）"""
    batch_dir = DATA_DIR / batch_id
    metadata_path = batch_dir / "output" / "metadata.json"

    if not batch_dir.exists():
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")

    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="metadata.json 不存在")

    import json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return {"success": True, "metadata": metadata}

@app.get("/batch/{batch_id}/extrinsic")
async def get_extrinsic(batch_id: str):
    """获取相机位姿 (extrinsic)"""
    batch_dir = DATA_DIR / batch_id
    extrinsic_path = batch_dir / "output" / "extrinsic.npy"

    if not batch_dir.exists():
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")

    if not extrinsic_path.exists():
        raise HTTPException(status_code=404, detail="extrinsic.npy 不存在")

    import numpy as np
    extrinsic = np.load(extrinsic_path, allow_pickle=True)
    
    return {
        "success": True,
        "batch_id": batch_id,
        "extrinsic": extrinsic.tolist(),
        "shape": list(extrinsic.shape)
    }

@app.get("/batch/{batch_id}/intrinsic")
async def get_intrinsic(batch_id: str):
    """获取相机内参 (intrinsic)"""
    batch_dir = DATA_DIR / batch_id
    intrinsic_path = batch_dir / "output" / "intrinsic.npy"

    if not batch_dir.exists():
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")

    if not intrinsic_path.exists():
        raise HTTPException(status_code=404, detail="intrinsic.npy 不存在")

    import numpy as np
    intrinsic = np.load(intrinsic_path, allow_pickle=True)
    
    return {
        "success": True,
        "batch_id": batch_id,
        "intrinsic": intrinsic.tolist(),
        "shape": list(intrinsic.shape)
    }

@app.get("/batch/{batch_id}/frame/{frame_index}/depth")
async def get_frame_depth(batch_id: str, frame_index: int):
    """获取单帧深度图"""
    batch_dir = DATA_DIR / batch_id
    depth_path = batch_dir / "output" / "depth" / f"depth_{frame_index:03d}.npy"

    if not batch_dir.exists():
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")

    if not depth_path.exists():
        raise HTTPException(status_code=404, detail=f"深度图 {frame_index} 不存在")

    import numpy as np
    depth = np.load(depth_path, allow_pickle=True)
    
    return {
        "success": True,
        "batch_id": batch_id,
        "frame_index": frame_index,
        "depth": depth.tolist(),
        "shape": list(depth.shape)
    }

@app.get("/batch/{batch_id}/frame/{frame_index}/depth_conf")
async def get_frame_depth_conf(batch_id: str, frame_index: int):
    """获取单帧深度置信度"""
    batch_dir = DATA_DIR / batch_id
    conf_path = batch_dir / "output" / "depth_conf" / f"depth_conf_{frame_index:03d}.npy"

    if not batch_dir.exists():
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")

    if not conf_path.exists():
        raise HTTPException(status_code=404, detail=f"深度置信度 {frame_index} 不存在")

    import numpy as np
    depth_conf = np.load(conf_path, allow_pickle=True)
    
    return {
        "success": True,
        "batch_id": batch_id,
        "frame_index": frame_index,
        "depth_conf": depth_conf.tolist(),
        "shape": list(depth_conf.shape)
    }

@app.get("/batch/{batch_id}/frame/{frame_index}/world_conf")
async def get_frame_world_conf(batch_id: str, frame_index: int):
    """获取单帧 world_points_conf"""
    batch_dir = DATA_DIR / batch_id
    conf_path = batch_dir / "output" / "world_points_conf" / f"world_conf_{frame_index:03d}.npy"

    if not batch_dir.exists():
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")

    if not conf_path.exists():
        raise HTTPException(status_code=404, detail=f"world_points_conf {frame_index} 不存在")

    import numpy as np
    world_conf = np.load(conf_path, allow_pickle=True)
    
    return {
        "success": True,
        "batch_id": batch_id,
        "frame_index": frame_index,
        "world_points_conf": world_conf.tolist(),
        "shape": list(world_conf.shape)
    }

@app.get("/batch/{batch_id}/frame/{frame_index}/camera")
async def get_frame_camera(batch_id: str, frame_index: int):
    """获取单帧相机参数（外参+内参）"""
    batch_dir = DATA_DIR / batch_id
    
    if not batch_dir.exists():
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")
    
    # ✅ 优先读取 cameras.json（已经正确格式化的 w2c 格式）
    cameras_path = batch_dir / "output" / "cameras.json"
    if cameras_path.exists():
        with open(cameras_path, 'r') as f:
            cameras = json.load(f)
        
        if frame_index >= len(cameras):
            raise HTTPException(status_code=404, detail=f"帧 {frame_index} 超出范围")
        
        return {
            "success": True,
            "batch_id": batch_id,
            "frame_index": frame_index,
            "camera": cameras[frame_index]
        }
    
    # 备用：从 extrinsic.npy 和 intrinsic.npy 计算
    import numpy as np
    
    extrinsic_path = batch_dir / "output" / "extrinsic.npy"
    if not extrinsic_path.exists():
        raise HTTPException(status_code=404, detail="cameras.json 和 extrinsic.npy 都不存在")
    
    extrinsic = np.load(extrinsic_path, allow_pickle=True)
    if frame_index >= extrinsic.shape[0]:
        raise HTTPException(status_code=404, detail=f"帧 {frame_index} 超出范围")
    
    intrinsic_path = batch_dir / "output" / "intrinsic.npy"
    if not intrinsic_path.exists():
        raise HTTPException(status_code=404, detail="intrinsic.npy 不存在")
    
    intrinsic = np.load(intrinsic_path, allow_pickle=True)
    
    # ✅ 关键修复：extrinsic.npy 中已经是 w2c 格式（经过 closed_form_inverse_se3_general）
    # 直接使用，不需要再反转
    ext_i = extrinsic[frame_index]  # (3, 4) w2c 格式
    
    R_w2c = ext_i[:3, :3].tolist()
    t_w2c = ext_i[:3, 3].tolist()
    
    fx = float(intrinsic[frame_index, 0, 0]) if intrinsic.ndim == 3 else float(intrinsic[0, 0])
    fy = float(intrinsic[frame_index, 1, 1]) if intrinsic.ndim == 3 else float(intrinsic[1, 1])
    cx = float(intrinsic[frame_index, 0, 2]) if intrinsic.ndim == 3 else float(intrinsic[0, 2])
    cy = float(intrinsic[frame_index, 1, 2]) if intrinsic.ndim == 3 else float(intrinsic[1, 2])
    
    image_w = 640
    image_h = 480
    metadata_path = batch_dir / "output" / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            if 'image_w' in metadata:
                image_w = metadata['image_w']
            if 'image_h' in metadata:
                image_h = metadata['image_h']
    
    return {
        "success": True,
        "batch_id": batch_id,
        "frame_index": frame_index,
        "camera": {
            "R_w2c": R_w2c,
            "t_w2c": t_w2c,
            "focal": [fx, fy],        # 焦距
            "pp": [cx, cy],           # 主点
            "image_w": image_w,       # 图像宽度
            "image_h": image_h        # 图像高度
        }
    }

@app.get("/batch/{batch_id}/frame/{frame_index}/image")
async def get_frame_image(batch_id: str, frame_index: int):
    """获取单帧图像"""
    from fastapi.responses import FileResponse
    
    batch_dir = DATA_DIR / batch_id
    img_dir = batch_dir / "output" / "images"
    image_path = img_dir / f"frame_{frame_index:06d}.jpg"

    if not batch_dir.exists():
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")

    if not image_path.exists():
        alt_path = img_dir / f"image_{frame_index:03d}.png"
        if alt_path.exists():
            return FileResponse(alt_path, media_type="image/png")
        raise HTTPException(status_code=404, detail=f"图像 {frame_index} 不存在")

    return FileResponse(image_path, media_type="image/jpeg")

@app.get("/batch/{batch_id}/view/{filename:path}")
async def serve_batch_file(batch_id: str, filename: str):
    """动态服务每个批次的 output 目录下的文件（PLY/JSON/JPEG 等）"""
    batch_dir = DATA_DIR / batch_id
    output_dir = batch_dir / "output"

    if not batch_dir.exists():
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")

    file_path = output_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文件 {filename} 不存在")

    if not file_path.resolve().is_relative_to(output_dir.resolve()):
        raise HTTPException(status_code=403, detail="禁止访问批次目录外的文件")

    media_map = {
        ".ply": "application/octet-stream",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".json": "application/json",
        ".bin": "application/octet-stream",
        ".npy": "application/octet-stream",
    }
    ext = file_path.suffix.lower()
    media_type = media_map.get(ext, "application/octet-stream")

    return FileResponse(file_path, media_type=media_type)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)