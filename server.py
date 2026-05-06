import os
import uuid
import torch
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ================ 直接导入 realtime_vis_demo.py 中的函数 ================
from realtime_vis_demo import postprocess, prepare_for_visualization
from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.geometry import unproject_depth_map_to_point_map
from lingbot_map.utils.load_fn import preprocess_image

# ================ 模型配置 ================
MODEL_PATH = "./checkpoints/robbyant/lingbot-map/lingbot-map-long.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
IMAGE_SIZE = 518
PATCH_SIZE = 14

# 自动创建文件夹
os.makedirs("upload", exist_ok=True)
os.makedirs("point_cloud_output", exist_ok=True)

# ================ FastAPI 应用 ================
app = FastAPI()

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
app.mount("/point_cloud", StaticFiles(directory="point_cloud_output"), name="point_cloud")

# 全局模型
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        print("🔹 正在加载实时模型...")
        print(f"   模型路径: {MODEL_PATH}")
        print(f"   设备: {DEVICE}")
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
        
        model = GCTStream(img_size=IMAGE_SIZE, patch_size=PATCH_SIZE)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device=DEVICE).eval()
        print("✅ 模型加载完成！")
    except Exception as e:
        print(f"⚠️ 模型加载失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

@app.get("/test/info")
def info():
    return {"status": "running", "model_loaded": model is not None}

def save_point_cloud_ply(points, filename):
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
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
    return ply_path

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
        
        frame = cv2.imread(image_save_path)
        if frame is None:
            return {"code": 500, "msg": "无法读取图片文件"}
        
        img_tensor = preprocess_image(frame, mode="crop", image_size=IMAGE_SIZE, patch_size=PATCH_SIZE)
        img_tensor = img_tensor.to(DEVICE).unsqueeze(0)
        
        images_for_post = torch.stack([
            torch.from_numpy(cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))).permute(2, 0, 1)
        ]).float() / 255.0
        
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
            result = model.inference_streaming(img_tensor)
        
        predictions, images_cpu = postprocess(result, images_for_post)
        vis_predictions = prepare_for_visualization(predictions, images_cpu)
        
        world_points = vis_predictions.get("world_points", vis_predictions.get("depth"))
        depth_conf = vis_predictions.get("depth_conf")
        
        if world_points is None:
            return {"code": 500, "msg": "模型未返回点云数据"}
        
        if world_points.ndim == 4 and world_points.shape[-1] == 1:
            extrinsic = vis_predictions["extrinsic"]
            intrinsic = vis_predictions["intrinsic"]
            world_points = unproject_depth_map_to_point_map(
                torch.from_numpy(world_points),
                torch.from_numpy(extrinsic),
                torch.from_numpy(intrinsic)
            )
        
        frame_points = world_points[0] if world_points.ndim > 3 else world_points
        pred_pts = frame_points.reshape(-1, 3)
        
        valid = np.isfinite(pred_pts).all(axis=1)
        pred_pts = pred_pts[valid]
        
        if depth_conf is not None:
            conf = depth_conf[0] if depth_conf.ndim > 3 else depth_conf
            conf_flat = conf.reshape(-1)[valid]
            mask = conf_flat > 1.5
            pred_pts = pred_pts[mask]
        
        max_points = 200
        if len(pred_pts) > max_points:
            indices = np.linspace(0, len(pred_pts) - 1, max_points, dtype=int)
            pred_pts = pred_pts[indices]
        
        if len(pred_pts) == 0:
            return {"code": 500, "msg": "生成的点云为空"}
        
        points = pred_pts.flatten().tolist()
        save_point_cloud_ply(points, ply_filename)
        point_cloud_url = f"http://localhost:8000/point_cloud/{ply_filename}"
        
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
