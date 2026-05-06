"""
LingBot-MAP Real-time Streaming Demo
完全按照 demo.py 的逻辑实现实时摄像头可视化

实现效果与 demo.py 一致：
- 3D点云（带颜色和置信度过滤）
- 相机轨迹（渐变颜色）
- 相机视锥体（渐变颜色）
- 原始图像序列
- 完整的GUI控制
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
import matplotlib.cm as cm
import uuid

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import viser
import viser.transforms as tf

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3, closed_form_inverse_se3_general, unproject_depth_map_to_point_map
from lingbot_map.utils.load_fn import preprocess_image

# FastAPI dependencies for HTTP API
try:
    from fastapi import FastAPI, UploadFile, File, Response, WebSocket, WebSocketDisconnect
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

class RealTimeViewer:
    """实时3D可视化器 - 完全模拟 PointCloudViewer 的效果"""
    
    def __init__(self, args):
        self.args = args
        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
        
        # State
        self.pc_handles = []
        self.cam_handles = []
        self.frame_nodes = []
        self.vis_threshold = args.conf_threshold
        self.point_size = args.point_size
        self.show_camera = True
        
        # Data storage
        self.pcs = {}  # {step: {"pc": ..., "color": ..., "conf": ...}}
        self.cam_dict = None  # {"focal": [], "pp": [], "R": [], "t": []}
        self.all_steps = []
        self.original_images = []
        self.traj_list = []
        
        # Accumulated point cloud for persistent mapping
        self.accumulated_points = None
        self.accumulated_colors = None
        self.accumulated_conf = None
        self.accumulated_pc_handle = None
        
        # GUI elements
        self.gui_timestep = None
        self.show_video_checkbox = None
        self.current_frame_image = None
        self.psize_slider = None
        self.camsize_slider = None
        self.downsample_slider = None
        self.vis_threshold_slider = None
        self.camera_downsample_slider = None
        self.show_camera_checkbox = None
        self.fourd = False
        
        # Setup GUI
        self._setup_gui()
        self.server.on_client_connect(self._connect_client)
    
    def _setup_gui(self):
        """Setup GUI controls (完全按照 PointCloudViewer)"""
        # Video frame display
        with self.server.gui.add_folder("Video Display"):
            self.show_video_checkbox = self.server.gui.add_checkbox("Show Current Frame", initial_value=True)
            self.current_frame_image = self.server.gui.add_image(
                np.zeros((480, 640, 3), dtype=np.uint8), label="Current Frame"
            )

        # Preset view direction buttons
        with self.server.gui.add_folder("Reset View Direction"):
            btn_overview = self.server.gui.add_button("Overview")
            btn_front = self.server.gui.add_button("Front (+Z)")
            btn_back = self.server.gui.add_button("Back (-Z)")
            btn_top = self.server.gui.add_button("Top (-Y)")
            btn_left = self.server.gui.add_button("Left (-X)")
            btn_right = self.server.gui.add_button("Right (+X)")
            btn_first_cam = self.server.gui.add_button("First Camera")

        @btn_overview.on_click
        def _(_):
            d = np.array([0.5, -0.6, 0.6])
            self._reset_view_to_direction(d / np.linalg.norm(d))

        @btn_front.on_click
        def _(_):
            self._reset_view_to_direction(np.array([0.0, 0.0, 1.0]))

        @btn_back.on_click
        def _(_):
            self._reset_view_to_direction(np.array([0.0, 0.0, -1.0]))

        @btn_top.on_click
        def _(_):
            self._reset_view_to_direction(np.array([0.0, -1.0, 0.0]), up=np.array([0.0, 0.0, 1.0]))

        @btn_left.on_click
        def _(_):
            self._reset_view_to_direction(np.array([-1.0, 0.0, 0.0]))

        @btn_right.on_click
        def _(_):
            self._reset_view_to_direction(np.array([1.0, 0.0, 0.0]))

        @btn_first_cam.on_click
        def _(_):
            self._move_to_camera(0, smooth=True)

        # 4D/3D mode buttons
        button3 = self.server.gui.add_button("4D (Only Show Current Frame)")
        button4 = self.server.gui.add_button("3D (Show All Frames)")

        @button3.on_click
        def _(_):
            self.fourd = True

        @button4.on_click
        def _(_):
            self.fourd = False

        # Sliders
        self.focal_slider = self.server.gui.add_slider(
            "Focal Length", min=0.1, max=99999, step=1, initial_value=533
        )
        self.psize_slider = self.server.gui.add_slider(
            "Point Size", min=0.00001, max=0.1, step=0.00001, initial_value=self.point_size
        )
        self.camsize_slider = self.server.gui.add_slider(
            "Camera Size", min=0.01, max=0.5, step=0.01, initial_value=0.1
        )
        self.downsample_slider = self.server.gui.add_slider(
            "Downsample Factor", min=1, max=1000, step=1, initial_value=self.args.downsample_factor
        )
        self.show_camera_checkbox = self.server.gui.add_checkbox(
            "Show Camera", initial_value=self.show_camera
        )
        self.vis_threshold_slider = self.server.gui.add_slider(
            "Visibility Threshold", min=1.0, max=5.0, step=0.01,
            initial_value=self.vis_threshold,
        )
        self.camera_downsample_slider = self.server.gui.add_slider(
            "Camera Downsample Factor", min=1, max=50, step=1, initial_value=1
        )

        @self.psize_slider.on_update
        def _(_):
            for handle in self.pc_handles:
                handle.point_size = self.psize_slider.value
            if self.accumulated_pc_handle is not None:
                self.accumulated_pc_handle.point_size = self.psize_slider.value

        @self.camsize_slider.on_update
        def _(_):
            for handle in self.cam_handles:
                handle.scale = self.camsize_slider.value

        @self.downsample_slider.on_update
        def _(_):
            # For accumulated point cloud, regenerate with new settings
            if self.accumulated_points is not None and self.accumulated_pc_handle is not None:
                try:
                    self.accumulated_pc_handle.remove()
                except (KeyError, AttributeError):
                    pass
                
                # Apply downsample and threshold filters to accumulated points
                pred_pts = self.accumulated_points
                pc_color = self.accumulated_colors
                conf_val = self.accumulated_conf
                
                # Confidence threshold filter
                if conf_val is not None:
                    mask = conf_val > self.vis_threshold
                    pred_pts = pred_pts[mask]
                    pc_color = pc_color[mask]
                
                # Downsample
                downsample_factor = self.downsample_slider.value
                if downsample_factor > 1 and len(pred_pts) > 0:
                    indices = np.arange(0, len(pred_pts), downsample_factor)
                    pred_pts = pred_pts[indices]
                    pc_color = pc_color[indices]
                
                self.accumulated_pc_handle = self.server.scene.add_point_cloud(
                    name="/accumulated/point_cloud",
                    points=pred_pts,
                    colors=pc_color,
                    point_size=self.psize_slider.value,
                )

        @self.show_camera_checkbox.on_update
        def _(_):
            self.show_camera = self.show_camera_checkbox.value
            if self.show_camera:
                self._regenerate_cameras()
            else:
                for handle in self.cam_handles:
                    handle.visible = False

        @self.vis_threshold_slider.on_update
        def _(_):
            self.vis_threshold = self.vis_threshold_slider.value
            # Update accumulated point cloud with new threshold
            if self.accumulated_points is not None and self.accumulated_pc_handle is not None:
                try:
                    self.accumulated_pc_handle.remove()
                except (KeyError, AttributeError):
                    pass
                
                # Apply confidence threshold filter to accumulated points
                pred_pts = self.accumulated_points
                pc_color = self.accumulated_colors
                conf_val = self.accumulated_conf
                
                # Confidence threshold filter
                if conf_val is not None:
                    mask = conf_val > self.vis_threshold
                    pred_pts = pred_pts[mask]
                    pc_color = pc_color[mask]
                
                # Downsample
                downsample_factor = self.downsample_slider.value
                if downsample_factor > 1 and len(pred_pts) > 0:
                    indices = np.arange(0, len(pred_pts), downsample_factor)
                    pred_pts = pred_pts[indices]
                    pc_color = pc_color[indices]
                
                self.accumulated_pc_handle = self.server.scene.add_point_cloud(
                    name="/accumulated/point_cloud",
                    points=pred_pts,
                    colors=pc_color,
                    point_size=self.psize_slider.value,
                )

        @self.camera_downsample_slider.on_update
        def _(_):
            self._regenerate_cameras()

    def _compute_scene_center_and_scale(self):
        """Compute scene center and scale from camera positions"""
        if self.cam_dict is not None and "t" in self.cam_dict:
            cam_positions = np.array([self.cam_dict["t"][s] for s in self.all_steps])
            center = np.mean(cam_positions, axis=0)
            if len(cam_positions) > 1:
                extent = np.ptp(cam_positions, axis=0)
                scale = np.linalg.norm(extent)
            else:
                scale = 1.0
        else:
            center = np.zeros(3)
            scale = 1.0
        return center, max(scale, 0.1)

    def _reset_view_to_direction(self, direction, up=np.array([0.0, -1.0, 0.0]), distance_scale=1.5, smooth=True):
        """Reset the viewer camera to look at scene center from a given direction."""
        center, scale = self._compute_scene_center_and_scale()
        distance = scale * distance_scale
        position = center + direction * distance

        for client in self.server.get_clients().values():
            client.camera.up_direction = tuple(up)
            client.camera.position = tuple(position)
            client.camera.look_at = tuple(center)

    def _move_to_camera(self, frame_idx, smooth=True):
        """Move viewer camera to match reconstructed camera at given frame."""
        if self.cam_dict is None or frame_idx >= len(self.all_steps):
            return

        step = self.all_steps[frame_idx]
        R = self.cam_dict["R"][step] if "R" in self.cam_dict else np.eye(3)
        t = self.cam_dict["t"][step] if "t" in self.cam_dict else np.zeros(3)
        focal = self.cam_dict["focal"][step] if "focal" in self.cam_dict else 1.0
        pp = self.cam_dict["pp"][step] if "pp" in self.cam_dict else (1.0, 1.0)

        offset = 0.5
        viewing_dir = R[:, 2]
        position = t - viewing_dir * offset
        look_at = t + viewing_dir * 0.5
        up = -R[:, 1]

        for client in self.server.get_clients().values():
            client.camera.up_direction = tuple(up)
            client.camera.position = tuple(position)
            client.camera.look_at = tuple(look_at)

    def _connect_client(self, client):
        """Setup client connection callbacks."""
        pass

    def parse_pc_data(self, pc, color, conf=None, edge_color=[0.251, 0.702, 0.902], 
                      set_border_color=False, downsample_factor=1):
        """Parse and filter point cloud data (完全按照 PointCloudViewer)"""
        pred_pts = pc.reshape(-1, 3)
        pc_shape = pc.shape[:2]  # (H, W) of point cloud

        if set_border_color and edge_color is not None:
            color = self._set_color_border(color, color=edge_color)
        
        # Ensure color matches point cloud dimensions
        if color.shape[:2] != pc_shape:
            color = cv2.resize(color, (pc_shape[1], pc_shape[0]))
        
        if np.isnan(color).any():
            color = np.zeros((pred_pts.shape[0], 3))
            color[:, 2] = 1
        else:
            color = color.reshape(-1, 3)

        # Remove NaN / Inf points
        valid = np.isfinite(pred_pts).all(axis=1)
        if not valid.all():
            pred_pts = pred_pts[valid]
            color = color[valid]
            if conf is not None:
                conf = conf.reshape(-1)[valid]

        # Confidence threshold filter
        if conf is not None:
            conf_flat = conf.reshape(-1) if conf.ndim > 1 else conf
            mask = conf_flat > self.vis_threshold
            pred_pts = pred_pts[mask]
            color = color[mask]

        if len(pred_pts) == 0:
            return pred_pts, color

        # Downsample
        if downsample_factor > 1 and len(pred_pts) > 0:
            indices = np.arange(0, len(pred_pts), downsample_factor)
            pred_pts = pred_pts[indices]
            color = color[indices]

        return pred_pts, color

    @staticmethod
    def _set_color_border(image, border_width=5, color=[1, 0, 0]):
        """Add colored border to image."""
        image[:border_width, :, 0] = color[0]
        image[:border_width, :, 1] = color[1]
        image[:border_width, :, 2] = color[2]
        image[-border_width:, :, 0] = color[0]
        image[-border_width:, :, 1] = color[1]
        image[-border_width:, :, 2] = color[2]
        image[:, :border_width, 0] = color[0]
        image[:, :border_width, 1] = color[1]
        image[:, :border_width, 2] = color[2]
        image[:, -border_width:, 0] = color[0]
        image[:, -border_width:, 1] = color[1]
        image[:, -border_width:, 2] = color[2]
        return image

    def _regenerate_point_clouds(self):
        """Regenerate all point clouds with current settings."""
        for handle in self.pc_handles:
            try:
                handle.remove()
            except (KeyError, AttributeError):
                pass
        self.pc_handles.clear()

        for i, step in enumerate(self.all_steps):
            if step not in self.pcs:
                continue
            pc = self.pcs[step]["pc"]
            color = self.pcs[step]["color"]
            conf = self.pcs[step]["conf"]

            pred_pts, pc_color = self.parse_pc_data(
                pc, color, conf, set_border_color=True,
                downsample_factor=self.downsample_slider.value
            )

            handle = self.server.scene.add_point_cloud(
                name=f"/frames/{step}/pred_pts",
                points=pred_pts,
                colors=pc_color,
                point_size=self.psize_slider.value,
            )
            self.pc_handles.append(handle)

    def _regenerate_cameras(self):
        """Regenerate camera visualizations with current settings."""
        for handle in self.cam_handles:
            try:
                handle.remove()
            except (KeyError, AttributeError):
                pass
        self.cam_handles.clear()
        self.traj_list.clear()

        if self.show_camera and self.cam_dict is not None:
            downsample_factor = int(self.camera_downsample_slider.value)
            for i, step in enumerate(self.all_steps):
                if i % downsample_factor == 0:
                    self.add_camera(step)

    def add_camera(self, step):
        """Add camera visualization for a frame (完全按照 PointCloudViewer)"""
        cam = self.cam_dict
        if cam is None:
            return

        focal = cam["focal"][step] if "focal" in cam else 1.0
        pp = cam["pp"][step] if "pp" in cam else (1.0, 1.0)
        R = cam["R"][step] if "R" in cam else np.eye(3)
        t = cam["t"][step] if "t" in cam else np.zeros(3)

        q = tf.SO3.from_matrix(R).wxyz
        fov = 2 * np.arctan(pp[0] / focal)
        aspect = pp[0] / pp[1]
        self.traj_list.append((q, t))

        step_index = self.all_steps.index(step) if step in self.all_steps else 0
        camera_color = self.camera_colors[step_index]
        camera_color_rgb = tuple((camera_color[:3] * 255).astype(int))

        self.server.scene.add_frame(
            f"/frames/{step}/camera_frame",
            wxyz=q,
            position=t,
            axes_length=0.05,
            axes_radius=0.002,
            origin_radius=0.002,
        )

        frustum_handle = self.server.scene.add_camera_frustum(
            name=f"/frames/{step}/camera",
            fov=fov,
            aspect=aspect,
            wxyz=q,
            position=t,
            scale=self.camsize_slider.value,
            color=camera_color_rgb,
        )

        @frustum_handle.on_click
        def _(event):
            look_at_pt = t + R[:, 2] * 0.5
            up_dir = -R[:, 1]
            for client in self.server.get_clients().values():
                client.camera.up_direction = tuple(up_dir)
                client.camera.position = tuple(t)
                client.camera.look_at = tuple(look_at_pt)

        self.cam_handles.append(frustum_handle)

    def add_frame(self, world_points, colors, conf, extrinsic, intrinsic, image):
        """Add a new frame to the viewer - model maintains unified world coordinate system"""
        step = len(self.all_steps)
        
        # Compute c2w from extrinsic (already c2w from postprocess)
        c2w = np.eye(4)
        c2w[:3, :4] = extrinsic
        current_R = c2w[:3, :3]
        current_t = c2w[:3, 3]
        
        # Store point cloud data directly (model already in unified world coordinates)
        self.pcs[step] = {
            "pc": world_points,
            "color": colors,
            "conf": conf,
        }
        self.all_steps.append(step)
        
        # Store original image
        self.original_images.append(image)
        
        # Update camera dictionary (use original extrinsic, no transformation)
        if self.cam_dict is None:
            self.cam_dict = {"focal": [], "pp": [], "R": [], "t": []}
        
        self.cam_dict["focal"].append(intrinsic[0, 0])
        self.cam_dict["pp"].append((intrinsic[0, 2], intrinsic[1, 2]))
        self.cam_dict["R"].append(current_R)
        self.cam_dict["t"].append(current_t)
        
        # Update camera colors
        num_cameras = len(self.all_steps)
        if num_cameras > 1:
            normalized_indices = np.array(list(range(num_cameras))) / (num_cameras - 1)
        else:
            normalized_indices = np.array([0.0])
        cmap = cm.get_cmap('viridis')
        self.camera_colors = cmap(normalized_indices)
        
        # Create frame node
        if not hasattr(self, 'frame_base_node'):
            self.frame_base_node = self.server.scene.add_frame("/frames", show_axes=False)
        
        frame_node = self.server.scene.add_frame(f"/frames/{step}", show_axes=False)
        self.frame_nodes.append(frame_node)
        
        # Accumulate point cloud for persistent mapping
        pc = self.pcs[step]["pc"]
        color = self.pcs[step]["color"]
        conf_val = self.pcs[step]["conf"]

        pred_pts, pc_color = self.parse_pc_data(
            pc, color, conf_val, set_border_color=True,
            downsample_factor=self.downsample_slider.value
        )
        
        # Add to accumulated point cloud
        if self.accumulated_points is None:
            self.accumulated_points = pred_pts
            self.accumulated_colors = pc_color
            if conf_val is not None:
                conf_flat = conf_val.reshape(-1) if conf_val.ndim > 1 else conf_val
                self.accumulated_conf = conf_flat
        else:
            self.accumulated_points = np.concatenate([self.accumulated_points, pred_pts], axis=0)
            self.accumulated_colors = np.concatenate([self.accumulated_colors, pc_color], axis=0)
            if conf_val is not None and self.accumulated_conf is not None:
                conf_flat = conf_val.reshape(-1) if conf_val.ndim > 1 else conf_val
                self.accumulated_conf = np.concatenate([self.accumulated_conf, conf_flat], axis=0)
        
        # Update accumulated point cloud visualization
        if self.accumulated_pc_handle is not None:
            try:
                self.accumulated_pc_handle.remove()
            except (KeyError, AttributeError):
                pass
        
        self.accumulated_pc_handle = self.server.scene.add_point_cloud(
            name="/accumulated/point_cloud",
            points=self.accumulated_points,
            colors=self.accumulated_colors,
            point_size=self.psize_slider.value,
        )
        
        # Add camera for this frame
        if self.show_camera:
            self.add_camera(step)
        
        # Update frame slider
        if self.gui_timestep is not None:
            self.gui_timestep.max = len(self.all_steps) - 1
        
        # Update current frame image
        if self.current_frame_image is not None and len(self.original_images) > 0:
            self.current_frame_image.image = self.original_images[-1]
        
        # Update visibility based on mode
        self.update_frame_visibility()

    def update_frame_visibility(self):
        """Update frame visibility based on current timestep and mode."""
        if not hasattr(self, 'frame_nodes') or self.gui_timestep is None:
            return

        current_timestep = self.gui_timestep.value
        for i, frame_node in enumerate(self.frame_nodes):
            frame_node.visible = (
                i <= current_timestep if not self.fourd else i == current_timestep
            )

    def clear_all_frames(self):
        """Clear all frames and point clouds from the viewer (for replacement mode)."""
        # Clear accumulated point cloud
        if self.accumulated_pc_handle is not None:
            try:
                self.accumulated_pc_handle.remove()
            except (KeyError, AttributeError):
                pass
            self.accumulated_pc_handle = None
        
        # Reset accumulated data
        self.accumulated_points = None
        self.accumulated_colors = None
        self.accumulated_conf = None
        
        # Remove camera frustums
        for handle in self.cam_handles:
            try:
                handle.remove()
            except (KeyError, AttributeError):
                pass
        self.cam_handles = []
        
        # Remove frame nodes
        for node in self.frame_nodes:
            try:
                node.remove()
            except (KeyError, AttributeError):
                pass
        self.frame_nodes = []
        
        # Clear stored data
        self.pcs = {}
        self.cam_dict = None
        self.all_steps = []
        self.original_images = []
        self.traj_list = []
        
        # Reset frame slider
        if self.gui_timestep is not None:
            self.gui_timestep.max = 0
            self.gui_timestep.value = 0

    def setup_animation(self):
        """Setup animation controls."""
        with self.server.gui.add_folder("Playback"):
            self.gui_timestep = self.server.gui.add_slider(
                "Train Step", min=0, max=max(0, len(self.all_steps) - 1), step=1, initial_value=0, disabled=False
            )
            gui_next_frame = self.server.gui.add_button("Next Step", disabled=False)
            gui_prev_frame = self.server.gui.add_button("Prev Step", disabled=False)
            gui_playing = self.server.gui.add_checkbox("Playing", True)
            gui_framerate = self.server.gui.add_slider("FPS", min=1, max=60, step=0.1, initial_value=10)

        @gui_next_frame.on_click
        def _(_):
            if self.gui_timestep is not None:
                self.gui_timestep.value = (self.gui_timestep.value + 1) % len(self.all_steps)

        @gui_prev_frame.on_click
        def _(_):
            if self.gui_timestep is not None:
                self.gui_timestep.value = (self.gui_timestep.value - 1) % len(self.all_steps)

        @gui_playing.on_update
        def _(_):
            if self.gui_timestep is not None:
                self.gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

        @self.gui_timestep.on_update
        def _(_):
            if self.gui_timestep is None:
                return
            current_timestep = self.gui_timestep.value
            
            # Update current frame image
            if self.current_frame_image is not None and current_timestep < len(self.original_images):
                self.current_frame_image.image = self.original_images[current_timestep]

        # Animation loop
        def animate_loop():
            while True:
                if self.gui_timestep is not None and gui_playing.value:
                    self.gui_timestep.value = (self.gui_timestep.value + 1) % len(self.all_steps)
                self.update_frame_visibility()
                time.sleep(1.0 / gui_framerate.value)
        
        animate_thread = threading.Thread(target=animate_loop, daemon=True)
        animate_thread.start()


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


def camera_capture_thread(cap, buffer, fps_list):
    """Camera capture thread with dynamic FPS"""
    frame_count = 0
    fps_idx = 0  # Index into fps_list
    frames_until_switch = fps_list[0][1]  # Frames to capture at current FPS
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_fps = fps_list[fps_idx][0]
        frame_interval = max(1, round(30 / current_fps))
        
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buffer.put(frame_rgb)
            frames_until_switch -= 1
            
            # Switch to next FPS level if needed
            if frames_until_switch <= 0 and fps_idx < len(fps_list) - 1:
                fps_idx += 1
                frames_until_switch = fps_list[fps_idx][1]
        
        frame_count += 1


def process_camera_stream(model, device, dtype, args):
    """Process camera stream with LingBot-MAP inference"""
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera_id}")
    
    print(f"Camera FPS: 30, Target processing FPS: {args.fps}")
    
    # Initialize frame buffer
    frame_buffer = FrameBuffer(max_size=args.kv_cache_sliding_window)
    
    # Start camera capture thread with dynamic FPS
    # Format: [(fps, num_frames), ...] - num_frames=0 means infinite
    fps_list = [
        (10, 200),  # First 200 frames at 10 FPS
        (3, 0),     # After that, 3 FPS indefinitely
    ]
    capture_thread = threading.Thread(
        target=camera_capture_thread,
        args=(cap, frame_buffer, fps_list),
        daemon=True
    )
    capture_thread.start()
    
    # Initialize viewer
    viewer = RealTimeViewer(args)
    print(f"Viewer server started on port {args.port}")
    print(f"Open http://localhost:{args.port} in your browser")
    
    # Main processing loop
    print("Starting real-time camera streaming...")
    
    # Collect initial scale frames
    scale_frames = []
    scale_images = []
    
    print(f"Collecting {args.num_scale_frames} scale frames...")
    while len(scale_frames) < args.num_scale_frames:
        frame = frame_buffer.get(block=True, timeout=5.0)
        if frame is None:
            print("Timeout waiting for scale frames")
            cap.release()
            return
        
        img_tensor = preprocess_image(frame, mode="crop", 
                                     image_size=args.image_size, 
                                     patch_size=args.patch_size)
        scale_frames.append(img_tensor)
        scale_images.append(frame)
        print(f"  Collected {len(scale_frames)}/{args.num_scale_frames}")
    
    # Stack scale frames: convert to [B, S, C, H, W] format
    scale_batch = torch.stack(scale_frames, dim=0).unsqueeze(0).to(device)
    
    # Process scale frames using model.inference_streaming (like demo.py)
    model.clean_kv_cache()
    
    print("Processing scale frames...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        predictions = model.inference_streaming(
            scale_batch,
            num_scale_frames=args.num_scale_frames,
            keyframe_interval=args.keyframe_interval,
        )
    
    # Post-process (like demo.py)
    images_for_post = torch.stack([
        torch.from_numpy(cv2.resize(f, (args.image_size, args.image_size))).permute(2, 0, 1)
        for f in scale_images
    ]).float() / 255.0
    
    predictions, images_cpu = postprocess(predictions, images_for_post)
    vis_predictions = prepare_for_visualization(predictions, images_cpu)
    
    # Extract data for visualization
    world_points = vis_predictions.get("world_points", vis_predictions.get("depth"))
    depth_conf = vis_predictions.get("depth_conf")
    extrinsic = vis_predictions["extrinsic"]
    intrinsic = vis_predictions["intrinsic"]
    images = vis_predictions["images"]
    
    # For scale frames, use depth map if world_points is not available
    # Check if it's depth (shape [..., 1]) vs already unprojected world_points (shape [..., 3])
    if world_points.ndim == 4 and world_points.shape[-1] == 1:  # depth map
        world_points = unproject_depth_map_to_point_map(
            torch.from_numpy(world_points),
            torch.from_numpy(extrinsic),
            torch.from_numpy(intrinsic)
        )
    
    # Add scale frames to viewer
    for i in range(args.num_scale_frames):
        viewer.add_frame(
            world_points[i],
            images[i].transpose(1, 2, 0),  # (H, W, 3)
            depth_conf[i] if depth_conf is not None else None,
            extrinsic[i],
            intrinsic[i],
            (images[i] * 255).astype(np.uint8).transpose(1, 2, 0)
        )
    
    # Setup animation after scale frames are added
    viewer.setup_animation()
    
    # ========== Periodic batch processing with KV cache reset ==========
    # Strategy:
    # 1. First batch: frames 0-199 (200 frames)
    # 2. Subsequent batches: frames N-100 to N+100 (200 frames with 100 overlap)
    # 3. Reset KV cache before each batch for fresh inference
    # 4. Replace point cloud instead of accumulating
    # 5. Use inference_streaming with output_device=cpu for memory optimization
    
    batch_size = 200  # Process 200 frames per batch
    overlap_size = 100  # Overlap 100 frames with previous batch
    new_frames_per_batch = batch_size - overlap_size  # 100 new frames per batch
    
    print(f"Starting periodic batch processing (batch={batch_size}, overlap={overlap_size}, new={new_frames_per_batch})...")
    print(f"Current GPU memory usage before processing: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Buffer to hold the sliding window
    sliding_window = []
    batch_idx = 0
    
    while True:
        # Collect frames for this batch
        print(f"\n=== Batch {batch_idx + 1} ===")
        
        # For first batch: collect full batch size
        # For subsequent batches: keep overlap frames + collect new frames
        if batch_idx == 0:
            needed_frames = batch_size
            print(f"Collecting {needed_frames} frames (first batch)...")
        else:
            needed_frames = batch_size - overlap_size
            print(f"Collecting {needed_frames} new frames (keeping {overlap_size} from previous batch)...")
        
        # Collect new frames
        while len(sliding_window) < batch_size:
            try:
                frame = frame_buffer.get(block=True, timeout=5.0)
                if frame is not None:
                    sliding_window.append(frame)
                    current_count = len(sliding_window)
                    if batch_idx == 0:
                        if current_count % 50 == 0:
                            print(f"  Collected {current_count}/{batch_size}")
                    else:
                        if current_count % 25 == 0:
                            print(f"  Collected {current_count - overlap_size}/{needed_frames} new frames")
            except queue.Empty:
                if len(sliding_window) > 0:
                    print(f"  Timeout, using {len(sliding_window)} frames")
                    break
                continue
        
        # Skip if not enough frames for first batch
        if len(sliding_window) < batch_size and batch_idx == 0:
            print(f"Waiting for more frames... ({len(sliding_window)}/{batch_size})")
            continue
        
        print(f"Processing batch of {len(sliding_window)} frames...")
        print(f"GPU memory before processing batch: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Clean KV cache before each batch (fresh start)
        model.clean_kv_cache()
        torch.cuda.empty_cache()
        
        # Preprocess the batch
        batch_tensors = []
        batch_resized = []
        
        for frame in sliding_window:
            img_tensor = preprocess_image(frame, mode="crop",
                                        image_size=args.image_size,
                                        patch_size=args.patch_size)
            img_tensor = img_tensor.to(device)
            batch_tensors.append(img_tensor)
            batch_resized.append(cv2.resize(frame, (args.image_size, args.image_size)))
        
        # Stack to [B, S, C, H, W]
        batch_tensor = torch.stack(batch_tensors, dim=0).unsqueeze(0)
        del batch_tensors
        
        # Process batch with streaming inference (fresh KV cache)
        # Use output_device=cpu to offload predictions during inference (critical for memory)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            batch_output = model.inference_streaming(
                batch_tensor,
                num_scale_frames=min(4, len(sliding_window)),
                keyframe_interval=args.keyframe_interval,
                output_device=torch.device("cpu"),  # Offload to CPU to prevent GPU OOM
            )
        
        print(f"GPU memory after inference: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Post-process
        images_for_post = torch.stack([
            torch.from_numpy(cv2.resize(f, (args.image_size, args.image_size))).permute(2, 0, 1)
            for f in batch_resized
        ]).float() / 255.0
        
        batch_output, _ = postprocess(batch_output, images_for_post)
        vis_output = prepare_for_visualization(batch_output, images_for_post)
        
        # Extract data
        world_points = vis_output.get("world_points", vis_output.get("depth"))
        depth_conf = vis_output.get("depth_conf")
        extrinsic = vis_output["extrinsic"]
        intrinsic = vis_output["intrinsic"]
        images = vis_output["images"]
        
        # Clear previous point cloud (REPLACE instead of accumulate)
        viewer.clear_all_frames()
        
        # Add all frames from current batch to viewer
        for i in range(len(sliding_window)):
            pts = world_points[i]
            conf = depth_conf[i] if depth_conf is not None else None
            ext = extrinsic[i]
            intr = intrinsic[i]
            img = images[i]
            
            # Unproject if needed
            if pts.ndim == 3 and pts.shape[-1] == 1:
                pts = unproject_depth_map_to_point_map(
                    torch.from_numpy(pts).unsqueeze(0),
                    torch.from_numpy(ext).unsqueeze(0),
                    torch.from_numpy(intr).unsqueeze(0)
                )[0]
            
            # Add to viewer
            viewer.add_frame(
                pts,
                img.transpose(1, 2, 0),
                conf,
                ext,
                intr,
                (img * 255).astype(np.uint8).transpose(1, 2, 0)
            )
        
        # Update sliding window: keep last overlap_size frames for next batch
        if len(sliding_window) >= overlap_size:
            sliding_window = sliding_window[-overlap_size:]
        
        batch_idx += 1
        
        # === AGGRESSIVE MEMORY CLEANUP ===
        model.clean_kv_cache()
        
        # Delete all temporary tensors immediately after use
        try:
            del window_tensor
        except:
            pass
        try:
            del window_output
        except:
            pass
        try:
            del images_for_post
        except:
            pass
        try:
            del vis_window
        except:
            pass
        try:
            del world_points
        except:
            pass
        try:
            del depth_conf
        except:
            pass
        try:
            del extrinsic
        except:
            pass
        try:
            del intrinsic
        except:
            pass
        try:
            del images
        except:
            pass
        try:
            del pts
        except:
            pass
        try:
            del conf
        except:
            pass
        try:
            del ext
        except:
            pass
        try:
            del intr
        except:
            pass
        try:
            del img
        except:
            pass
        
        # Force garbage collection and GPU memory cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Print progress every batch
        print(f"Completed batch {batch_idx}, total frames processed: {batch_idx * new_frames_per_batch + batch_size}")


# =============================================================================
# Frontend Stream Processing (复用 process_camera_stream 的流式逻辑)
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
    
    @app.websocket("/ws")
    async def websocket_stream(websocket: WebSocket):
        """WebSocket 流式接口 - 批量上传模式（参考 viser_wrapper 逻辑）"""
        await websocket.accept()
        print("🔌 WebSocket 客户端已连接")
        
        BATCH_SIZE = 200
        batch_frames = []
        is_processing = False
        CONF_THRESHOLD_PERCENT = 50.0  # 与 viser_wrapper 一致
        
        try:
            while True:
                data = await websocket.receive_json()
                frame_base64 = data.get('frame')
                frame_id = data.get('frame_id', 0)
                is_batch_end = data.get('batch_end', False)
                
                if frame_base64:
                    try:
                        import base64
                        frame_data = base64.b64decode(frame_base64.split(',')[1] if ',' in frame_base64 else frame_base64)
                        
                        if not frame_data or len(frame_data) == 0:
                            print(f"❌ 帧 #{frame_id} 数据为空，跳过")
                        else:
                            import numpy as np
                            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                            
                            if frame_array.size == 0:
                                print(f"❌ 帧 #{frame_id} 数组为空，跳过")
                            else:
                                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                                
                                if frame is None:
                                    print(f"❌ 帧 #{frame_id} 解码失败，跳过")
                                else:
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    batch_frames.append(frame_rgb)
                                    print(f"🔹 收到帧 #{frame_id}，当前队列: {len(batch_frames)}/{BATCH_SIZE}")
                    except Exception as e:
                        print(f"❌ 接收帧失败: {type(e).__name__}: {e}")
                        await websocket.send_json({"type": "error", "msg": str(e)})
                
                if len(batch_frames) >= BATCH_SIZE or (is_batch_end and len(batch_frames) > 0):
                    if is_processing:
                        await websocket.send_json({"type": "error", "msg": "正在处理上一批，请等待"})
                        continue
                    
                    is_processing = True
                    batch_size = len(batch_frames)
                    print(f"🚀 开始批量处理 {batch_size} 帧...")
                    await websocket.send_json({"type": "processing_start", "frame_count": batch_size})
                    
                    try:
                        batch_tensors = []
                        batch_resized = []
                        
                        for frame in batch_frames:
                            img_tensor = preprocess_image(frame, mode="crop",
                                                        image_size=image_size,
                                                        patch_size=patch_size)
                            img_tensor = img_tensor.to(device)
                            batch_tensors.append(img_tensor)
                            batch_resized.append(cv2.resize(frame, (image_size, image_size)))
                        
                        batch_tensor = torch.stack(batch_tensors, dim=0).unsqueeze(0)
                        del batch_tensors
                        
                        model.clean_kv_cache()
                        torch.cuda.empty_cache()
                        
                        print("📊 正在进行模型推理...")
                        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                            batch_output = model.inference_streaming(
                                batch_tensor,
                                num_scale_frames=10,
                                keyframe_interval=2,
                            )
                        
                        images_for_post = torch.stack([
                            torch.from_numpy(f).permute(2, 0, 1) for f in batch_resized
                        ]).float() / 255.0
                        
                        predictions, images_cpu = postprocess(batch_output, images_for_post)
                        vis_predictions = prepare_for_visualization(predictions, images_cpu)
                        
                        world_points_map = vis_predictions.get("world_points")
                        conf_map = vis_predictions.get("world_points_conf")
                        depth_map = vis_predictions.get("depth")
                        depth_conf = vis_predictions.get("depth_conf")
                        extrinsic = vis_predictions["extrinsic"]
                        intrinsic = vis_predictions["intrinsic"]
                        
                        if isinstance(extrinsic, torch.Tensor):
                            extrinsic = extrinsic.cpu().numpy()
                        if isinstance(intrinsic, torch.Tensor):
                            intrinsic = intrinsic.cpu().numpy()
                        
                        use_point_map = world_points_map is not None
                        if not use_point_map:
                            world_points = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
                            conf = depth_conf
                        else:
                            world_points = world_points_map
                            conf = conf_map
                        
                        if isinstance(world_points, torch.Tensor):
                            world_points = world_points.cpu().numpy()
                        if isinstance(conf, torch.Tensor):
                            conf = conf.cpu().numpy()
                        
                        colors = images_cpu.permute(0, 2, 3, 1).cpu().numpy()
                        
                        S, H, W = world_points.shape[:3]
                        points_all = world_points.reshape(-1, 3)
                        colors_cropped = colors[:S, :H, :W, :]
                        colors_all = (colors_cropped.reshape(-1, 3) * 255).astype(np.uint8)
                        conf_all = conf.reshape(-1)
                        
                        threshold_val = np.percentile(conf_all, CONF_THRESHOLD_PERCENT)
                        conf_mask = (conf_all >= threshold_val) & (conf_all > 0.1)
                        
                        points_filtered = points_all[conf_mask]
                        colors_filtered = colors_all[conf_mask]
                        
                        scene_center = np.mean(points_filtered, axis=0)
                        points_centered = points_filtered - scene_center
                        
                        cam_to_world_mat = closed_form_inverse_se3(extrinsic)
                        cam_to_world = cam_to_world_mat[:, :3, :]
                        cam_to_world[..., -1] -= scene_center
                        
                        frame_indices = np.repeat(np.arange(S), H * W)[conf_mask]
                        
                        for i in range(S):
                            frame_mask = frame_indices == i
                            frame_points = points_centered[frame_mask]
                            frame_colors = colors_filtered[frame_mask]
                            
                            max_points = 5000
                            if len(frame_points) > max_points:
                                indices = np.linspace(0, len(frame_points) - 1, max_points, dtype=int)
                                frame_points = frame_points[indices]
                                frame_colors = frame_colors[indices]
                            
                            points = frame_points.flatten().tolist()
                            colors = frame_colors.flatten().tolist()
                            
                            ply_filename = f"batch_{frame_id}_{i}.ply"
                            save_point_cloud_ply_with_color(points, colors, ply_filename)
                            
                            cam_pose = {
                                'x': float(cam_to_world[i, 0, 3]),
                                'y': float(cam_to_world[i, 1, 3]),
                                'z': float(cam_to_world[i, 2, 3])
                            }
                            
                            print(f"📤 发送帧 {i} 的点云: {len(points)//3} 个点")
                            await websocket.send_json({
                                'type': 'result',
                                'frame_id': f'{frame_id}_{i}',
                                'point_count': len(points) // 3,
                                'point_cloud': points,
                                'point_colors': colors,
                                'point_cloud_url': f'/point_cloud/{ply_filename}',
                                'camera_pose': cam_pose,
                                'scene_center': scene_center.tolist()
                            })
                        
                        await websocket.send_json({"type": "processing_done", "frame_count": batch_size})
                        print(f"✅ 批量处理完成，共 {batch_size} 帧")
                        
                    except Exception as e:
                        print(f"❌ 批量处理失败: {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exc()
                        await websocket.send_json({"type": "error", "msg": str(e)})
                    
                    batch_frames = []
                    is_processing = False
        
        except WebSocketDisconnect:
            print("🔌 WebSocket 客户端已断开")
        except Exception as e:
            print(f"❌ WebSocket 错误: {type(e).__name__}: {e}")
    
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
    
    import uvicorn
    print(f"🚀 HTTP API 服务启动在 http://localhost:{http_port}")
    print(f"   测试接口: GET /test/info")
    print(f"   上传接口: POST /upload")
    print(f"   流式接口: WebSocket /ws/stream")
    uvicorn.run(app, host="0.0.0.0", port=http_port)

def main():
    parser = argparse.ArgumentParser(description="LingBot-MAP Real-time Streaming Demo")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="camera", choices=["camera", "api"], 
                        help="运行模式: camera(摄像头实时流) 或 api(HTTP接口)")
    
    # Input
    parser.add_argument("--camera_id", type=int, default=0, help="Webcam device ID")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    
    # Output
    parser.add_argument("--port", type=int, default=8080, help="Viser server port (camera mode)")
    parser.add_argument("--http_port", type=int, default=8000, help="HTTP API port (api mode)")
    
    # Inference
    parser.add_argument("--fps", type=int, default=3, help="Target processing FPS")
    parser.add_argument("--image_size", type=int, default=518, help="Image size")
    parser.add_argument("--patch_size", type=int, default=14, help="Patch size")
    parser.add_argument("--num_scale_frames", type=int, default=4, help="Number of scale frames")
    parser.add_argument("--keyframe_interval", type=int, default=10, help="Keyframe interval")
    parser.add_argument("--kv_cache_sliding_window", "--window_size", type=int, default=300, help="KV cache sliding window size")
    
    # Visualization
    parser.add_argument("--point_size", type=float, default=0.005, help="Point size")
    parser.add_argument("--downsample_factor", type=int, default=10, help="Point cloud downsample factor")
    parser.add_argument("--conf_threshold", type=float, default=1.5, help="Confidence threshold for visualization")
    
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
    
    # Run in selected mode
    if args.mode == "camera":
        print(f"📸 启动摄像头模式，Viser 服务端口: {args.port}")
        process_camera_stream(model, device, dtype, args)
    else:
        print(f"🌐 启动 API 模式，HTTP 服务端口: {args.http_port}")
        start_http_server(model, device, dtype, args.image_size, args.patch_size, args.http_port)


if __name__ == "__main__":
    main()
