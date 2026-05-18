import os

import cv2
import numpy as np
import time
import threading
from flask import Flask, jsonify
from waitress import serve

from keyframe import frameDiff_Flow
from keyframe.fine_detector import KeyFrameAnalyzer


def extract_keyframe_segments(video_path, diff_thresh=6, flow_thresh=1.0, buffer_frame=15,
                              min_recording_time=3, frame_interval=1, verbose=False):
    """
    从视频中提取关键帧片段（基于帧差和光流运动检测）

    Args:
        video_path: 输入视频文件路径
        diff_thresh: 帧差阈值，超过此阈值认为有运动（默认6）
        flow_thresh: 光流运动幅度阈值，超过此阈值认为有真实运动（默认1.0）
        buffer_frame: 事件结束缓冲帧数（默认15）
        min_recording_time: 最少录制时间（秒）（默认3）
        frame_interval: 帧间间隔，每隔多少帧处理一次（默认1）

    Returns:
        tuple: (keyframe_segments, total_frames)
            - keyframe_segments: 关键帧片段列表，每个元素为元组 (start_frame, end_frame)
            - total_frames: 视频总帧数
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return []

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频信息: {width}x{height} @ {fps}fps, 共 {total_frames} 帧")

    # LK光流参数
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 初始化前一帧
    ret, prev_frame = cap.read()
    if not ret:
        print("错误：无法读取视频帧")
        cap.release()
        return []

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100,
                                       qualityLevel=0.3, minDistance=7, blockSize=7)

    # 状态变量
    recording = False  # 是否正在录制
    buffer_count = 0  # 结束缓冲计数器
    segment_start_frame = 0  # 当前片段起始帧号
    keyframe_segments = []  # 存储关键帧片段 [(start_frame, end_frame), ...]

    frame_idx = 0
    while cap.isOpened():
        ret, curr_frame = cap.read()
        if not ret:
            break

        # 按帧间隔处理
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 1. 帧差粗筛
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        diff_mean = frame_diff.mean()

        # 2. 光流运动计算（仅在帧差超过阈值时进行）
        motion_magnitude = 0
        if diff_mean > diff_thresh and prev_pts is not None:
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None, **lk_params)
            if curr_pts is not None:
                good_new = curr_pts[status == 1]
                good_old = prev_pts[status == 1]
                if len(good_new) > 0:
                    motion_magnitude = np.mean(np.linalg.norm(good_new - good_old, axis=1))

        # 事件触发逻辑
        if diff_mean > diff_thresh and motion_magnitude > flow_thresh:
            buffer_count = 0
            if not recording:
                recording = True
                segment_start_frame = frame_idx
                start_time = frame_idx / fps
                if verbose:
                    print(f"【事件触发】开始录制，帧号: {segment_start_frame}, 时间戳: {start_time:.2f}s")

        else:
            if recording:
                buffer_count += 1

                if buffer_count >= buffer_frame:
                    elapsed_frames = frame_idx - segment_start_frame
                    elapsed_time = elapsed_frames / fps

                    if elapsed_time >= min_recording_time:
                        # 结束当前片段
                        segment_end_frame = frame_idx
                        keyframe_segments.append((segment_start_frame, segment_end_frame))
                        recording = False
                        buffer_count = 0
                        if verbose:
                            print(f"【事件结束】停止录制，帧号: {segment_end_frame}, 时长: {elapsed_time:.2f}s, "
                                  f"帧范围: [{segment_start_frame}, {segment_end_frame}]")
                    else:
                        # 未达到最少录制时间，继续录制
                        buffer_count = buffer_frame - 1

        # 更新帧与角点
        prev_gray = curr_gray.copy()
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100,
                                           qualityLevel=0.3, minDistance=7, blockSize=7)

        frame_idx += 1

    cap.release()

    # 检查是否有未结束的片段
    if recording:
        segment_end_frame = frame_idx - 1
        elapsed_time = (segment_end_frame - segment_start_frame) / fps
        if elapsed_time >= min_recording_time:
            keyframe_segments.append((segment_start_frame, segment_end_frame))
            print(f"【视频结束】结束录制，帧范围: [{segment_start_frame}, {segment_end_frame}]")

    # 计算选择的关键帧片段的总帧数
    selected_total_frames = sum(end - start + 1 for start, end in keyframe_segments)

    print(f"关键帧提取完成，共 {len(keyframe_segments)} 个片段")
    for i, (start, end) in enumerate(keyframe_segments):
        duration = (end - start) / fps
        print(f"  片段 {i + 1}: 帧 [{start}, {end}], 时长 {duration:.2f}s")
    print(f"选择的关键帧片段总帧数: {selected_total_frames}")

    return keyframe_segments, selected_total_frames


def extract_keyframe_timestamps(video_path, diff_thresh=6, flow_thresh=1.0, buffer_frame=15,
                                min_recording_time=3, frame_interval=1):
    """
    从视频中提取关键帧的时间戳列表（基于帧差和光流运动检测）

    Args:
        video_path: 输入视频文件路径
        diff_thresh: 帧差阈值，超过此阈值认为有运动（默认6）
        flow_thresh: 光流运动幅度阈值，超过此阈值认为有真实运动（默认1.0）
        buffer_frame: 事件结束缓冲帧数（默认15）
        min_recording_time: 最少录制时间（秒）（默认3）
        frame_interval: 帧间间隔，每隔多少帧处理一次（默认1）

    Returns:
        list: 提取的关键帧时间戳列表（单位：秒）
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return []

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频信息: {width}x{height} @ {fps}fps, 共 {total_frames} 帧")

    # LK光流参数
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 初始化前一帧
    ret, prev_frame = cap.read()
    if not ret:
        print("错误：无法读取视频帧")
        cap.release()
        return []

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100,
                                       qualityLevel=0.3, minDistance=7, blockSize=7)

    # 状态变量
    recording = False  # 是否正在录制
    buffer_count = 0  # 结束缓冲计数器
    start_time = 0  # 开始录制时间
    keyframe_timestamps = []  # 存储关键帧时间戳

    frame_idx = 0
    while cap.isOpened():
        ret, curr_frame = cap.read()
        if not ret:
            break

        # 按帧间隔处理
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 1. 帧差粗筛
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        diff_mean = frame_diff.mean()

        # 2. 光流运动计算（仅在帧差超过阈值时进行）
        motion_magnitude = 0
        if diff_mean > diff_thresh and prev_pts is not None:
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None, **lk_params)
            if curr_pts is not None:
                good_new = curr_pts[status == 1]
                good_old = prev_pts[status == 1]
                if len(good_new) > 0:
                    motion_magnitude = np.mean(np.linalg.norm(good_new - good_old, axis=1))

        # 事件触发逻辑
        current_timestamp = frame_idx / fps

        if diff_mean > diff_thresh and motion_magnitude > flow_thresh:
            buffer_count = 0
            if not recording:
                recording = True
                start_time = current_timestamp
                print(f"【事件触发】开始录制，时间戳: {start_time:.2f}s")

            # 记录关键帧时间戳
            keyframe_timestamps.append(current_timestamp)

        else:
            if recording:
                buffer_count += 1
                # 继续记录时间戳
                keyframe_timestamps.append(current_timestamp)

                elapsed_time = current_timestamp - start_time

                if buffer_count >= buffer_frame and elapsed_time >= min_recording_time:
                    recording = False
                    buffer_count = 0
                    print(f"【事件结束】停止录制，时间戳: {current_timestamp:.2f}s, 时长: {elapsed_time:.2f}s")
                elif buffer_count >= buffer_frame and elapsed_time < min_recording_time:
                    buffer_count = buffer_frame - 1

        # 更新帧与角点
        prev_gray = curr_gray.copy()
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100,
                                           qualityLevel=0.3, minDistance=7, blockSize=7)

        frame_idx += 1

    cap.release()

    # 去重并排序时间戳
    keyframe_timestamps = sorted(list(set(keyframe_timestamps)))

    print(f"关键帧提取完成，共 {len(keyframe_timestamps)} 个关键帧时间戳")

    return keyframe_timestamps


class EventBasedRecorder:
    def __init__(self, camera_index=0, width=1920, height=1080, fps=30,
                 output_dir="./recordings/", keyframe_dir="stage1_filter", diff_thresh=6, flow_thresh=1.0, min_interval=5,
                 buffer_frame=15, min_recording_time=3, min_conf=0.5, weight_ths=15, show_preview=True, use_fine_filter=False):
        """
        初始化事件触发式摄像头录制器

        Args:
            camera_index: 摄像头索引，默认0为默认摄像头
            width: 摄像头宽度，默认1920
            height: 摄像头高度，默认1080
            fps: 摄像头帧率，默认25
            output_dir: 输出目录，默认./recordings/
            diff_thresh: 差异阈值，默认8
            flow_thresh: 光流阈值，默认1.0
            buffer_frame: 事件结束缓冲帧数，默认15
            min_recording_time: 最少录制时间（秒），默认3
            show_preview: 是否显示实时预览画面，默认为True
            use_fine_filter: 是否使用语义筛选对粗筛选的结果进行筛选，默认为False
        """
        # 生活场景事件触发参数
        self.diff_thresh = diff_thresh
        self.flow_thresh = flow_thresh
        self.min_interval = min_interval  # 最小间隔（帧数），默认5
        self.buffer_frame = buffer_frame  # 事件结束缓冲帧数
        self.min_recording_time = min_recording_time  # 最少录制时间（秒）
        self.min_conf = min_conf  # 最小置信度（用于语义筛选）
        self.weight_ths = weight_ths  # 权重阈值（用于语义筛选）
        self.output_dir = output_dir  # 输出目录
        self.keyframe_dir = keyframe_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.show_preview = show_preview
        self.stop_event = threading.Event()  # 用于停止录制的事件
        self.is_running = False  # 记录是否正在运行
        self.use_fine_filter = use_fine_filter

        # LK光流参数
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # 初始化摄像头
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # 获取实际摄像头参数
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 状态机变量
        self.recording = False  # 是否正在录制
        self.buffer_count = 0  # 结束缓冲计数器
        self.recording_frames = []  # 当前录制的帧列表（内存中）
        self.recording_timestamps = []  # 与 recording_frames 对应的时间戳列表
        self.global_cur_timestamp = None
        self.start_time = 0  # 开始录制时间

        # 初始化前一帧
        ret, prev_frame = self.cap.read()
        if not ret:
            raise Exception("无法读取摄像头帧")
        self.prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=100,
                                                qualityLevel=0.3, minDistance=7, blockSize=7)

        print(f"摄像头初始化成功: {self.width}x{self.height} @ {self.fps}fps")
        print("系统进入静止休眠模式，仅进行帧差实时监测...")

        # 初始化关键帧分析器
        self.analyzer = None
        self.analyzer = KeyFrameAnalyzer()

    def start_fine_filter(self):
        self.use_fine_filter = True

    def stop_fine_filter(self):
        self.use_fine_filter = False

    def start_recording(self):
        """
        开始事件触发式录制
        """
        self.is_running = True
        self.stop_event.clear()  # 清除停止事件

        # 检查是否在主线程中运行
        import threading
        is_main_thread = threading.current_thread() is threading.main_thread()

        # 如果在后台线程中运行，强制禁用预览
        if not is_main_thread:
            print("在后台线程中运行，自动禁用预览")
            self.show_preview = False

        while not self.stop_event.is_set():
            ret, curr_frame = self.cap.read()
            if not ret:
                print("无法读取摄像头帧，退出")
                break

            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            self._process_frame(curr_frame, curr_gray)

            # 显示实时画面（如果启用且在主线程中）
            if self.show_preview and is_main_thread:
                cv2.imshow('Camera', curr_frame)
                # 按 'q' 键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self._cleanup()
        self.is_running = False

    def stop_recording(self):
        """
        停止录制
        """
        if self.is_running:
            print("【API调用】停止录制")
            self.stop_event.set()  # 设置停止事件
            return True
        else:
            print("【API调用】录制未运行，无需停止")
            return False

    def _process_frame(self, curr_frame, curr_gray):
        """
        处理每一帧，检测运动并控制录制

        Args:
            curr_frame: 当前彩色帧
            curr_gray: 当前灰度帧
        """
        # 1. 帧差粗筛
        frame_diff = cv2.absdiff(self.prev_gray, curr_gray)
        diff_mean = frame_diff.mean()

        # 2. 光流运动计算（仅在帧差超过阈值时进行，节省计算资源）
        motion_magnitude = 0
        if diff_mean > self.diff_thresh:
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, curr_gray, self.prev_pts, None, **self.lk_params)
            if curr_pts is not None:
                good_new = curr_pts[status == 1]
                good_old = self.prev_pts[status == 1]
                if len(good_new) > 0:
                    motion_magnitude = np.mean(np.linalg.norm(good_new - good_old, axis=1))

        # 事件触发状态机核心逻辑
        self._handle_recording(curr_frame, diff_mean, motion_magnitude)

        # 更新帧与角点
        self.prev_gray = curr_gray.copy()
        self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=100,
                                                qualityLevel=0.3, minDistance=7, blockSize=7)

    def _handle_recording(self, curr_frame, diff_mean, motion_magnitude):
        """
        处理录制逻辑

        Args:
            curr_frame: 当前帧
            diff_mean: 帧差均值
            motion_magnitude: 运动幅度
        """
        # 条件：检测到真实运动 → 唤醒开始录制
        if diff_mean > self.diff_thresh and motion_magnitude > self.flow_thresh:
            self.buffer_count = 0  # 重置缓冲
            if not self.recording:
                self._start_new_recording()

            # 正在录制：保存当前帧及时间戳到列表
            self.recording_frames.append(curr_frame.copy())
            self.recording_timestamps.append(time.time())
            # 在画面上显示录制状态
            cv2.putText(curr_frame, "RECORDING", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 静止状态：未触发运动
        else:
            if self.recording:
                self.buffer_count += 1
                # 依旧先保存帧及时间戳，维持缓冲
                self.recording_frames.append(curr_frame.copy())
                self.recording_timestamps.append(time.time())
                # 在画面上显示录制状态
                cv2.putText(curr_frame, "RECORDING", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 计算已录制时间
                elapsed_time = time.time() - self.start_time

                # 缓冲满且已达到最少录制时间 → 停止录制
                if self.buffer_count >= self.buffer_frame and elapsed_time >= self.min_recording_time:
                    self._stop_recording(elapsed_time)
                elif self.buffer_count >= self.buffer_frame and elapsed_time < self.min_recording_time:
                    # 缓冲满但未达到最少录制时间，继续录制
                    self.buffer_count = self.buffer_frame - 1  # 重置缓冲计数器，继续录制
                    # print(f"【录制中】缓冲已满，但未达到最少录制时间，继续录制...")

    def _start_new_recording(self):
        """
        开始新的录制：清空帧列表，记录开始时间
        """
        self.recording = True
        self.start_time = time.time()  # 记录开始录制时间
        self.recording_frames = []  # 清空上一轮录制的帧
        self.recording_timestamps = []  # 清空上一轮录制的时间戳
        self.global_cur_timestamp = time.strftime("%Y%m%d_%H%M%S")
        print('=' * 100)
        print(f"【事件触发】开始连续录制，开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def _stop_recording(self, elapsed_time):
        """
        停止录制：直接使用内存中的帧和时间戳进行关键帧提取

        Args:
            elapsed_time: 已录制时间
        """
        self.recording = False
        self.buffer_count = 0
        print(f"【事件结束】停止录制，系统返回休眠检测状态（录制时长：{elapsed_time:.1f}秒，"
              f"帧数：{len(self.recording_frames)}）")


        time1 = time.time()
        # 直接使用内存中的帧和时间戳进行关键帧提取
        try:
            if self.recording_frames:
                keyframe_dir = self.keyframe_dir
                print(f"\n开始处理视频序列: {self.global_cur_timestamp}，帧数: {len(self.recording_frames)}")
                print(f"关键帧输出目录: {keyframe_dir}")

                # 确保关键帧输出目录存在
                os.makedirs(keyframe_dir, exist_ok=True)

                # 调用提取关键帧方法（直接传递帧列表和时间戳）
                if self.use_fine_filter:
                    keyframe_count, selected_frames, selected_timestamps = frameDiff_Flow.extract_keyframes_with_last_keyframe(
                        self.recording_frames, self.recording_timestamps, self.global_cur_timestamp,
                        keyframe_dir, use_fine_filter=True,
                        diff_thresh=self.diff_thresh, flow_thresh=self.flow_thresh, min_interval=self.min_interval
                    )
                    print(f"粗筛完成，共 {keyframe_count} 个候选帧")

                    if keyframe_count > 0:
                        # # 将候选帧保存到磁盘供精细筛选处理
                        # candidate_paths = []
                        # for i, frame in enumerate(selected_frames):
                        #     candidate_path = os.path.join(keyframe_dir, f"candidate_{i:04d}_{selected_timestamps[i]:.2f}.jpg")
                        #     cv2.imwrite(candidate_path, frame)
                        #     candidate_paths.append(os.path.abspath(candidate_path))
                        # # 执行精细筛选（fine_filter 会删除非关键帧，仅保留最终结果）

                        print('*' * 60)
                        selected_frames_fine, selected_timestamps_fine, selected_frames_idx_fine, stats = (
                            self.analyzer.fine_filter_with_framelist(selected_frames, selected_timestamps,
                                                                     min_conf=self.min_conf, weight_ths=self.weight_ths))

                        for frame_idx, curr_frame in enumerate(selected_frames_fine):
                            save_path = f"{keyframe_dir}/{self.global_cur_timestamp}_{frame_idx:04d}.jpg"
                            cv2.imwrite(save_path, curr_frame)

                        print(f"精细筛完成，共 {len(selected_frames_fine)} 个关键帧")

                else:
                    keyframe_count, keyframe_filename_list = frameDiff_Flow.extract_keyframes_with_last_keyframe(
                        self.recording_frames, self.recording_timestamps, self.global_cur_timestamp,
                        keyframe_dir,
                        diff_thresh=self.diff_thresh, flow_thresh=self.flow_thresh, min_interval=self.min_interval
                    )
                    print(f"关键帧提取完成，共提取 {keyframe_count} 个关键帧")

                # 提取完成后清空内存
                self.recording_frames = []
                self.recording_timestamps = []

            else:
                print("recording_frames 为空，跳过处理")
        except Exception as e:
            print(f"处理视频帧时出错: {str(e)}")
        time2 = time.time()
        print(f"帧选择耗时: {time2 - time1:.1f}秒")

    def _cleanup(self):
        """
        清理资源
        """
        self.cap.release()
        cv2.destroyAllWindows()
        print("系统已退出")


# 创建Flask应用
app = Flask(__name__)

# 创建全局录制器实例
recorder = None
recording_thread = None


@app.route('/frames/start', methods=['GET'])
def start_recording_api():
    """
    开始录制API
    """
    global recorder, recording_thread

    if recorder and recorder.is_running:
        return jsonify({"status": "error", "message": "录制已经在运行中"})

    try:
        # 计算关键帧存储路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        keyframe_path = os.path.join(base_dir, "stage1_filter")
        recording_path = os.path.join(base_dir, "recording")
        os.makedirs(keyframe_path, exist_ok=True)
        os.makedirs(recording_path, exist_ok=True)
        # 创建录制器实例（禁用预览以适合服务器环境）
        recorder = EventBasedRecorder(show_preview=False, keyframe_dir=keyframe_path, output_dir=recording_path)
        # 在后台线程中启动录制
        recording_thread = threading.Thread(target=recorder.start_recording, )
        recording_thread.daemon = True
        recording_thread.start()

        return jsonify({"status": "success", "message": "开始录制", "keyframe_path": keyframe_path})
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": f"启动录制失败: {str(e)}"})


@app.route('/frames/stop', methods=['GET'])
def stop_recording_api():
    """
    停止录制API
    """
    global recorder

    if not recorder or not recorder.is_running:
        return jsonify({"status": "error", "message": "录制未在运行"})

    try:
        recorder.stop_recording()
        return jsonify({"status": "success", "message": "停止录制"})
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": f"停止录制失败: {str(e)}"})


@app.route('/frames/stopFine', methods=['GET'])
def stopfine_recording_api():
    """
    停止精细录制API
    """
    global recorder

    if not recorder or not recorder.is_running:
        return jsonify({"status": "error", "message": "录制未在运行"})

    try:
        recorder.stop_fine_filter()
        return jsonify({"status": "success", "message": "停止细筛选"})
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": f"停止录制失败: {str(e)}"})


@app.route('/frames/startFine', methods=['GET'])
def startfine_recording_api():
    """
    开始精细录制API
    """
    global recorder

    if not recorder or not recorder.is_running:
        return jsonify({"status": "error", "message": "录制未在运行"})

    try:
        recorder.start_fine_filter()
        return jsonify({"status": "success", "message": "开始细筛选"})
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": f"停止录制失败: {str(e)}"})


def start_keyframe_server():
    # 启动Flask服务器
    port = 12341
    print("启动服务器，API地址:")
    print(f"  开始录制: http://localhost:{port}/frames/start")
    print(f"  停止录制: http://localhost:{port}/frames/stop")
    print(f"  停止录制: http://localhost:{port}/frames/startFine")
    print(f"  停止录制: http://localhost:{port}/frames/stopFine")
    print("按 Ctrl+C 退出服务器")

    # app.run(host='0.0.0.0', port=port, debug=False)
    serve(app, host='0.0.0.0', port=port)


if __name__ == "__main__":
    start_keyframe_server()