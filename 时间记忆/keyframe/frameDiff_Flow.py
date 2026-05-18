import time

import cv2
import numpy as np
import os


def extract_keyframes(video_path, output_dir="keyframes",
                      diff_thresh=8,  # 帧差阈值: origin: 30
                      flow_thresh=1.5,  # 光流平均运动阈值
                      min_interval=5,
                      return_keyframe_idx=False, verbose=True,
                      save_keyframes=True):  # 最小关键帧间隔（去重）
    """
    工业级：帧差 + LK光流 融合关键帧提取
    """

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数：{total_frames}，FPS：{fps}")

    # 用于光流 LK 参数
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 第一帧初始化
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # 检测角点（光流用）
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    # import matplotlib.pyplot as plt
    # plt.imshow(prev_frame)
    # plt.scatter([x[0, 0] for x in prev_pts], [x[0, 1] for x in prev_pts], s=50, color='green', marker='+')
    # plt.show()
    keyframe_count = 0
    frame_idx = 0
    last_keyframe = -min_interval  # 上一个关键帧位置
    frame_idx_list = []
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # ======================
        # 步骤1：帧差法粗筛
        # ======================
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        diff_mean = frame_diff.mean()  # 整帧平均差值

        # 差值太小 → 静止 → 跳过
        if diff_mean < diff_thresh:
            prev_gray = curr_gray.copy()
            continue

        # ======================
        # 步骤2：光流法精筛（判断真实运动）
        # ======================
        if prev_pts is None:
            prev_gray = curr_gray.copy()
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            continue

        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

        # 筛选有效点
        if curr_pts is not None:
            good_new = curr_pts[status == 1]
            good_old = prev_pts[status == 1]

            if len(good_new) > 0:
                # 计算平均运动幅度
                motion_magnitude = np.mean(np.linalg.norm(good_new - good_old, axis=1))
            else:
                motion_magnitude = 0
        else:
            motion_magnitude = 0

        # ======================
        # 步骤3：双条件判定关键帧
        # ======================
        if (motion_magnitude > flow_thresh and
                frame_idx - last_keyframe >= min_interval):

            keyframe_count += 1
            last_keyframe = frame_idx
            if save_keyframes:
                save_path = f"{output_dir}/keyframe_{frame_idx:04d}.jpg"
                cv2.imwrite(save_path, curr_frame)
            if verbose:
                print(
                    f"关键帧 {keyframe_count} → 帧号 {frame_idx} | 差值:{diff_mean:.1f} | 运动:{motion_magnitude:.2f}")
            frame_idx_list.append(frame_idx)

        # 更新上一帧状态
        prev_gray = curr_gray.copy()
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    cap.release()

    print(
        f"\n提取完成！共关键帧 {keyframe_count} 张, 共 {frame_idx + 1} 帧，压缩比 {keyframe_count / (frame_idx + 1):.6f},  diff_thresh={diff_thresh}, flow_thresh={flow_thresh}, min_interval={min_interval}")
    if return_keyframe_idx:
        return frame_idx_list
    else:
        return keyframe_count


def extract_keyframes_with_last_keyframe(recording_frames, timestamps, cur_timestamp, output_dir="keyframes",
                                         diff_thresh=8, flow_thresh=1.5, min_interval=5, only_coarse_filter=True,
                                         use_fine_filter=False):
    """
    工业级：帧差 + LK光流 融合关键帧提取
    如果关键帧文件夹已存在，则使用最后一个关键帧作为初始帧

    Args:
        recording_frames: 视频帧列表，每个元素为BGR帧 (numpy array)
        timestamps: 录制时间戳列表，与recording_frames一一对应
        output_dir: 输出关键帧文件夹
        diff_thresh: 帧差阈值
        flow_thresh: 光流平均运动阈值
        min_interval: 最小关键帧间隔（帧数）

    Returns:
        tuple:
            - use_fine_filter=False: (关键帧数量, 关键帧文件路径列表)
            - use_fine_filter=True: (关键帧数量, 关键帧帧列表, 关键帧时间戳列表)
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 检查是否存在已有的关键帧
    existing_keyframes = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
    initial_keyframe = None

    if existing_keyframes:
        # 找到最后一个关键帧
        last_keyframe_file = existing_keyframes[-1]
        last_keyframe_path = os.path.join(output_dir, last_keyframe_file)
        initial_keyframe = cv2.imread(last_keyframe_path)
        if initial_keyframe is not None:
            print(f"使用已有的最后关键帧作为初始帧: {last_keyframe_file}")
        else:
            print(f"无法读取最后关键帧: {last_keyframe_file}，使用视频第一帧")
            initial_keyframe = None
    else:
        print("没有找到已有的关键帧，使用视频第一帧")

    if not recording_frames:
        print("输入帧列表为空")
        return 0, []

    total_frames = len(recording_frames)
    base_time = timestamps[0] if timestamps else 0
    print(f"输入帧总数：{total_frames}")

    # 用于光流 LK 参数
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 第一帧初始化
    if initial_keyframe is not None:
        prev_frame = initial_keyframe
    else:
        prev_frame = recording_frames[0]

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # 检测角点（光流用）
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    keyframe_count = 0
    last_keyframe = -min_interval  # 上一个关键帧位置

    keyframe_filename_list = []
    selected_frames = []  # use_fine_filter=True 时使用
    selected_timestamps = []  # use_fine_filter=True 时使用

    for frame_idx, curr_frame in enumerate(recording_frames):
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # ======================
        # 步骤1：帧差法粗筛
        # ======================
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        diff_mean = frame_diff.mean()  # 整帧平均差值

        # 差值太小 → 静止 → 跳过
        if diff_mean < diff_thresh:
            prev_gray = curr_gray.copy()
            continue

        # ======================
        # 步骤2：光流法精筛（判断真实运动）
        # ======================
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

        # 筛选有效点
        if curr_pts is not None:
            good_new = curr_pts[status == 1]
            good_old = prev_pts[status == 1]

            if len(good_new) > 0:
                # 计算平均运动幅度
                motion_magnitude = np.mean(np.linalg.norm(good_new - good_old, axis=1))
            else:
                motion_magnitude = 0
        else:
            motion_magnitude = 0

        # ======================
        # 步骤3：双条件判定关键帧
        # ======================
        if (motion_magnitude > flow_thresh and
                frame_idx - last_keyframe >= min_interval):
            keyframe_count += 1
            last_keyframe = frame_idx
            rel_time = timestamps[frame_idx] - base_time

            if use_fine_filter:
                # 不保存文件，收集到内存列表
                selected_frames.append(curr_frame)
                selected_timestamps.append(timestamps[frame_idx])
            else:
                # 保存文件到磁盘
                save_path = f"{output_dir}/{cur_timestamp}_{frame_idx:04d}.jpg"
                cv2.imwrite(save_path, curr_frame)
                keyframe_filename_list.append(os.path.abspath(save_path))

            print(f"关键帧 {keyframe_count} → 帧号 {frame_idx} | 时间:{rel_time:.2f}s | 差值:{diff_mean:.1f} | 运动:{motion_magnitude:.2f}")
        # 更新上一帧状态
        prev_gray = curr_gray.copy()
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    print(
        f"\n提取完成！共关键帧 {keyframe_count} 张, 共 {total_frames} 帧，压缩比 {keyframe_count / total_frames:.6f},  diff_thresh={diff_thresh}, flow_thresh={flow_thresh}, min_interval={min_interval}")
    if use_fine_filter:
        return keyframe_count, selected_frames, selected_timestamps
    else:
        return keyframe_count, keyframe_filename_list


if __name__ == "__main__":
    video_path = '../video1.mp4'

    t0 = time.time()
    # extract_keyframes(video_path)
    extract_keyframes(video_path, diff_thresh=5, flow_thresh=1.5, min_interval=5)
    t1 = time.time()

    print(f"提取关键帧耗时：{t1 - t0:.2f} 秒")


def refine_keyframes_from_folder(existing_keyframes_dir, raw_frames_dir,
                                 diff_thresh=8, flow_thresh=1.5, min_interval=5):
    """
    基于已有的关键帧，对原始帧文件夹进行进一步筛选

    Args:
        existing_keyframes_dir: 已有的关键帧文件夹路径
        raw_frames_dir: 原始帧文件夹路径
        diff_thresh: 帧差阈值
        flow_thresh: 光流平均运动阈值
        min_interval: 最小关键帧间隔

    Returns:
        int: 新添加的关键帧数量
    """
    # 确保输出目录存在
    os.makedirs(existing_keyframes_dir, exist_ok=True)

    # 获取现有关键帧文件
    existing_keyframes = sorted([f for f in os.listdir(existing_keyframes_dir) if f.endswith('.jpg')])
    print(f"现有关键帧数量：{len(existing_keyframes)}")

    # 获取原始帧文件并排序
    raw_frames = sorted([f for f in os.listdir(raw_frames_dir) if f.endswith(('.jpg', '.png'))])
    print(f"原始帧数量：{len(raw_frames)}")

    if not raw_frames:
        print("原始帧文件夹为空")
        return 0

    # 用于光流 LK 参数
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # 读取第一个原始帧作为初始帧
    first_frame_path = os.path.join(raw_frames_dir, raw_frames[0])
    prev_frame = cv2.imread(first_frame_path)
    if prev_frame is None:
        print(f"无法读取第一帧：{first_frame_path}")
        return 0

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # 跟踪变量
    new_keyframe_count = 0
    frame_idx = 0
    last_keyframe = -min_interval  # 上一个关键帧位置

    # 处理所有原始帧
    for i, frame_file in enumerate(raw_frames):
        frame_path = os.path.join(raw_frames_dir, frame_file)
        curr_frame = cv2.imread(frame_path)

        if curr_frame is None:
            print(f"无法读取帧：{frame_path}")
            continue

        frame_idx = i
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # ======================
        # 步骤1：帧差法粗筛
        # ======================
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        diff_mean = frame_diff.mean()  # 整帧平均差值

        # 差值太小 → 静止 → 跳过
        if diff_mean < diff_thresh:
            prev_gray = curr_gray.copy()
            continue

        # ======================
        # 步骤2：光流法精筛（判断真实运动）
        # ======================
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

        # 筛选有效点
        if curr_pts is not None:
            good_new = curr_pts[status == 1]
            good_old = prev_pts[status == 1]

            if len(good_new) > 0:
                # 计算平均运动幅度
                motion_magnitude = np.mean(np.linalg.norm(good_new - good_old, axis=1))
            else:
                motion_magnitude = 0
        else:
            motion_magnitude = 0

        # ======================
        # 步骤3：双条件判定关键帧
        # ======================
        if (motion_magnitude > flow_thresh and
                frame_idx - last_keyframe >= min_interval):

            # 检查是否已存在同名关键帧
            save_filename = f"keyframe_{frame_idx:04d}.jpg"
            save_path = os.path.join(existing_keyframes_dir, save_filename)

            if not os.path.exists(save_path):
                new_keyframe_count += 1
                last_keyframe = frame_idx
                cv2.imwrite(save_path, curr_frame)
                print(
                    f"新关键帧 {new_keyframe_count} → 帧号 {frame_idx} | 差值:{diff_mean:.1f} | 运动:{motion_magnitude:.2f}")
            else:
                print(f"关键帧已存在：{save_filename}")

        # 更新上一帧状态
        prev_gray = curr_gray.copy()
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    print(f"\n筛选完成！新添加关键帧 {new_keyframe_count} 张, 共处理 {frame_idx + 1} 帧")
    print(f"现有关键帧总数：{len(existing_keyframes) + new_keyframe_count}")
    return new_keyframe_count


# ======================
# 运行示例
# ======================
if __name__ == "__main__":
    # 示例1：从视频提取关键帧
    video_path = '../video1.mp4'

    t0 = time.time()
    extract_keyframes(video_path, diff_thresh=5, flow_thresh=1.5, min_interval=5)
    t1 = time.time()

    print(f"提取关键帧耗时：{t1 - t0:.2f} 秒")

    # 示例2：基于已有关键帧对原始帧进行筛选
    # existing_keyframes_dir = 'keyframes'
    # raw_frames_dir = '../frames'
    # refine_keyframes_from_folder(existing_keyframes_dir, raw_frames_dir)
    # t2 = time.time()
    # print(f"筛选关键帧耗时：{t2 - t1:.2f} 秒")


def extract_keyframes_from_multiple_videos(video_files, output_dir="keyframes",
                                           diff_thresh=8, flow_thresh=1.5, min_interval=5):
    """
    从多个视频文件中提取关键帧，按时间顺序处理，并使用原视频名称_帧序号命名

    Args:
        video_files: 视频文件路径列表
        output_dir: 输出关键帧文件夹
        diff_thresh: 帧差阈值
        flow_thresh: 光流平均运动阈值
        min_interval: 最小关键帧间隔

    Returns:
        int: 提取的关键帧总数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 按文件名排序（假设文件名包含时间戳）
    sorted_videos = sorted(video_files)
    print(f"处理视频列表（按时间顺序）：")
    for i, video_path in enumerate(sorted_videos, 1):
        print(f"  {i}. {os.path.basename(video_path)}")

    total_keyframes = 0
    initial_keyframe = None  # 上一个视频的最后一个关键帧
    total_frames = 0

    # 更新初始关键帧为当前视频的最后一个关键帧
    # 查找当前视频的最后一个关键帧
    # video_keyframe_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
    # if video_keyframe_files:
    #     # 按帧序号排序
    #     video_keyframe_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    #     last_keyframe = video_keyframe_files[-1]
    #     initial_keyframe = os.path.join(output_dir, last_keyframe)
    #     print(f"设置下一个视频的初始关键帧: {last_keyframe}")

    for video_idx, video_path in enumerate(sorted_videos, 1):
        print(f"\n处理第 {video_idx}/{len(sorted_videos)} 个视频: {os.path.basename(video_path)}")
        t0 = time.time()
        # 提取视频名称（不含扩展名）
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # 处理单个视频
        video_keyframes, video_i_total_frames, initial_keyframe = _extract_keyframes_with_naming(
            video_path,
            output_dir=output_dir,
            video_name=video_name,
            diff_thresh=diff_thresh,
            flow_thresh=flow_thresh,
            min_interval=min_interval,
            initial_keyframe=initial_keyframe
        )
        t1 = time.time()
        print(f"提取关键帧耗时：{t1 - t0:.2f} 秒")

        total_frames += video_i_total_frames

        total_keyframes += video_keyframes

    print(f"\n所有视频处理完成！")
    print(f"总计提取 {total_keyframes} 个关键帧, 共处理 {total_frames} 帧，提取比：{total_keyframes / total_frames:.6f} ")
    return total_keyframes


def _extract_keyframes_with_naming(video_path, output_dir="keyframes", video_name="video",
                                   diff_thresh=8, flow_thresh=1.5, min_interval=5,
                                   initial_keyframe=None):
    """
    从单个视频中提取关键帧，使用视频名称_帧序号命名

    Args:
        video_path: 视频文件路径
        output_dir: 输出关键帧文件夹
        video_name: 视频名称（用于命名）
        diff_thresh: 帧差阈值
        flow_thresh: 光流平均运动阈值
        min_interval: 最小关键帧间隔
        initial_keyframe: 初始关键帧（可以是路径字符串或帧数据）

    Returns:
        tuple: (关键帧数量, 处理的总帧数, 最后一个关键帧)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return 0, 0, None

    # 第一帧初始化
    if initial_keyframe is not None:
        if isinstance(initial_keyframe, str):
            # 是路径字符串
            if os.path.exists(initial_keyframe):
                prev_frame = cv2.imread(initial_keyframe)
                if prev_frame is not None:
                    print(f"使用指定的初始关键帧: {initial_keyframe}")
                else:
                    print(f"无法读取初始关键帧: {initial_keyframe}，使用视频第一帧")
                    ret, prev_frame = cap.read()
            else:
                print(f"初始关键帧路径不存在: {initial_keyframe}，使用视频第一帧")
                ret, prev_frame = cap.read()
        else:
            # 是帧数据
            prev_frame = initial_keyframe
            print("使用前一个视频的最后关键帧作为初始帧")
    else:
        # 使用视频的第一帧
        ret, prev_frame = cap.read()

    if prev_frame is None:
        print(f"无法读取视频的第一帧: {video_path}")
        cap.release()
        return 0, 0, None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # 用于光流 LK 参数
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    keyframe_count = 0
    frame_idx = 0
    last_keyframe = -min_interval  # 上一个关键帧位置

    last_keyframe_frame = None

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # ======================
        # 步骤1：帧差法粗筛
        # ======================
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        diff_mean = frame_diff.mean()  # 整帧平均差值

        # 差值太小 → 静止 → 跳过
        if diff_mean < diff_thresh:
            prev_gray = curr_gray.copy()
            continue

        # ======================
        # 步骤2：光流法精筛（判断真实运动）
        # ======================
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

        # 筛选有效点
        if curr_pts is not None:
            good_new = curr_pts[status == 1]
            good_old = prev_pts[status == 1]

            if len(good_new) > 0:
                # 计算平均运动幅度
                motion_magnitude = np.mean(np.linalg.norm(good_new - good_old, axis=1))
            else:
                motion_magnitude = 0
        else:
            motion_magnitude = 0

        # ======================
        # 步骤3：双条件判定关键帧
        # ======================
        if (motion_magnitude > flow_thresh and
                frame_idx - last_keyframe >= min_interval):
            # 使用视频名称_帧序号命名
            save_filename = f"{video_name}_{frame_idx:06d}.jpg"
            save_path = os.path.join(output_dir, save_filename)

            # 保存关键帧
            cv2.imwrite(save_path, curr_frame)
            keyframe_count += 1
            last_keyframe = frame_idx

            last_keyframe_frame = curr_frame
            print(f"关键帧 {keyframe_count} → {save_filename} | 差值:{diff_mean:.1f} | 运动:{motion_magnitude:.2f}")

        # 更新上一帧状态
        prev_gray = curr_gray.copy()
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    cap.release()
    print(f"视频 {video_name} 处理完成，提取 {keyframe_count} 个关键帧")
    return keyframe_count, frame_idx, last_keyframe_frame

# 示例3：处理多个视频
# if __name__ == "__main__":
#     video_files = ['video1.mp4', 'video2.mp4', 'video3.mp4']
#     extract_keyframes_from_multiple_videos(video_files, output_dir='keyframes')