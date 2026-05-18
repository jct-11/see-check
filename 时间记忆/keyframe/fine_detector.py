import json
import os
import cv2
import numpy as np

from keyframe.detector import YOLODetector


class KeyFrameAnalyzer:

    def __init__(self, keyword_json_path=r'keyword.json', detector=None):
        """
        初始化分析器
        :param keyword_json_path: 关键字库JSON文件路径
        """
        # 1. 加载关键字库
        base_dir = os.path.dirname(os.path.abspath(__file__))
        keyword_json_path = os.path.join(base_dir, keyword_json_path)
        with open(keyword_json_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 2. 构建快速查找字典 {category_name: weight}
        # 我们将所有子类别的关键词合并到一个大字典中方便查询
        self.weights_map = {}

        # 提取 core_subjects
        for item in self.config['categories']['core_subjects']['keywords']:
            self.weights_map[item['name']] = item['weight']

        # 提取 interaction_objects (遍历所有子分类)
        interactions = self.config['categories']['interaction_objects']['sub_categories']
        for sub_cat in interactions.values():
            for item in sub_cat:
                self.weights_map[item['name']] = item['weight']

        # 提取 environment_context
        for item in self.config['categories']['environment_context']['keywords']:
            self.weights_map[item['name']] = item['weight']

        print(f"✅ 语义库加载完成，共加载 {len(self.weights_map)} 个有效关键词。")

        '''
            load detection model
            current: yolo26n
        '''
        if detector:
            self.detector = detector
        else:
            os.makedirs('weights', exist_ok=True)
            model_path = os.path.join(base_dir, 'yolo26n.pt')
            self.detector = YOLODetector(model_path)


    def fine_filter(self, keyframe_file_list):
        """
        分析关键帧文件列表，删除非关键帧
        
        Args:
            keyframe_file_list: 关键帧文件路径列表
        
        Returns:
            dict: 包含统计信息的字典
        """
        if not keyframe_file_list:
            return {"total": 0, "deleted": 0, "ratio": 0.0}

        total_count = len(keyframe_file_list)
        deleted_count = 0
        deleted_files = []

        print(f"开始分析 {total_count} 个关键帧文件...")

        for i, keyframe_file in enumerate(keyframe_file_list, 1):
            print(f"分析第 {i}/{total_count} 个文件: {keyframe_file}")

            try:
                # 执行检测
                detection_result = self.detector.detect(keyframe_file, show=False, save=False)

                # 分析是否为关键帧
                is_keyframe, score, details = self.analyze(detection_result)

                if not is_keyframe:
                    # 删除非关键帧
                    os.remove(keyframe_file)
                    deleted_count += 1
                    deleted_files.append(keyframe_file)
                    print(f"❌ 删除非关键帧: {keyframe_file} (得分: {score})")
                else:
                    print(f"✅ 保留关键帧: {keyframe_file} (得分: {score})")

            except Exception as e:
                print(f"处理文件 {keyframe_file} 时出错: {str(e)}")
                continue

        # 计算删除比例
        delete_ratio = deleted_count / total_count if total_count > 0 else 0.0

        # 输出统计信息
        print(f"\n分析完成！")
        print(f"总文件数: {total_count}")
        print(f"删除文件数: {deleted_count}")
        print(f"删除比例: {delete_ratio:.2f}")

        # 返回统计结果
        return {
            "total": total_count,
            "deleted": deleted_count,
            "ratio": delete_ratio,
            "deleted_files": deleted_files
        }

    def fine_filter_with_framelist(self, frame_list, timestamp_list,
                                   min_conf=0.5, weight_ths=15.0):
        """
        直接对帧列表进行语义筛选，返回通过筛选的帧和时间戳

        Args:
            frame_list: 帧列表（numpy array, BGR格式）
            timestamp_list: 与帧列表对应的时间戳列表
            min_conf: 检测置信度阈值（默认0.5）
            weight_ths: 关键帧判定阈值（默认15.0）

        Returns:
            tuple: (selected_frames, selected_timestamps, stats_dict)
                - selected_frames: 通过筛选的帧列表
                - selected_timestamps: 对应的时间戳列表
                - stats_dict: 统计信息
        """
        if not frame_list or not timestamp_list:
            return [], [], {"total": 0, "selected": 0, "deleted": 0, "ratio": 0.0}

        total_count = len(frame_list)
        selected_frames = []
        selected_timestamps = []
        selected_frames_idx = []
        deleted_count = 0

        print(f"开始语义筛选 {total_count} 帧...")

        for i, (frame, timestamp) in enumerate(zip(frame_list, timestamp_list), 1):
            print(f"分析第 {i}/{total_count} 帧")

            try:
                # 执行检测（YOLO 可直接接受 numpy array）
                detection_result = self.detector.detect(frame, show=False, save=False)

                # 分析是否为关键帧（传入实际分辨率）
                h, w = frame.shape[:2]
                is_keyframe, score, matched_objects = self.analyze(
                    detection_result, img_h=h, img_w=w,
                    min_conf=min_conf, weight_ths=weight_ths
                )

                if is_keyframe:
                    selected_frames.append(frame)
                    selected_timestamps.append(timestamp)
                    selected_frames_idx.append(i)
                    print(f"✅ 保留关键帧 (得分: {score})")
                else:
                    deleted_count += 1
                    print(f"❌ 删除非关键帧 (得分: {score})")

            except Exception as e:
                print(f"处理第 {i} 帧时出错: {str(e)}")
                continue

        # 计算筛选比例
        delete_ratio = deleted_count / total_count if total_count > 0 else 0.0

        # 输出统计信息
        print(f"\n语义筛选完成！")
        print(f"总帧数: {total_count}")
        print(f"保留帧数: {len(selected_frames)}")
        print(f"删除帧数: {deleted_count}")
        print(f"删除比例: {delete_ratio:.2f}")

        stats = {
            "total": total_count,
            "selected": len(selected_frames),
            "deleted": deleted_count,
            "ratio": delete_ratio
        }

        return selected_frames, selected_timestamps, selected_frames_idx, stats

    def fine_filter_by_frame_idx(self, video_path, frame_indices, verbose=False, min_conf=0.5, weight_ths=15.0):
        """
        基于视频路径和第一阶段选择的帧序号进行语义筛选
        
        Args:
            video_path: 原始视频文件路径
            frame_indices: 第一阶段选择的帧序号列表（0-based）
        
        Returns:
            dict: 包含筛选结果的字典
                - selected_indices: 第二阶段筛选后保留的帧序号列表
                - deleted_indices: 被删除的帧序号列表
                - total: 输入帧总数
                - selected: 保留帧数量
                - deleted: 删除帧数量
                - ratio: 删除比例
                - details: 每个帧的分析详情 {frame_idx: {is_keyframe, score, matched_objects}}
        """
        if not frame_indices:
            return {
                "selected_indices": [],
                "deleted_indices": [],
                "total": 0,
                "selected": 0,
                "deleted": 0,
                "ratio": 0.0,
                "details": {}
            }

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_path}")
            return {
                "selected_indices": [],
                "deleted_indices": [],
                "total": 0,
                "selected": 0,
                "deleted": 0,
                "ratio": 0.0,
                "details": {}
            }

        # 获取视频属性
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        total_count = len(frame_indices)
        selected_count = 0
        deleted_count = 0
        selected_indices = []
        deleted_indices = []
        details = {}

        print(f"\n开始对视频 {os.path.basename(video_path)} 进行语义筛选...")
        print(f"视频总帧数: {total_frames}, 输入帧数量: {total_count}")

        for i, frame_idx in enumerate(frame_indices, 1):
            # 检查帧序号是否有效
            if frame_idx < 0 or frame_idx >= total_frames:
                print(f"⚠️ 跳过无效帧序号: {frame_idx}")
                deleted_indices.append(frame_idx)
                deleted_count += 1
                details[frame_idx] = {
                    "is_keyframe": False,
                    "score": 0.0,
                    "matched_objects": [],
                    "reason": "无效帧序号"
                }
                continue

            if verbose:
                print(f"\n分析第 {i}/{total_count} 帧: 帧序号={frame_idx}")

            try:
                # 设置帧位置并读取帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    print(f"❌ 无法读取帧 {frame_idx}")
                    deleted_indices.append(frame_idx)
                    deleted_count += 1
                    details[frame_idx] = {
                        "is_keyframe": False,
                        "score": 0.0,
                        "matched_objects": [],
                        "reason": "无法读取"
                    }
                    continue

                # 执行检测
                detection_result = self.detector.detect(frame, show=False, save=False)

                # 分析是否为关键帧
                is_keyframe, score, matched_objects = self.analyze(detection_result, img_h=height, img_w=width,
                                                                   min_conf=min_conf, weight_ths=weight_ths)

                if is_keyframe:
                    selected_indices.append(frame_idx)
                    selected_count += 1
                    if verbose:
                        print(f"✅ 保留关键帧: 帧序号={frame_idx} (得分: {score})")
                else:
                    deleted_indices.append(frame_idx)
                    deleted_count += 1

                    if verbose:
                        print(f"❌ 删除非关键帧: 帧序号={frame_idx} (得分: {score})")

                # 记录详情
                details[frame_idx] = {
                    "is_keyframe": is_keyframe,
                    "score": score,
                    "matched_objects": matched_objects
                }

            except Exception as e:
                print(f"处理帧 {frame_idx} 时出错: {str(e)}")
                deleted_indices.append(frame_idx)
                deleted_count += 1
                details[frame_idx] = {
                    "is_keyframe": False,
                    "score": 0.0,
                    "matched_objects": [],
                    "reason": str(e)
                }
                continue

        # 释放资源
        cap.release()

        # 计算删除比例
        delete_ratio = deleted_count / total_count if total_count > 0 else 0.0

        # 输出统计信息
        print(f"\n语义筛选完成！")
        print(f"总帧数量: {total_count}")
        print(f"保留帧数量: {selected_count}")
        print(f"删除帧数量: {deleted_count}")
        print(f"删除比例: {delete_ratio:.2f}")

        # 返回结果
        return {
            "selected_indices": selected_indices,
            "deleted_indices": deleted_indices,
            "total": total_count,
            "selected": selected_count,
            "deleted": deleted_count,
            "ratio": delete_ratio,
            "details": details
        }

    def analyze(self, detection_data, img_h=1080, img_w=1920, min_conf=0.5, weight_ths=15.0):
        """
        根据模型输出的字典数据进行关键帧判断
        :param detection_data: 符合指定JSON结构的字典
        :return: (bool: 是否关键帧, float: 得分, list: 命中的物体详情)
        """
        total_score = 0.0
        matched_objects = []

        # 获取图像尺寸（用于计算面积占比，假设原图是1920x1080，实际可根据需要调整）
        # 注意：你的JSON结构中未直接提供原图宽高，这里假设一个标准值或需从外部传入
        frame_area = img_h * img_w

        # 阈值设定 (可配置)
        min_size_ratio = self.config['global_settings']['min_object_size_ratio']

        # 遍历检测结果
        # 结构定位: results[0]['boxes']
        if not detection_data.get('results'):
            return False, 0.0, []

        boxes = detection_data['results'][0].get('boxes', [])

        for box in boxes:
            class_name = box.get('class_name')
            conf = box.get('conf')
            xyxy = box.get('xyxy')

            # 1. 检查是否在关键字库中
            if class_name not in self.weights_map:
                continue

            weight = self.weights_map[class_name]

            # 2. 基础置信度过滤 (可选，例如只考虑 conf > 0.5 的检测)
            if conf < min_conf:
                continue

            # 3. 计算尺寸权重 (防止远处微小物体干扰)
            x1, y1, x2, y2 = xyxy
            obj_area = (x2 - x1) * (y2 - y1)
            area_ratio = obj_area / frame_area

            if area_ratio < min_size_ratio:
                continue  # 太小了，忽略

            # 4. 计算最终得分
            # 公式：权重 * 置信度 * (1 + 面积系数)
            # 面积系数让占据画面大的物体得分更高
            size_factor = 1 + area_ratio
            score = weight * conf * size_factor

            total_score += score
            matched_objects.append({
                "name": class_name,
                "score": round(score, 2),
                "conf": conf
            })

        # 5. 判定逻辑
        # 这里简单设定一个全局阈值，比如 15.0
        # 实际应用中可以根据不同场景模式读取JSON中的 threshold 字段
        is_keyframe = total_score >= weight_ths

        return is_keyframe, round(total_score, 2), matched_objects

    def select_event_segments(self, video_path, event_segments, sample_frames=5, 
                              min_conf=0.5, weight_ths=15.0, verbose=False):
        """
        事件流选择方法：根据语义权重选择重要的事件片段

        Args:
            video_path: 原始视频文件路径
            event_segments: 事件片段列表，每个元素为元组 (start_frame, end_frame)
            sample_frames: 每个片段采样的帧数（默认5）
            min_conf: 检测置信度阈值（默认0.5）
            weight_ths: 关键帧判定阈值（默认15.0）
            verbose: 是否打印详细信息（默认False）

        Returns:
            dict: 包含选择结果的字典
                - selected_segments: 选中的重要事件片段列表
                - rejected_segments: 被拒绝的事件片段列表
                - segment_weights: 每个片段的权重字典 {segment_idx: {score, details}}
                - total_segments: 输入片段总数
                - selected_count: 选中片段数量
                - rejected_count: 拒绝片段数量
        """
        if not event_segments:
            return {
                "selected_segments": [],
                "rejected_segments": [],
                "segment_weights": {},
                "total_segments": 0,
                "selected_count": 0,
                "rejected_count": 0
            }

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_path}")
            return {
                "selected_segments": [],
                "rejected_segments": [],
                "segment_weights": {},
                "total_segments": 0,
                "selected_count": 0,
                "rejected_count": 0
            }

        # 获取视频属性
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        print(f"\n开始事件流选择分析...")
        print(f"视频信息: {width}x{height} @ {fps}fps, 共 {total_frames} 帧")
        print(f"输入事件片段数量: {len(event_segments)}")

        selected_segments = []
        rejected_segments = []
        segment_weights = {}

        for seg_idx, (start_frame, end_frame) in enumerate(event_segments):
            # 检查片段边界有效性
            if start_frame < 0:
                start_frame = 0
            if end_frame >= total_frames:
                end_frame = total_frames - 1
            if start_frame >= end_frame:
                print(f"⚠️ 跳过无效片段 {seg_idx}: [{start_frame}, {end_frame}]")
                rejected_segments.append((start_frame, end_frame))
                segment_weights[seg_idx] = {
                    "score": 0.0,
                    "is_selected": False,
                    "reason": "无效片段边界"
                }
                continue

            segment_duration = (end_frame - start_frame) / fps
            if verbose:
                print(f"\n分析片段 {seg_idx}: [{start_frame}, {end_frame}], 时长: {segment_duration:.2f}s")

            # 在片段中均匀采样帧
            segment_length = end_frame - start_frame + 1
            if segment_length <= sample_frames:
                # 如果片段帧数少于采样数，取所有帧
                sample_indices = list(range(start_frame, end_frame + 1))
            else:
                # 均匀采样
                step = segment_length // sample_frames
                sample_indices = [start_frame + i * step for i in range(sample_frames)]

            # 分析每个采样帧
            frame_scores = []
            all_matched_objects = []

            for frame_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                # 执行检测
                detection_result = self.detector.detect(frame, show=False, save=False)
                is_keyframe, score, matched_objects = self.analyze(
                    detection_result, img_h=height, img_w=width,
                    min_conf=min_conf, weight_ths=weight_ths
                )
                frame_scores.append(score)
                all_matched_objects.extend(matched_objects)

            # 计算片段的综合得分（取所有采样帧的平均得分）
            if frame_scores:
                segment_score = np.mean(frame_scores)
            else:
                segment_score = 0.0

            # 根据得分判断是否选中
            is_selected = segment_score >= weight_ths

            # 记录结果
            if is_selected:
                selected_segments.append((start_frame, end_frame))
                if verbose:
                    print(f"✅ 选中事件片段 {seg_idx}: 得分={segment_score:.2f}")
            else:
                rejected_segments.append((start_frame, end_frame))
                if verbose:
                    print(f"❌ 拒绝事件片段 {seg_idx}: 得分={segment_score:.2f}")

            # 保存详细信息
            segment_weights[seg_idx] = {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "score": segment_score,
                "duration_s": segment_duration,
                "is_selected": is_selected,
                "matched_objects": all_matched_objects
            }

        # 释放资源
        cap.release()

        # 计算选中片段的总帧数
        selected_frames = sum(end - start + 1 for start, end in selected_segments)
        
        # 输出统计信息
        print(f"\n事件流选择完成！")
        print(f"总片段数: {len(event_segments)}")
        print(f"选中片段数: {len(selected_segments)}")
        print(f"拒绝片段数: {len(rejected_segments)}")
        print(f"选择比例: {len(selected_segments) / len(event_segments):.2f}")
        print(f"选择帧数/原视频帧数: {selected_frames}/{total_frames} ({selected_frames / total_frames:.2%})")
        compression_ratio = total_frames / selected_frames if selected_frames > 0 else float('inf')
        print(f"压缩比: {compression_ratio:.2f}x")
        

        # 返回结果
        return {
            "selected_segments": selected_segments,
            "rejected_segments": rejected_segments,
            "segment_weights": segment_weights,
            "total_segments": len(event_segments),
            "selected_count": len(selected_segments),
            "rejected_count": len(rejected_segments)
        }


# ==================== 模拟测试 ====================

if __name__ == "__main__":
    # 测试 fine_filter 方法
    # 1. 初始化分析器
    # analyzer = KeyFrameAnalyzer()
    #
    # # 2. 准备测试文件列表
    # # 这里假设 stage1_filter 文件夹中有关键帧文件
    # stage1_filter_dir = r'/Users/libin/LVNet/stage1_filter'
    # keyframe_files = []
    #
    # if os.path.exists(stage1_filter_dir):
    #     for file in os.listdir(stage1_filter_dir):
    #         if file.endswith('.jpg'):
    #             keyframe_files.append(os.path.join(stage1_filter_dir, file))
    #
    # if keyframe_files:
    #     # 3. 执行精细筛选
    #     result = analyzer.fine_filter(keyframe_files)
    #
    #     # 4. 输出结果
    #     print(f"\n精细筛选结果:")
    #     print(f"总文件数: {result['total']}")
    #     print(f"删除文件数: {result['deleted']}")
    #     print(f"删除比例: {result['ratio']:.2f}")
    # else:
    #     print("没有找到关键帧文件")

    # 初始化检测器
    model_path = r'/Users/libin/LVNet/week3/yolo26n.pt'
    detector = YOLODetector(model_path, )

    # 测试图像路径
    img_path = r'/Users/libin/LVNet/key_frames_20260417_152102/I_frame_000000.jpg'

    # 执行检测
    mock_detection_result = detector.detect(img_path, show=True, save=True)

    # 1. 初始化分析器
    analyzer = KeyFrameAnalyzer(detector=detector)

    # 2. 构造一个符合你描述的 JSON 数据结构 (模拟 YOLO 的输出)
    # mock_detection_result = {
    #     "image_path": "test_vlog_001.jpg",
    #     "results": [
    #         {
    #             "boxes": [
    #                 # 场景描述：一个人在吃饭 (Person + Cup + Fork) -> 应该是高分关键帧
    #                 {
    #                     "xyxy": [100, 100, 800, 1000],  # 一个人 (很大)
    #                     "conf": 0.95,
    #                     "cls": 0,
    #                     "class_name": "person"
    #                 },
    #                 {
    #                     "xyxy": [400, 600, 450, 700],  # 一个杯子
    #                     "conf": 0.88,
    #                     "cls": 39,
    #                     "class_name": "cup"
    #                 },
    #                 {
    #                     "xyxy": [420, 620, 460, 650],  # 一把叉子
    #                     "conf": 0.75,
    #                     "cls": 40,
    #                     "class_name": "fork"
    #                 },
    #                 # 背景杂物 (低权重)
    #                 {
    #                     "xyxy": [10, 10, 50, 50],  # 远处的盆栽 (很小)
    #                     "conf": 0.60,
    #                     "cls": 59,
    #                     "class_name": "potted plant"
    #                 }
    #             ],
    #             "masks": {"shape": (0, 0, 0), "count": 0},
    #             "keypoints": {"shape": (0, 0, 0), "count": 0},
    #             "probs": {"top1": 0, "top1_conf": 0},
    #             "obb": {"shape": (0, 0), "count": 0}
    #         }
    #     ],
    #     "total_detections": 4
    # }

    # 3. 执行判断
    is_key, score, details = analyzer.analyze(mock_detection_result)

    # 4. 输出结果
    print("-" * 30)
    print(f"📷 图片: {mock_detection_result['image_path']}")
    if is_key:
        print(f"✅ 判定结果: 【关键帧】")
    else:
        print(f"❌ 判定结果: 【非关键帧】")

    print(f"💯 总得分: {score}")
    print(f"📝 命中物体详情:")
    for obj in details:
        print(f"   - {obj['name']} (贡献分: {obj['score']}, 置信度: {obj['conf']})")