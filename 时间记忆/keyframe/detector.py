import os
import cv2
import numpy as np

from ultralytics import YOLO

class YOLODetector:
    """
    YOLO 目标检测模型封装类
    """
    
    def __init__(self, model_path):
        """
        初始化 YOLO 模型
        
        Args:
            model_path: 模型文件路径
        """
        self.model = YOLO(model_path)
        print(f"模型加载成功: {model_path}")
    
    def detect(self, img_path, show=False, save=False, save_path="result.jpg"):
        """
        执行目标检测
        
        Args:
            img_path: 图像路径
            show: 是否显示结果
            save: 是否保存结果
            save_path: 保存路径
        
        Returns:
            dict: 检测结果字典
        """
        # 执行推理
        results = self.model(img_path, verbose=show)
        
        # 处理结果
        detection_results = []
        
        for result in results:
            # 构建结果字典
            result_dict = {
                "boxes": [],
                "masks": None,
                "keypoints": None,
                "probs": None,
                "obb": None
            }
            
            # 处理边界框
            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    box_dict = {
                        "xyxy": boxes.xyxy[i].tolist(),  # 边界框坐标
                        "conf": float(boxes.conf[i]),     # 置信度
                        "cls": int(boxes.cls[i]),         # 类别ID
                        "class_name": result.names[int(boxes.cls[i])]  # 类别名称
                    }
                    result_dict["boxes"].append(box_dict)
            
            # 处理掩码（如果有）
            if result.masks is not None:
                result_dict["masks"] = {
                    "shape": result.masks.data.shape,
                    "count": len(result.masks)
                }
            
            # 处理关键点（如果有）
            if result.keypoints is not None:
                result_dict["keypoints"] = {
                    "shape": result.keypoints.data.shape,
                    "count": len(result.keypoints)
                }
            
            # 处理分类结果（如果有）
            if result.probs is not None:
                result_dict["probs"] = {
                    "top1": float(result.probs.top1),
                    "top1_conf": float(result.probs.top1conf)
                }
            
            # 处理定向边界框（如果有）
            if result.obb is not None:
                result_dict["obb"] = {
                    "shape": result.obb.data.shape,
                    "count": len(result.obb)
                }
            
            detection_results.append(result_dict)
            
            # 显示结果
            if show:
                result.show()
            
            # 保存结果
            if save:
                result.save(filename=save_path)
        
        # 构建最终返回字典
        return {
            "image_path": img_path,
            "results": detection_results,
            "total_detections": sum(len(r["boxes"]) for r in detection_results)
        }


# 使用示例
if __name__ == "__main__":
    # 初始化检测器
    model_path = r'/Users/libin/LVNet/week3/yolo26n.pt'
    detector = YOLODetector(model_path)
    
    # 测试图像路径
    img_path = r'/Users/libin/LVNet/key_frames_20260417_152102/I_frame_000000.jpg'
    
    # 执行检测
    results = detector.detect(img_path, show=True, save=True)

    # 打印结果
    print("\n检测结果:")
    print(f"图像路径: {results['image_path']}")
    print(f"总检测数量: {results['total_detections']}")

    for i, result in enumerate(results['results']):
        print(f"\n结果 {i+1}:")
        print(f"边界框数量: {len(result['boxes'])}")

        for j, box in enumerate(result['boxes']):
            print(f"  目标 {j+1}: {box['class_name']} (置信度: {box['conf']:.2f})")
            print(f"    边界框: {box['xyxy']}")

