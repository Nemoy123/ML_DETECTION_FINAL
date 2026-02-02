from streamlit import image
from ultralytics import YOLO
import numpy as np
from PIL import Image
import json
from typing import List, Dict, Any
from app.config import settings
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel
from torchvision.ops import nms

import os
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
import torch
torch.set_num_threads(6)
torch.set_num_interop_threads(6)
import cv2
cv2.setNumThreads(6)

def nms_torch_class_agnostic(boxes, iou_threshold=0.6):
    if len(boxes) == 0:
        return []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    boxes_xyxy = torch.tensor(
        [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes],
        dtype=torch.float32,
        device=device
    )

    scores = torch.tensor(
        [b["confidence"] for b in boxes],
        dtype=torch.float32,
        device=device
    )

    keep = nms(boxes_xyxy, scores, iou_threshold)

    return [boxes[i] for i in keep.tolist()]

class YOLOModel:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YOLOModel, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance
    
    def _initialize_model(self):
        """Инициализация модели"""
        print(f"Loading YOLO model: {settings.MODEL_NAME}")
        self.model= UltralyticsDetectionModel(
            model_path=settings.MODEL_NAME,  # Ваши веса
            confidence_threshold=0.25,
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            image_size=640 
        )
        self.class_names = self.model.model.names
        print(self.class_names)
        print("Model loaded successfully")
    
    def predict_image(self, image_path: str) -> Dict[str, Any]:
        """Предсказание на изображении"""
        try:
            
            # Запуск предсказания 

            results = get_sliced_prediction(
                image_path,
                self.model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )
            
            boxes = []
            # Получение информации о изображении
            img = Image.open(image_path)
            
            img_w, img_h = img.size

            for obj in results.object_prediction_list:
                class_id = obj.category.id

                x1 = obj.bbox.minx
                y1 = obj.bbox.miny
                x2 = obj.bbox.maxx
                y2 = obj.bbox.maxy

                # xyxy → xywh
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                width = x2 - x1
                height = y2 - y1

                # нормализация под исходное изображение
                x_center /= img_w
                y_center /= img_h
                width /= img_w
                height /= img_h
                boxes.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": float(obj.score.value),
                    "class": class_id,
                    "class_name": self.class_names.get(class_id, str(class_id))
                })
            
            boxes = nms_torch_class_agnostic(boxes, iou_threshold=0.85)
            
            return {
                "success": True,
                "image_size": {
                    "width": img.width,
                    "height": img.height
                },
                "detections": boxes,
                "detections_count": len(boxes),
                "model": settings.MODEL_NAME
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """Предсказание на байтах изображения"""
        try:
            # Конвертация bytes в numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"success": False, "error": "Invalid image data"}
            
            # Сохранение временного файла
            temp_path = os.path.join(settings.CACHE_DIR, "temp.jpg")
            cv2.imwrite(temp_path, img)
            
            # Предсказание
            result = self.predict_image(temp_path)
            
            # Удаление временного файла
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# Глобальный экземпляр модели
yolo_model = YOLOModel()