import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np

class YOLODetector:
    def __init__(self, model_version='yolov8n.pt'):
        self.model = YOLO(model_version)

    def crop_primary_item(self, image: Image.Image, gt_bbox=None) -> Image.Image:
        # Detects objects in the image and crops the bounding box of the most 
        # probable primary item (often a person or clothing item).
        # If gt_bbox is provided, uses it directly.
        # If no detection is confident enough, returns the original image.
        if gt_bbox is not None:
            # Assuming gt_bbox is (x_1, y_1, x_2, y_2)
            try:
                # Bboxes can be provided as strings from pandas, ensure int
                x1, y1, x2, y2 = [int(v) for v in gt_bbox]
                return image.crop((x1, y1, x2, y2))
            except Exception as e:
                pass # fallback to YOLO if gt_bbox parsing fails
                
        results = self.model(image, verbose=False)
        
        if not results or len(results[0].boxes) == 0:
            return image # No boxes detected, return original
            
        # Get the first result
        result = results[0]
        boxes = result.boxes
        
        # Heuristic: Find the box with the maximum area or highest confidence that could be relevant
        # YOLO COCO class 0 is 'person', which works well for fashion modelling.
        # Other items like 'tie' (27), 'backpack' (24), 'handbag' (26), 'suitcase' (28), 'shoe' (if available).
        
        best_box = None
        best_area = 0
        
        for box in boxes:
            conf = box.conf.item()
            cls_id = int(box.cls.item())
            xyxy = box.xyxy[0].cpu().numpy()
            
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            
            # Prioritize person or just take the largest bounding box over a confidence threshold
            if conf > 0.3 and area > best_area:
                best_area = area
                best_box = xyxy
                
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            cropped_image = image.crop((x1, y1, x2, y2))
            return cropped_image
            
        return image
