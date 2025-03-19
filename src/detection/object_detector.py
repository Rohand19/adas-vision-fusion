import torch
import numpy as np
import logging
from typing import List, Dict, Optional
import cv2
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self):
        """Initialize the object detector with YOLOv8."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conf_threshold = 0.1  # Lowered from 0.3 to catch more potential detections
        self.iou_threshold = 0.45
        self.model = None
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self.load_model()
        logger.info("Object detector initialized")

    def load_model(self):
        """Load the YOLOv5 model."""
        try:
            # Load YOLOv5 model
            self.model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model
            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            raise

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in the image.
        
        Args:
            image: Raw or preprocessed image in RGB format with shape (H, W, 3)
            
        Returns:
            List of detected objects with their properties
        """
        try:
            logger.info(f"Input image shape: {image.shape}, dtype: {image.dtype}, value range: [{image.min()}, {image.max()}]")
            
            # Ensure image is in uint8 format
            if image.dtype != np.uint8:
                logger.info(f"Converting image from {image.dtype} to uint8")
                image = (image * 255).astype(np.uint8)
            
            # Run inference
            results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold)
            logger.info(f"Found {len(results[0].boxes)} raw detections")
            
            # Log details of each detection
            for i, box in enumerate(results[0].boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                logger.info(f"Detection {i}: class={self.class_names[cls_id]}, confidence={conf:.4f}")
            
            # Process results
            detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': self.class_names[cls_id]
                })
            
            logger.info(f"Returning {len(detections)} filtered detections")
            return detections
            
        except Exception as e:
            logger.error(f"Error in detect: {str(e)}")
            return []

    def draw_detections(self, image: np.ndarray, 
                       detections: List[Dict]) -> np.ndarray:
        """
        Draw detection boxes on the image.
        
        Args:
            image: Original image
            detections: List of detected objects
            
        Returns:
            Image with detection boxes
        """
        try:
            image_with_boxes = image.copy()
            
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                conf = det['confidence']
                cls_name = det['class_name']
                
                # Draw box
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{cls_name}: {conf:.2f}"
                cv2.putText(image_with_boxes, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return image_with_boxes
            
        except Exception as e:
            logger.error(f"Error drawing detections: {str(e)}")
            return image

    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
        torch.cuda.empty_cache()
        logger.info("Object detector cleaned up") 