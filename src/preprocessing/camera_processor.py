import cv2
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class CameraProcessor:
    def __init__(self):
        """Initialize the camera processor."""
        self.image_size = (640, 480)  # Standard size for processing
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        logger.info("Camera processor initialized")

    def preprocess(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess the camera image for object detection.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image ready for model input, or None if preprocessing fails.
            Output will be RGB format with shape (H, W, 3).
        """
        try:
            if image is None or image.size == 0:
                logger.error("Empty image received")
                return None

            logger.info(f"Input image shape: {image.shape}")
            logger.info(f"Input image dtype: {image.dtype}")
            logger.info(f"Input image value range: [{image.min()}, {image.max()}]")

            if len(image.shape) != 3:
                logger.error(f"Invalid image dimensions: expected 3 dimensions, got {len(image.shape)}")
                return None

            # Check if image is valid
            height, width = image.shape[:2]
            if height == 0 or width == 0:
                logger.error(f"Invalid image size: {height}x{width}")
                return None

            # Ensure uint8 format
            if image.dtype != np.uint8:
                logger.info(f"Converting image from {image.dtype} to uint8")
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
                logger.info(f"Converted image value range: [{image.min()}, {image.max()}]")

            # Resize image
            image = cv2.resize(image, self.image_size)
            logger.info(f"Resized image shape: {image.shape}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.info("Converted image from BGR to RGB")
            logger.info(f"Final image shape: {image.shape}")
            logger.info(f"Final image dtype: {image.dtype}")
            logger.info(f"Final image value range: [{image.min()}, {image.max()}]")
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None

    def postprocess(self, 
                   image: np.ndarray,
                   detections: np.ndarray,
                   confidence_threshold: float = 0.5) -> Tuple[np.ndarray, list]:
        """
        Postprocess the detection results.
        
        Args:
            image: Original image
            detections: Model predictions
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            Processed image with bounding boxes and list of detected objects
        """
        try:
            # Filter detections by confidence
            mask = detections[:, 4] > confidence_threshold
            detections = detections[mask]
            
            # Convert detections to list of objects
            objects = []
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                objects.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': int(cls)
                })
            
            # Draw bounding boxes
            image_with_boxes = image.copy()
            for obj in objects:
                x1, y1, x2, y2 = map(int, obj['bbox'])
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"{obj['class']}: {obj['confidence']:.2f}"
                cv2.putText(image_with_boxes, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return image_with_boxes, objects
            
        except Exception as e:
            logger.error(f"Error postprocessing detections: {str(e)}")
            return image, []

    def cleanup(self):
        """Clean up resources."""
        logger.info("Camera processor cleaned up") 