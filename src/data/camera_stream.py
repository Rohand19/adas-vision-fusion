import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class CameraStream:
    def __init__(self, source: str = 0):
        """
        Initialize the camera stream.
        
        Args:
            source: Camera source (0 for default camera, or path to video file)
        """
        self.source = source
        self.cap = None
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30
        logger.info(f"Camera stream initialized with source: {source}")

    def start(self) -> bool:
        """
        Start the camera stream.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                logger.error("Failed to open camera stream")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            logger.info("Camera stream started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera stream: {str(e)}")
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get a frame from the camera stream.
        
        Returns:
            np.ndarray: Frame data, or None if failed
        """
        try:
            if self.cap is None:
                logger.error("Camera stream not started")
                return None
            
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame")
                return None
            
            return frame
            
        except Exception as e:
            logger.error(f"Error getting frame: {str(e)}")
            return None

    def get_frame_size(self) -> Tuple[int, int]:
        """
        Get the frame size.
        
        Returns:
            Tuple[int, int]: Width and height of the frame
        """
        return self.frame_width, self.frame_height

    def get_fps(self) -> int:
        """
        Get the frame rate.
        
        Returns:
            int: Frames per second
        """
        return self.fps

    def stop(self):
        """Stop the camera stream."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera stream stopped")

    def __del__(self):
        """Clean up resources."""
        self.stop() 