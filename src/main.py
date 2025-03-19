import os
import sys
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.preprocessing.camera_processor import CameraProcessor
from src.preprocessing.radar_processor import RadarProcessor
from src.detection.object_detector import ObjectDetector
from src.detection.road_detector import RoadDetector
from src.fusion.sensor_fusion import SensorFusion
from src.safety.collision_avoidance import CollisionAvoidance
from src.visualization.visualizer import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ADASSystem:
    """Main ADAS system class that coordinates all components."""
    
    def __init__(self, video_path: str, radar_data_path: str):
        """Initialize the ADAS system.
        
        Args:
            video_path: Path to the input video file
            radar_data_path: Path to the radar data file
        """
        self.video_path = video_path
        self.radar_data_path = radar_data_path
        
        # Initialize components
        self.camera_processor = CameraProcessor()
        self.radar_processor = RadarProcessor()
        self.object_detector = ObjectDetector()
        self.road_detector = RoadDetector()
        self.sensor_fusion = SensorFusion()
        self.collision_avoidance = CollisionAvoidance()
        self.visualizer = Visualizer()
        
        logger.info("ADAS system initialized successfully")
    
    def process_frame(self, frame: np.ndarray, radar_data: Dict) -> Dict:
        """Process a single frame with all components.
        
        Args:
            frame: Input video frame
            radar_data: Radar data for the current frame
            
        Returns:
            Dictionary containing processing results
        """
        try:
            # Process camera data
            processed_frame = self.camera_processor.preprocess(frame)
            camera_objects = self.object_detector.detect(processed_frame)
            
            # Process radar data
            radar_objects = self.radar_processor.process(radar_data)
            
            # Detect road infrastructure
            road_data = self.road_detector.process_frame(frame)
            
            # Fuse sensor data
            fused_objects = self.sensor_fusion.fuse(camera_objects, radar_objects)
            
            # Assess collision risk
            risk_assessment = self.collision_avoidance.assess_risk(
                fused_objects,
                road_data
            )
            
            return {
                'camera_objects': camera_objects,
                'radar_objects': radar_objects,
                'fused_objects': fused_objects,
                'road_data': road_data,
                'risk_assessment': risk_assessment
            }
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return {
                'camera_objects': [],
                'radar_objects': [],
                'fused_objects': [],
                'road_data': {'lanes': [], 'boundaries': [], 'signs': []},
                'risk_assessment': {'risk_level': 0.0, 'warnings': []}
            }
    
    def run(self):
        """Run the ADAS system."""
        try:
            # Open video capture
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError("Failed to open video file")
            
            # Load radar data
            radar_data = self.radar_processor.load_data(self.radar_data_path)
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video reached")
                    break
                
                # Get radar data for current frame
                current_radar_data = radar_data.get(frame_count, {})
                
                # Process frame
                results = self.process_frame(frame, current_radar_data)
                
                # Visualize results
                visualization = self.visualizer.visualize(
                    frame,
                    results
                )
                
                # Display results
                cv2.imshow('ADAS System', visualization)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
            
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Cleanup components
            self.camera_processor.cleanup()
            self.radar_processor.cleanup()
            self.object_detector.cleanup()
            self.road_detector.cleanup()
            self.sensor_fusion.cleanup()
            self.collision_avoidance.cleanup()
            self.visualizer.cleanup()
            
            logger.info("ADAS system stopped successfully")
            
        except Exception as e:
            logger.error(f"Error running ADAS system: {str(e)}")
            raise

def main():
    """Main entry point for the ADAS system."""
    try:
        # Define paths
        video_path = str(Path('data/test_video.mp4'))
        radar_data_path = str(Path('data/radar_data.json'))
        
        # Create and run system
        system = ADASSystem(video_path, radar_data_path)
        system.run()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main() 