import os
import sys
import logging
import unittest
import numpy as np
import cv2
from pathlib import Path

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.preprocessing.camera_processor import CameraProcessor
from src.preprocessing.radar_processor import RadarProcessor, RadarObject
from src.detection.object_detector import ObjectDetector
from src.fusion.sensor_fusion import SensorFusion, FusedObject
from src.decision.collision_avoidance import CollisionAvoidance, CollisionRisk
from src.visualization.visualizer import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestADAS(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.camera_processor = CameraProcessor()
        self.radar_processor = RadarProcessor()
        self.object_detector = ObjectDetector()
        self.sensor_fusion = SensorFusion()
        self.collision_avoidance = CollisionAvoidance()
        self.visualizer = Visualizer()

    def tearDown(self):
        """Clean up after tests."""
        self.camera_processor.cleanup()
        self.radar_processor.cleanup()
        self.object_detector.cleanup()
        self.visualizer.cleanup()

    def test_camera_processor(self):
        """Test camera processing functionality."""
        # Create a test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[100:200, 100:200] = 255  # White rectangle
        
        # Test preprocessing
        processed = self.camera_processor.preprocess(test_image)
        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape, (480, 640, 3))  # No batch dimension
        self.assertEqual(processed.dtype, np.uint8)

    def test_radar_processor(self):
        """Test radar processing functionality."""
        # Create test radar data
        test_data = np.array([
            [10.0, 5.0, 45.0, 0.0, 1.0, 0.8],  # Valid detection
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.1],    # Invalid detection
            [100.0, 30.0, 0.0, 0.0, 1.0, 0.9]   # Valid detection
        ])
        
        # Test processing
        objects = self.radar_processor.process(test_data)
        self.assertEqual(len(objects), 2)  # Only valid detections
        
        # Test filtering
        filtered = self.radar_processor.filter_by_confidence(objects, 0.85)
        self.assertEqual(len(filtered), 1)  # Only high confidence detection

    def test_object_detector(self):
        """Test object detection functionality."""
        # Create a test image with a simple shape
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)
        
        # Test detection
        detections = self.object_detector.detect(test_image)
        self.assertIsInstance(detections, list)

    def test_sensor_fusion(self):
        """Test sensor fusion functionality."""
        # Set up calibration parameters
        camera_matrix = np.array([
            [1000, 0, 320],
            [0, 1000, 240],
            [0, 0, 1]
        ])
        distortion_coeffs = np.zeros(5)
        radar_to_camera_transform = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        self.sensor_fusion.set_camera_calibration(camera_matrix, distortion_coeffs)
        self.sensor_fusion.set_radar_to_camera_transform(radar_to_camera_transform)
        
        # Create test camera detection
        camera_detection = {
            'bbox': [320, 240, 400, 320],  # Center of image
            'confidence': 0.8,
            'class': 2,  # car
            'class_name': 'car'
        }
        
        # Create test radar object
        radar_object = RadarObject(
            id=0,
            distance=10.0,
            velocity=5.0,
            azimuth=0.0,  # Straight ahead
            elevation=0.0,
            rcs=1.0,
            confidence=0.7
        )
        
        # Test fusion
        fused_objects = self.sensor_fusion.fuse(
            [camera_detection],
            [radar_object]
        )
        self.assertEqual(len(fused_objects), 1)

    def test_collision_avoidance(self):
        """Test collision avoidance functionality."""
        # Create test fused object
        fused_object = FusedObject(
            id=0,
            camera_detection={
                'bbox': [320, 240, 400, 320],
                'confidence': 0.8,
                'class': 2,
                'class_name': 'car'
            },
            radar_detection=RadarObject(
                id=0, distance=5.0, velocity=10.0,
                azimuth=0.0, elevation=0.0, rcs=1.0, confidence=0.7
            ),
            fused_confidence=0.75,
            distance=5.0,
            velocity=10.0,
            position=(0.0, 0.0, 5.0)
        )
        
        # Test risk assessment
        risk = self.collision_avoidance.assess_risk([fused_object])
        self.assertIsNotNone(risk)
        self.assertGreaterEqual(risk.risk_level, 0.0)
        self.assertLessEqual(risk.risk_level, 1.0)

    def test_visualizer(self):
        """Test visualization functionality."""
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create test fused object
        fused_object = FusedObject(
            id=0,
            camera_detection={
                'bbox': [320, 240, 400, 320],
                'confidence': 0.8,
                'class': 2,
                'class_name': 'car'
            },
            radar_detection=RadarObject(
                id=0, distance=5.0, velocity=10.0,
                azimuth=0.0, elevation=0.0, rcs=1.0, confidence=0.7
            ),
            fused_confidence=0.75,
            distance=5.0,
            velocity=10.0,
            position=(0.0, 0.0, 5.0)
        )
        
        # Create test collision risk
        risk = CollisionRisk(
            risk_level=0.5,
            time_to_collision=2.0,
            recommended_action="reduce_speed",
            braking_force=0.3,
            steering_angle=0.0
        )
        
        # Create results dictionary
        results = {
            "fused_objects": [fused_object],
            "road_data": {
                "lanes": [(0, 0, 100, 100)],
                "boundaries": [np.array([[0, 0], [100, 0], [100, 100], [0, 100]])],
                "signs": [{"bbox": (50, 50, 20, 20), "type": "stop"}]
            },
            "risk_assessment": {
                "risk_level": risk.risk_level,
                "warnings": [risk.recommended_action]
            }
        }
        
        # Test visualization
        vis_image = self.visualizer.visualize(test_image, results)
        self.assertEqual(vis_image.shape, (480, 640, 3))

def main():
    """Run all tests."""
    unittest.main(verbosity=2)

if __name__ == "__main__":
    main() 