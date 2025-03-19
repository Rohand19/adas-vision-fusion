import os
import logging
import time
from preprocessing.camera_processor import CameraProcessor
from preprocessing.radar_processor import RadarProcessor
from preprocessing.nuscenes_loader import NuScenesLoader
from detection.object_detector import ObjectDetector
from detection.road_detector import RoadDetector
from fusion.sensor_fusion import SensorFusion
from safety.safety_systems import SafetySystems
from analysis.performance_analyzer import PerformanceAnalyzer, PerformanceMetrics
from visualization.visualizer import Visualizer
import cv2

class ADASNuScenes:
    def __init__(self, nuscenes_dataroot: str):
        """
        Initialize ADAS system with nuScenes data
        Args:
            nuscenes_dataroot: Path to nuScenes dataset
        """
        # Initialize components
        self.camera_processor = CameraProcessor()
        self.radar_processor = RadarProcessor()
        self.object_detector = ObjectDetector()
        self.road_detector = RoadDetector()
        self.sensor_fusion = SensorFusion()
        self.safety_systems = SafetySystems()
        self.performance_analyzer = PerformanceAnalyzer()
        self.visualizer = Visualizer()
        
        # Initialize nuScenes loader
        self.nuscenes_loader = NuScenesLoader(nuscenes_dataroot)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize performance tracking
        self.frame_count = 0
        self.start_time = None
        
    def process_scene(self):
        """
        Process a complete scene from nuScenes dataset
        """
        self.start_time = time.time()
        
        try:
            while True:
                frame_start_time = time.time()
                
                # Get camera data
                camera_img, camera_calib = self.nuscenes_loader.get_camera_data()
                if camera_img is None:
                    break
                    
                # Get radar data
                radar_points, radar_metadata = self.nuscenes_loader.get_radar_data()
                
                # Process camera image
                processed_img = self.camera_processor.preprocess(camera_img)
                
                # Detect objects
                detections_start = time.time()
                detections = self.object_detector.detect(processed_img)
                detection_time = (time.time() - detections_start) * 1000
                
                # Initialize metrics
                metrics = PerformanceMetrics(
                    frame_rate=0.0,
                    detection_time=detection_time,
                    tracking_time=0.0,
                    fusion_time=0.0,
                    decision_time=0.0,
                    total_time=0.0,
                    memory_usage=0.0,
                    cpu_usage=0.0,
                    gpu_usage=None,
                    detection_accuracy=0.9,
                    tracking_accuracy=0.85,
                    false_positives=0,
                    false_negatives=0,
                    missed_detections=0,
                    incorrect_tracks=0
                )
                
                # Process radar data if available
                if radar_points is not None:
                    radar_start = time.time()
                    processed_points = self.radar_processor.process(radar_points)
                    radar_time = (time.time() - radar_start) * 1000
                    
                    # Detect road infrastructure
                    road_features = self.road_detector.detect(processed_img)
                    
                    # Perform sensor fusion
                    fused_data = self.sensor_fusion.fuse(
                        camera_detections=detections,
                        radar_points=processed_points,
                        road_features=road_features
                    )
                    
                    # Update safety systems
                    safety_start = time.time()
                    safety_commands = self.safety_systems.update(
                        tracked_objects=fused_data,
                        ego_velocity=30.0,  # TODO: Get actual vehicle velocity
                        steering_angle=0.0  # TODO: Get actual steering angle
                    )
                    safety_time = (time.time() - safety_start) * 1000
                    
                    # Update metrics
                    metrics.fusion_time = radar_time
                    metrics.decision_time = safety_time
                    
                    # Visualize results
                    vis_img = self.visualizer.visualize(processed_img, {
                        'road_data': {
                            'lanes': road_features.get('lanes', []),
                            'boundaries': road_features.get('boundaries', []),
                            'signs': road_features.get('signs', [])
                        },
                        'fused_objects': fused_data,
                        'risk_assessment': {
                            'risk_level': 0.0,
                            'warnings': []
                        },
                        'safety_state': self.safety_systems.get_state(),
                        'performance_metrics': metrics
                    })
                else:
                    # Visualize without radar data
                    vis_img = self.visualizer.visualize(processed_img, {
                        'road_data': {},
                        'fused_objects': detections,
                        'risk_assessment': {
                            'risk_level': 0.0,
                            'warnings': []
                        },
                        'safety_state': self.safety_systems.get_state(),
                        'performance_metrics': metrics
                    })
                
                # Calculate total processing time and update frame rate
                total_time = (time.time() - frame_start_time) * 1000
                metrics.total_time = total_time
                metrics.frame_rate = 1000.0 / total_time
                
                # Update performance analyzer
                self.performance_analyzer.update(metrics)
                
                # Display results
                cv2.imshow('ADAS NuScenes', cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Move to next sample
                if not self.nuscenes_loader.next_sample():
                    break
                    
                self.frame_count += 1
                    
        except Exception as e:
            self.logger.error(f"Error processing scene: {str(e)}")
            raise
        finally:
            cv2.destroyAllWindows()
            
            # Generate final performance report
            report = self.performance_analyzer.generate_report()
            self.logger.info("Performance Report:")
            self.logger.info(f"Total frames processed: {self.frame_count}")
            self.logger.info(f"Average frame rate: {report['metrics']['frame_rate']:.1f} FPS")
            self.logger.info(f"Average detection time: {report['metrics']['detection_time']:.1f} ms")
            
            # Log bottlenecks
            for bottleneck in report['bottlenecks']:
                self.logger.warning(f"Bottleneck detected in {bottleneck['component']}: {bottleneck['issue']}")
                self.logger.warning(f"Current value: {bottleneck['current_value']}, Target: {bottleneck['target_value']}")
                self.logger.warning("Suggestions:")
                for suggestion in bottleneck['suggestions']:
                    self.logger.warning(f"- {suggestion}")
            
    def cleanup(self):
        """
        Clean up resources
        """
        self.camera_processor.cleanup()
        self.radar_processor.cleanup()
        self.object_detector.cleanup()
        self.road_detector.cleanup()
        self.sensor_fusion.cleanup()
        self.safety_systems.cleanup()
        self.performance_analyzer.cleanup()
        self.visualizer.cleanup()

def main():
    # Setup paths
    nuscenes_dataroot = os.path.join(os.getcwd(), 'data/nuscenes')
    
    # Initialize ADAS system
    adas = ADASNuScenes(nuscenes_dataroot)
    
    try:
        # Process scene
        adas.process_scene()
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
    finally:
        # Cleanup
        adas.cleanup()

if __name__ == "__main__":
    main() 