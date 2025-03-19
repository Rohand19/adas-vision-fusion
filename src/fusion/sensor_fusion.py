import numpy as np
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
from preprocessing.radar_processor import RadarObject

logger = logging.getLogger(__name__)

@dataclass
class FusedObject:
    """Data class for fused sensor data."""
    id: int
    camera_detection: Dict
    radar_detection: RadarObject
    fused_confidence: float
    distance: float
    velocity: float
    position: Tuple[float, float, float]  # x, y, z coordinates

class SensorFusion:
    def __init__(self):
        """Initialize the sensor fusion module."""
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.radar_to_camera_transform = None
        self.max_association_distance = 20.0  # meters
        self.confidence_weight_camera = 0.7
        self.confidence_weight_radar = 0.3
        logger.info("Sensor fusion module initialized")

    def set_camera_calibration(self, camera_matrix: np.ndarray, 
                             distortion_coeffs: np.ndarray):
        """Set camera calibration parameters."""
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs

    def set_radar_to_camera_transform(self, transform: np.ndarray):
        """Set the transformation matrix from radar to camera coordinates."""
        self.radar_to_camera_transform = transform

    def fuse(self, camera_objects: List[Dict], 
             radar_objects: List[RadarObject]) -> List[FusedObject]:
        """
        Fuse camera and radar detections.
        
        Args:
            camera_objects: List of camera detections
            radar_objects: List of radar detections
            
        Returns:
            List of fused objects
        """
        try:
            fused_objects = []
            used_radar_objects = set()
            
            logger.info(f"Starting fusion with {len(camera_objects)} camera objects and {len(radar_objects)} radar objects")
            
            for i, cam_obj in enumerate(camera_objects):
                # Convert camera bbox to 3D position estimate
                cam_pos = self._bbox_to_3d_position(cam_obj['bbox'])
                logger.info(f"Camera object {i} position: {cam_pos}")
                
                # Find matching radar object
                best_radar_obj = None
                min_distance = float('inf')
                
                for j, radar_obj in enumerate(radar_objects):
                    if j in used_radar_objects:
                        continue
                        
                    # Convert radar coordinates to camera frame
                    radar_pos = self._radar_to_camera_coordinates(
                        radar_obj.distance,
                        radar_obj.azimuth,
                        radar_obj.elevation
                    )
                    logger.info(f"Radar object {j} position: {radar_pos}")
                    
                    # Calculate distance between detections
                    dist = np.linalg.norm(cam_pos - radar_pos)
                    logger.info(f"Distance between camera {i} and radar {j}: {dist}")
                    
                    if dist < min_distance and dist < self.max_association_distance:
                        min_distance = dist
                        best_radar_obj = radar_obj
                        logger.info(f"Found better match: radar {j} at distance {dist}")
                
                if best_radar_obj is not None:
                    # Calculate fused confidence
                    fused_conf = (
                        self.confidence_weight_camera * cam_obj['confidence'] +
                        self.confidence_weight_radar * best_radar_obj.confidence
                    )
                    logger.info(f"Creating fused object with confidence {fused_conf}")
                    
                    # Create fused object
                    fused_obj = FusedObject(
                        id=i,
                        camera_detection=cam_obj,
                        radar_detection=best_radar_obj,
                        fused_confidence=fused_conf,
                        distance=best_radar_obj.distance,
                        velocity=best_radar_obj.velocity,
                        position=cam_pos
                    )
                    
                    fused_objects.append(fused_obj)
                    used_radar_objects.add(radar_objects.index(best_radar_obj))
            
            return fused_objects
            
        except Exception as e:
            logger.error(f"Error during sensor fusion: {str(e)}")
            return []

    def _bbox_to_3d_position(self, bbox: List[float]) -> np.ndarray:
        """
        Convert bounding box to 3D position estimate.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box coordinates
            
        Returns:
            3D position estimate [x, y, z] in meters
        """
        # Use bottom center of bbox as position estimate in pixels
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = bbox[3]  # Use bottom y-coordinate
        
        # Convert to normalized coordinates
        if self.camera_matrix is not None:
            # Convert to normalized coordinates using camera matrix
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]
            
            # Convert to normalized coordinates
            x = (center_x - cx) / fx
            y = (center_y - cy) / fy
        else:
            # Fallback to simple normalization
            x = (center_x - 320) / 1000  # Assuming focal length of 1000
            y = (center_y - 240) / 1000
        
        # Estimate depth using bbox height (assuming known object size)
        height = bbox[3] - bbox[1]
        z = 1.5 * 1000 / height  # Convert 1.5m to mm and divide by height in pixels
        
        # Scale x and y by z to get meters
        x = x * z
        y = y * z
        
        return np.array([x, y, z])

    def _radar_to_camera_coordinates(self, distance: float, 
                                   azimuth: float, 
                                   elevation: float) -> np.ndarray:
        """
        Convert radar coordinates to camera frame.
        
        Args:
            distance: Distance in meters
            azimuth: Azimuth angle in degrees
            elevation: Elevation angle in degrees
            
        Returns:
            3D position in camera coordinates
        """
        # Convert angles to radians
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        # Convert spherical to Cartesian coordinates
        x = distance * np.cos(el_rad) * np.sin(az_rad)
        y = distance * np.sin(el_rad)
        z = distance * np.cos(el_rad) * np.cos(az_rad)
        
        # Apply radar to camera transform if available
        if self.radar_to_camera_transform is not None:
            pos = np.array([x, y, z, 1])
            transformed = self.radar_to_camera_transform @ pos
            return transformed[:3]
        
        return np.array([x, y, z])

    def cleanup(self):
        """Clean up resources."""
        logger.info("Sensor fusion module cleaned up") 