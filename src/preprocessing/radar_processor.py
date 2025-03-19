import numpy as np
import logging
import json
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RadarObject:
    """Data class for radar-detected objects."""
    id: int
    distance: float  # Distance in meters
    velocity: float  # Velocity in m/s
    azimuth: float  # Azimuth angle in degrees
    elevation: float  # Elevation angle in degrees
    rcs: float  # Radar Cross Section
    confidence: float  # Detection confidence

class RadarProcessor:
    def __init__(self):
        """Initialize the radar processor."""
        self.min_distance = 0.1  # meters
        self.max_distance = 200.0  # meters
        self.min_velocity = -50.0  # m/s
        self.max_velocity = 50.0  # m/s
        self.confidence_threshold = 0.3
        logger.info("Radar processor initialized")

    def load_data(self, file_path: str) -> Dict:
        """
        Load radar data from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing radar data
            
        Returns:
            Dictionary containing radar data for each timestamp
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded radar data from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading radar data from {file_path}: {str(e)}")
            return {}

    def process(self, radar_data: Union[Dict, np.ndarray]) -> List[RadarObject]:
        """
        Process radar data to extract object information.
        
        Args:
            radar_data: Either a dictionary containing radar data or a numpy array
            
        Returns:
            List of detected radar objects
        """
        try:
            objects = []
            
            if isinstance(radar_data, dict):
                # Process JSON format data
                for obj in radar_data.get("objects", []):
                    if (self.min_distance <= obj["distance"] <= self.max_distance and
                        self.min_velocity <= obj["velocity"] <= self.max_velocity and
                        obj["confidence"] >= self.confidence_threshold):
                        
                        radar_obj = RadarObject(
                            id=obj["id"],
                            distance=float(obj["distance"]),
                            velocity=float(obj["velocity"]),
                            azimuth=float(obj["angle"]),  # angle in JSON maps to azimuth
                            elevation=0.0,  # Not provided in JSON
                            rcs=1.0,  # Not provided in JSON
                            confidence=float(obj["confidence"])
                        )
                        objects.append(radar_obj)
            
            elif isinstance(radar_data, np.ndarray):
                # Process numpy array format
                valid_mask = (
                    (radar_data[:, 0] >= self.min_distance) &
                    (radar_data[:, 0] <= self.max_distance) &
                    (radar_data[:, 1] >= self.min_velocity) &
                    (radar_data[:, 1] <= self.max_velocity) &
                    (radar_data[:, 5] >= self.confidence_threshold)
                )
                
                filtered_data = radar_data[valid_mask]
                
                for i, data in enumerate(filtered_data):
                    obj = RadarObject(
                        id=i,
                        distance=float(data[0]),
                        velocity=float(data[1]),
                        azimuth=float(data[2]),
                        elevation=float(data[3]),
                        rcs=float(data[4]),
                        confidence=float(data[5])
                    )
                    objects.append(obj)
            
            else:
                logger.error(f"Unsupported radar data type: {type(radar_data)}")
                return []
            
            return objects
            
        except Exception as e:
            logger.error(f"Error processing radar data: {str(e)}")
            return []

    def filter_by_confidence(self, objects: List[RadarObject], 
                           threshold: float = 0.5) -> List[RadarObject]:
        """
        Filter radar objects by confidence threshold.
        
        Args:
            objects: List of radar objects
            threshold: Confidence threshold
            
        Returns:
            Filtered list of radar objects
        """
        return [obj for obj in objects if obj.confidence >= threshold]

    def get_closest_objects(self, objects: List[RadarObject], 
                          max_distance: float = 50.0) -> List[RadarObject]:
        """
        Get objects within a maximum distance.
        
        Args:
            objects: List of radar objects
            max_distance: Maximum distance in meters
            
        Returns:
            List of objects within max_distance
        """
        return [obj for obj in objects if obj.distance <= max_distance]

    def get_moving_objects(self, objects: List[RadarObject], 
                         min_velocity: float = 0.5) -> List[RadarObject]:
        """
        Get objects moving faster than a minimum velocity.
        
        Args:
            objects: List of radar objects
            min_velocity: Minimum velocity in m/s
            
        Returns:
            List of moving objects
        """
        return [obj for obj in objects if abs(obj.velocity) >= min_velocity]

    def cleanup(self):
        """Clean up resources."""
        logger.info("Radar processor cleaned up") 