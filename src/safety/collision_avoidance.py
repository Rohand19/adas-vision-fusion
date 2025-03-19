import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class CollisionAvoidance:
    """Collision avoidance system that assesses risks and provides warnings."""
    
    def __init__(self):
        """Initialize the collision avoidance system."""
        # Risk assessment parameters
        self.risk_params = {
            'min_distance': 2.0,  # meters
            'max_speed': 30.0,    # m/s
            'time_horizon': 2.0,  # seconds
            'lane_deviation_threshold': 0.5,  # meters
            'sign_importance': {
                'stop': 1.0,
                'yield': 0.8,
                'speed_limit': 0.6
            }
        }
        
        # Vehicle parameters
        self.vehicle_params = {
            'length': 4.5,  # meters
            'width': 2.0,   # meters
            'max_acceleration': 2.0,  # m/s^2
            'max_deceleration': -4.0  # m/s^2
        }
        
        logger.info("Collision avoidance system initialized")
    
    def assess_risk(self, fused_objects: List[Dict], road_data: Dict) -> Dict:
        """Assess collision risk based on fused objects and road infrastructure.
        
        Args:
            fused_objects: List of fused objects from sensor fusion
            road_data: Dictionary containing road infrastructure data
            
        Returns:
            Dictionary containing risk assessment results
        """
        try:
            warnings = []
            risk_level = 0.0
            
            # Check for immediate collision risks
            collision_risk = self._check_collision_risk(fused_objects)
            if collision_risk['risk_level'] > 0.0:
                warnings.extend(collision_risk['warnings'])
                risk_level = max(risk_level, collision_risk['risk_level'])
            
            # Check lane deviation
            lane_risk = self._check_lane_deviation(road_data['lanes'])
            if lane_risk['risk_level'] > 0.0:
                warnings.extend(lane_risk['warnings'])
                risk_level = max(risk_level, lane_risk['risk_level'])
            
            # Check traffic signs
            sign_risk = self._check_traffic_signs(road_data['signs'])
            if sign_risk['risk_level'] > 0.0:
                warnings.extend(sign_risk['warnings'])
                risk_level = max(risk_level, sign_risk['risk_level'])
            
            # Check road boundaries
            boundary_risk = self._check_boundaries(road_data['boundaries'])
            if boundary_risk['risk_level'] > 0.0:
                warnings.extend(boundary_risk['warnings'])
                risk_level = max(risk_level, boundary_risk['risk_level'])
            
            return {
                'risk_level': risk_level,
                'warnings': warnings
            }
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {str(e)}")
            return {
                'risk_level': 0.0,
                'warnings': []
            }
    
    def _check_collision_risk(self, fused_objects: List[Dict]) -> Dict:
        """Check for potential collisions with detected objects.
        
        Args:
            fused_objects: List of fused objects
            
        Returns:
            Dictionary containing collision risk assessment
        """
        warnings = []
        risk_level = 0.0
        
        for obj in fused_objects:
            # Calculate time to collision
            distance = obj.get('distance', float('inf'))
            relative_velocity = obj.get('relative_velocity', 0.0)
            
            if distance < self.risk_params['min_distance']:
                warnings.append(f"Immediate collision risk with {obj.get('class_name', 'object')}")
                risk_level = 1.0
                continue
            
            if relative_velocity > 0:  # Object is moving towards vehicle
                ttc = distance / relative_velocity
                if ttc < self.risk_params['time_horizon']:
                    warnings.append(f"Potential collision with {obj.get('class_name', 'object')} in {ttc:.1f}s")
                    risk_level = max(risk_level, 0.8)
            
            # Check for high-speed objects
            if abs(relative_velocity) > self.risk_params['max_speed']:
                warnings.append(f"High-speed object detected: {obj.get('class_name', 'object')}")
                risk_level = max(risk_level, 0.6)
        
        return {
            'risk_level': risk_level,
            'warnings': warnings
        }
    
    def _check_lane_deviation(self, lanes: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Dict:
        """Check for lane deviation based on detected lane lines.
        
        Args:
            lanes: List of detected lane lines
            
        Returns:
            Dictionary containing lane deviation risk assessment
        """
        warnings = []
        risk_level = 0.0
        
        if not lanes:
            warnings.append("No lane lines detected")
            risk_level = 0.3
            return {'risk_level': risk_level, 'warnings': warnings}
        
        # Calculate vehicle position relative to lanes
        vehicle_position = self._estimate_vehicle_position()
        lane_deviation = self._calculate_lane_deviation(vehicle_position, lanes)
        
        if lane_deviation > self.risk_params['lane_deviation_threshold']:
            warnings.append(f"Lane deviation detected: {lane_deviation:.1f}m")
            risk_level = 0.7
        
        return {
            'risk_level': risk_level,
            'warnings': warnings
        }
    
    def _check_traffic_signs(self, signs: List[Dict]) -> Dict:
        """Check for relevant traffic signs and assess compliance.
        
        Args:
            signs: List of detected traffic signs
            
        Returns:
            Dictionary containing traffic sign risk assessment
        """
        warnings = []
        risk_level = 0.0
        
        for sign in signs:
            # Extract sign type and importance
            sign_type = self._classify_sign(sign)
            importance = self.risk_params['sign_importance'].get(sign_type, 0.0)
            
            if importance > 0.0:
                warnings.append(f"Traffic sign detected: {sign_type}")
                risk_level = max(risk_level, importance)
        
        return {
            'risk_level': risk_level,
            'warnings': warnings
        }
    
    def _check_boundaries(self, boundaries: List[np.ndarray]) -> Dict:
        """Check for potential road boundary violations.
        
        Args:
            boundaries: List of road boundary contours
            
        Returns:
            Dictionary containing boundary violation risk assessment
        """
        warnings = []
        risk_level = 0.0
        
        if not boundaries:
            warnings.append("No road boundaries detected")
            risk_level = 0.3
            return {'risk_level': risk_level, 'warnings': warnings}
        
        # Calculate distance to nearest boundary
        vehicle_position = self._estimate_vehicle_position()
        min_boundary_distance = self._calculate_boundary_distance(vehicle_position, boundaries)
        
        if min_boundary_distance < self.risk_params['min_distance']:
            warnings.append(f"Close to road boundary: {min_boundary_distance:.1f}m")
            risk_level = 0.6
        
        return {
            'risk_level': risk_level,
            'warnings': warnings
        }
    
    def _estimate_vehicle_position(self) -> Tuple[float, float]:
        """Estimate the vehicle's position in the world frame.
        
        Returns:
            Tuple of (x, y) coordinates
        """
        # This is a simplified estimation
        # In a real system, this would use vehicle state estimation
        return (0.0, 0.0)
    
    def _calculate_lane_deviation(self, vehicle_pos: Tuple[float, float], 
                                lanes: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> float:
        """Calculate the vehicle's deviation from the lane center.
        
        Args:
            vehicle_pos: Vehicle position
            lanes: List of lane lines
            
        Returns:
            Lane deviation in meters
        """
        if not lanes:
            return float('inf')
        
        # Simplified calculation
        # In a real system, this would use more sophisticated lane tracking
        return 0.0
    
    def _calculate_boundary_distance(self, vehicle_pos: Tuple[float, float],
                                   boundaries: List[np.ndarray]) -> float:
        """Calculate the minimum distance to any road boundary.
        
        Args:
            vehicle_pos: Vehicle position
            boundaries: List of boundary contours
            
        Returns:
            Minimum distance to boundary in meters
        """
        if not boundaries:
            return float('inf')
        
        # Simplified calculation
        # In a real system, this would use proper distance calculation
        return 5.0
    
    def _classify_sign(self, sign: Dict) -> str:
        """Classify a detected traffic sign.
        
        Args:
            sign: Dictionary containing sign properties
            
        Returns:
            Classified sign type
        """
        # Simplified classification
        # In a real system, this would use proper sign classification
        return "stop"
    
    def cleanup(self):
        """Clean up resources."""
        pass 