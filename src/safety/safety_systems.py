import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SafetySystemStatus(Enum):
    """Enum for safety system status."""
    INACTIVE = 0
    ACTIVE = 1
    WARNING = 2
    INTERVENTION = 3
    ERROR = 4

@dataclass
class SafetySystemState:
    """Data class for safety system state."""
    status: SafetySystemStatus
    target_velocity: float
    current_velocity: float
    target_distance: float
    current_distance: float
    steering_angle: float
    brake_pressure: float
    throttle_position: float
    warning_message: str = ""

class SafetySystems:
    """Implements safety systems including ACC and emergency braking."""
    
    def __init__(self):
        """Initialize safety systems."""
        self.params = {
            'min_safe_distance': 2.0,  # meters
            'max_acceleration': 2.0,  # m/s^2
            'max_deceleration': 4.0,  # m/s^2
            'max_steering_rate': 0.5,  # rad/s
            'acc_time_gap': 1.5,  # seconds
            'brake_reaction_time': 0.2,  # seconds
            'min_velocity': 0.0,  # m/s
            'max_velocity': 30.0,  # m/s
            'steering_angle_limit': 0.5  # radians
        }
        self.state = SafetySystemState(
            status=SafetySystemStatus.INACTIVE,
            target_velocity=0.0,
            current_velocity=0.0,
            target_distance=self.params['min_safe_distance'],
            current_distance=float('inf'),
            steering_angle=0.0,
            brake_pressure=0.0,
            throttle_position=0.0
        )
        logger.info("Safety systems initialized")
    
    def update(self, tracked_objects: List[Dict], ego_velocity: float, 
               steering_angle: float) -> Dict:
        """Update safety systems state and generate control commands.
        
        Args:
            tracked_objects: List of tracked objects
            ego_velocity: Current vehicle velocity
            steering_angle: Current steering angle
            
        Returns:
            Dictionary of control commands
        """
        # Update system state
        self.state.current_velocity = ego_velocity
        self.state.steering_angle = steering_angle
        
        # Find closest object in front
        closest_object = self._find_closest_object(tracked_objects)
        
        if closest_object is None:
            self.state.status = SafetySystemStatus.ACTIVE
            return self._generate_normal_control()
        
        # Update current distance
        self.state.current_distance = closest_object['distance']
        
        # Check for emergency situations
        if self._is_emergency_situation(closest_object):
            self.state.status = SafetySystemStatus.INTERVENTION
            return self._generate_emergency_control(closest_object)
        
        # Check for warning conditions
        if self._is_warning_situation(closest_object):
            self.state.status = SafetySystemStatus.WARNING
            return self._generate_warning_control(closest_object)
        
        # Normal operation
        self.state.status = SafetySystemStatus.ACTIVE
        return self._generate_normal_control()
    
    def _find_closest_object(self, tracked_objects: List[Dict]) -> Optional[Dict]:
        """Find closest object in front of the vehicle.
        
        Args:
            tracked_objects: List of tracked objects
            
        Returns:
            Closest object or None if no objects found
        """
        if not tracked_objects:
            return None
        
        # Filter objects in front (assuming forward direction is positive x)
        front_objects = [
            obj for obj in tracked_objects 
            if obj.get('position', [0, 0])[0] > 0
        ]
        
        if not front_objects:
            return None
        
        # Find closest object
        return min(front_objects, key=lambda obj: obj.get('distance', float('inf')))
    
    def _is_emergency_situation(self, obj: Dict) -> bool:
        """Check if current situation requires emergency intervention.
        
        Args:
            obj: Closest object
            
        Returns:
            True if emergency intervention is needed
        """
        if obj is None:
            return False
        
        distance = obj.get('distance', float('inf'))
        relative_velocity = obj.get('relative_velocity', 0.0)
        
        # Emergency conditions
        if distance < self.params['min_safe_distance'] * 0.5:
            return True
        
        if relative_velocity < -5.0 and distance < self.params['min_safe_distance']:
            return True
        
        return False
    
    def _is_warning_situation(self, obj: Dict) -> bool:
        """Check if current situation requires warning.
        
        Args:
            obj: Closest object
            
        Returns:
            True if warning is needed
        """
        if obj is None:
            return False
        
        distance = obj.get('distance', float('inf'))
        relative_velocity = obj.get('relative_velocity', 0.0)
        
        # Warning conditions
        if distance < self.params['min_safe_distance'] * 1.5:
            return True
        
        if relative_velocity < -3.0 and distance < self.params['min_safe_distance'] * 2:
            return True
        
        return False
    
    def _generate_emergency_control(self, obj: Dict) -> Dict:
        """Generate emergency control commands.
        
        Args:
            obj: Closest object
            
        Returns:
            Dictionary of control commands
        """
        # Emergency braking
        brake_pressure = 1.0
        throttle_position = 0.0
        
        # Calculate evasive steering
        steering_angle = self._calculate_evasive_steering(obj)
        
        # Update state
        self.state.brake_pressure = brake_pressure
        self.state.throttle_position = throttle_position
        self.state.steering_angle = steering_angle
        self.state.warning_message = "Emergency braking and steering"
        
        return {
            'throttle': throttle_position,
            'brake': brake_pressure,
            'steer': steering_angle
        }
    
    def _generate_warning_control(self, obj: Dict) -> Dict:
        """Generate warning control commands.
        
        Args:
            obj: Closest object
            
        Returns:
            Dictionary of control commands
        """
        # Moderate braking
        brake_pressure = 0.5
        throttle_position = 0.3
        
        # Maintain current steering
        steering_angle = self.state.steering_angle
        
        # Update state
        self.state.brake_pressure = brake_pressure
        self.state.throttle_position = throttle_position
        self.state.steering_angle = steering_angle
        self.state.warning_message = "Warning: Maintain safe distance"
        
        return {
            'throttle': throttle_position,
            'brake': brake_pressure,
            'steer': steering_angle
        }
    
    def _generate_normal_control(self) -> Dict:
        """Generate normal control commands.
        
        Returns:
            Dictionary of control commands
        """
        # Adaptive Cruise Control
        if self.state.current_velocity < self.state.target_velocity:
            throttle_position = min(1.0, self.state.throttle_position + 0.1)
            brake_pressure = 0.0
        else:
            throttle_position = max(0.0, self.state.throttle_position - 0.1)
            brake_pressure = 0.0
        
        # Maintain current steering
        steering_angle = self.state.steering_angle
        
        # Update state
        self.state.brake_pressure = brake_pressure
        self.state.throttle_position = throttle_position
        self.state.steering_angle = steering_angle
        self.state.warning_message = ""
        
        return {
            'throttle': throttle_position,
            'brake': brake_pressure,
            'steer': steering_angle
        }
    
    def _calculate_evasive_steering(self, obj: Dict) -> float:
        """Calculate evasive steering angle.
        
        Args:
            obj: Closest object
            
        Returns:
            Steering angle in radians
        """
        # Simple evasive steering based on object position
        # This should be replaced with more sophisticated path planning
        if obj.get('position', [0, 0])[1] > 0:
            return self.params['steering_angle_limit']
        else:
            return -self.params['steering_angle_limit']
    
    def set_target_velocity(self, velocity: float):
        """Set target velocity for ACC.
        
        Args:
            velocity: Target velocity in m/s
        """
        self.state.target_velocity = np.clip(
            velocity,
            self.params['min_velocity'],
            self.params['max_velocity']
        )
    
    def set_target_distance(self, distance: float):
        """Set target following distance for ACC.
        
        Args:
            distance: Target distance in meters
        """
        self.state.target_distance = max(
            distance,
            self.params['min_safe_distance']
        )
    
    def get_state(self) -> SafetySystemState:
        """Get current safety system state.
        
        Returns:
            Current system state
        """
        return self.state
    
    def cleanup(self):
        """Clean up resources."""
        self.state = SafetySystemState(
            status=SafetySystemStatus.INACTIVE,
            target_velocity=0.0,
            current_velocity=0.0,
            target_distance=self.params['min_safe_distance'],
            current_distance=float('inf'),
            steering_angle=0.0,
            brake_pressure=0.0,
            throttle_position=0.0
        )
        logger.info("Safety systems cleaned up") 