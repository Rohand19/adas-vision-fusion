import numpy as np
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
from ..fusion.sensor_fusion import FusedObject

logger = logging.getLogger(__name__)

@dataclass
class CollisionRisk:
    """Data class for collision risk assessment."""
    risk_level: float  # 0.0 to 1.0
    time_to_collision: float  # seconds
    recommended_action: str
    braking_force: float  # 0.0 to 1.0
    steering_angle: float  # degrees

class CollisionAvoidance:
    def __init__(self):
        """Initialize the collision avoidance module."""
        self.min_safe_distance = 2.0  # meters
        self.max_safe_velocity = 30.0  # m/s
        self.vehicle_length = 4.5  # meters
        self.reaction_time = 0.5  # seconds
        self.max_braking_force = 1.0
        self.max_steering_angle = 45.0  # degrees
        logger.info("Collision avoidance module initialized")

    def assess_risk(self, fused_objects: List[FusedObject]) -> CollisionRisk:
        """
        Assess collision risk for all detected objects.
        
        Args:
            fused_objects: List of fused sensor objects
            
        Returns:
            Overall collision risk assessment
        """
        try:
            max_risk = 0.0
            min_ttc = float('inf')
            most_critical_object = None
            
            for obj in fused_objects:
                risk, ttc = self._calculate_object_risk(obj)
                
                if risk > max_risk:
                    max_risk = risk
                    min_ttc = ttc
                    most_critical_object = obj
            
            if most_critical_object is None:
                return CollisionRisk(
                    risk_level=0.0,
                    time_to_collision=float('inf'),
                    recommended_action="maintain_course",
                    braking_force=0.0,
                    steering_angle=0.0
                )
            
            # Determine recommended action
            action, braking, steering = self._determine_action(
                most_critical_object,
                max_risk,
                min_ttc
            )
            
            return CollisionRisk(
                risk_level=max_risk,
                time_to_collision=min_ttc,
                recommended_action=action,
                braking_force=braking,
                steering_angle=steering
            )
            
        except Exception as e:
            logger.error(f"Error assessing collision risk: {str(e)}")
            return CollisionRisk(
                risk_level=1.0,
                time_to_collision=0.0,
                recommended_action="emergency_brake",
                braking_force=1.0,
                steering_angle=0.0
            )

    def _calculate_object_risk(self, obj: FusedObject) -> Tuple[float, float]:
        """
        Calculate collision risk for a single object.
        
        Args:
            obj: Fused sensor object
            
        Returns:
            Tuple of (risk_level, time_to_collision)
        """
        # Calculate time to collision
        if obj.velocity <= 0:
            return 0.0, float('inf')
        
        ttc = obj.distance / obj.velocity
        
        # Calculate risk based on distance and velocity
        distance_risk = max(0.0, 1.0 - (obj.distance / (self.min_safe_distance * 2)))
        velocity_risk = min(1.0, obj.velocity / self.max_safe_velocity)
        
        # Combine risks with weights
        risk = 0.7 * distance_risk + 0.3 * velocity_risk
        
        return risk, ttc

    def _determine_action(self, obj: FusedObject, risk: float, 
                         ttc: float) -> Tuple[str, float, float]:
        """
        Determine the appropriate action based on risk assessment.
        
        Args:
            obj: Most critical object
            risk: Collision risk level
            ttc: Time to collision
            
        Returns:
            Tuple of (action, braking_force, steering_angle)
        """
        if risk < 0.3:
            return "maintain_course", 0.0, 0.0
        elif risk < 0.6:
            # Moderate risk - gentle braking
            return "reduce_speed", 0.3, 0.0
        elif risk < 0.8:
            # High risk - strong braking
            return "strong_brake", 0.7, 0.0
        else:
            # Critical risk - emergency maneuver
            if obj.position[0] > 0:  # Object on right
                return "steer_left", 0.8, -self.max_steering_angle
            else:  # Object on left
                return "steer_right", 0.8, self.max_steering_angle

    def take_action(self, risk: CollisionRisk):
        """
        Execute the recommended action.
        
        Args:
            risk: Collision risk assessment
        """
        try:
            if risk.risk_level > 0.7:
                logger.warning(f"Emergency action required: {risk.recommended_action}")
                # TODO: Implement actual vehicle control
                # self._apply_braking(risk.braking_force)
                # self._apply_steering(risk.steering_angle)
            elif risk.risk_level > 0.3:
                logger.info(f"Taking preventive action: {risk.recommended_action}")
                # TODO: Implement preventive measures
                
        except Exception as e:
            logger.error(f"Error executing action: {str(e)}")

    def cleanup(self):
        """Clean up resources."""
        logger.info("Collision avoidance module cleaned up") 