import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class Visualizer:
    """Visualizes ADAS system outputs including objects, road infrastructure, and warnings."""
    
    def __init__(self):
        """Initialize the visualizer with default parameters."""
        # Visualization parameters
        self.params = {
            'box_thickness': 2,
            'text_thickness': 1,
            'text_scale': 0.5,
            'warning_height': 30,
            'warning_padding': 10
        }
        
        # Color definitions
        self.colors = {
            # Vehicles
            'car': (255, 0, 0),         # Blue
            'truck': (0, 0, 255),       # Red
            'bus': (255, 165, 0),       # Orange
            'motorcycle': (128, 0, 128), # Purple
            
            # People and Animals
            'person': (0, 255, 0),      # Green
            'dog': (0, 128, 0),         # Dark Green
            
            # Traffic Infrastructure
            'traffic light': (255, 255, 0),  # Yellow
            'stop sign': (0, 255, 255),      # Cyan
            'parking meter': (255, 0, 255),  # Magenta
            
            # Road Infrastructure
            'lane': (255, 255, 0),      # Yellow
            'boundary': (0, 255, 255),  # Cyan
            'sign': (255, 0, 255),      # Magenta
            
            # Other Objects
            'skateboard': (128, 128, 0),    # Olive
            'vase': (0, 128, 128),          # Teal
            'warning': (0, 0, 255),         # Red
            'text': (0,0,0)         # White
        }
        
        logger.info("Visualizer initialized")
    
    def visualize(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Visualize all ADAS system outputs on the frame.
        
        Args:
            frame: Input video frame
            results: Dictionary containing all processing results
            
        Returns:
            Frame with visualizations
        """
        try:
            # Create a copy of the frame
            vis_frame = frame.copy()
            
            # Draw road infrastructure
            road_data = results.get('road_data', {})
            if road_data:
                # Draw lanes
                for lane in road_data.get('lanes', []):
                    if len(lane) == 2:  # Format: ((x1, y1), (x2, y2))
                        start_point = tuple(map(int, lane[0]))
                        end_point = tuple(map(int, lane[1]))
                        cv2.line(vis_frame, start_point, end_point, self.colors['lane'], 
                               self.params['box_thickness'])
                
                # Draw boundaries
                for boundary in road_data.get('boundaries', []):
                    try:
                        boundary_array = np.array(boundary, dtype=np.int32)
                        cv2.drawContours(vis_frame, [boundary_array], -1, self.colors['boundary'],
                                       self.params['box_thickness'])
                    except Exception as e:
                        logger.warning(f"Failed to draw boundary: {e}")
                        continue
                
                # Draw traffic signs
                for sign in road_data.get('signs', []):
                    try:
                        x, y, w, h = map(int, sign)
                        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), self.colors['sign'],
                                    self.params['box_thickness'])
                    except Exception as e:
                        logger.warning(f"Failed to draw sign: {e}")
                        continue
            
            # Draw detected objects
            fused_objects = results.get('fused_objects', [])
            if isinstance(fused_objects, list):
                for obj in fused_objects:
                    # Handle both FusedObject instances and regular dictionaries
                    if hasattr(obj, 'camera_detection'):  # FusedObject instance
                        bbox = obj.camera_detection.get('bbox', None)
                        class_name = obj.camera_detection.get('class_name', 'unknown')
                        confidence = obj.fused_confidence
                    else:  # Regular dictionary
                        bbox = obj.get('bbox', None)
                        class_name = obj.get('class_name', 'unknown')
                        confidence = obj.get('confidence', 0.0)
                    
                    if bbox is not None:
                        x1, y1, x2, y2 = map(int, bbox)
                        color = self.colors.get(class_name, self.colors['text'])
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, self.params['box_thickness'])
                        
                        # Draw label with class name and confidence
                        label = f"{class_name} {confidence:.2f}"
                        
                        # Background for text
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, self.params['text_scale'],
                            self.params['text_thickness']
                        )
                        cv2.rectangle(vis_frame, (x1, y1 - text_height - 5),
                                    (x1 + text_width, y1), color, -1)
                        
                        # Text
                        cv2.putText(vis_frame, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, self.params['text_scale'],
                                  self.colors['text'], self.params['text_thickness'])
            
            return vis_frame
            
        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")
            return frame
    
    def _draw_risk_assessment(self, frame: np.ndarray, risk_assessment: Dict):
        """Draw risk assessment information on the frame.
        
        Args:
            frame: Frame to draw on
            risk_assessment: Dictionary containing risk assessment results
        """
        height, width = frame.shape[:2]
        risk_level = risk_assessment.get('risk_level', 0.0)
        warnings = risk_assessment.get('warnings', [])
        
        # Draw risk level indicator
        risk_color = self._get_risk_color(risk_level)
        cv2.rectangle(frame, (10, 10), (100, 30), risk_color, -1)
        cv2.putText(frame, f"Risk: {risk_level:.2f}", (15, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, self.params['text_scale'],
                   self.colors['text'], self.params['text_thickness'])
        
        # Draw warnings
        y_offset = 40
        for warning in warnings:
            # Background for warning
            (text_width, text_height), _ = cv2.getTextSize(
                warning, cv2.FONT_HERSHEY_SIMPLEX, self.params['text_scale'],
                self.params['text_thickness']
            )
            cv2.rectangle(frame, (10, y_offset),
                         (10 + text_width, y_offset + text_height + 5),
                         self.colors['warning'], -1)
            
            # Warning text
            cv2.putText(frame, warning, (10, y_offset + text_height),
                       cv2.FONT_HERSHEY_SIMPLEX, self.params['text_scale'],
                       self.colors['text'], self.params['text_thickness'])
            
            y_offset += text_height + 10
    
    def _get_risk_color(self, risk_level: float) -> Tuple[int, int, int]:
        """Get color based on risk level.
        
        Args:
            risk_level: Risk level between 0 and 1
            
        Returns:
            BGR color tuple
        """
        if risk_level >= 0.8:
            return (0, 0, 255)  # Red
        elif risk_level >= 0.5:
            return (0, 165, 255)  # Orange
        elif risk_level >= 0.2:
            return (0, 255, 255)  # Yellow
        else:
            return (0, 255, 0)  # Green
    
    def cleanup(self):
        """Clean up resources."""
        pass 