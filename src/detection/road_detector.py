import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class RoadDetector:
    """Detects road infrastructure including lanes, boundaries, and signs."""
    
    def __init__(self):
        """Initialize the road detector with default parameters."""
        # Lane detection parameters
        self.lane_detection_params = {
            'canny_low': 50,
            'canny_high': 150,
            'hough_threshold': 20,
            'min_line_length': 20,
            'max_line_gap': 300
        }
        
        # Road boundary detection parameters
        self.boundary_params = {
            'blur_kernel': (5, 5),
            'edge_threshold': 100,
            'min_contour_area': 1000
        }
        
        # Traffic sign detection parameters
        self.sign_params = {
            'min_area': 100,
            'max_area': 10000,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0
        }
        
        logger.info("Road detector initialized with default parameters")
    
    def detect_lanes(self, image: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Detect lane lines in the image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of lane lines as (start_point, end_point) tuples
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 
                            self.lane_detection_params['canny_low'],
                            self.lane_detection_params['canny_high'])
            
            # Define region of interest
            height, width = edges.shape
            roi_vertices = np.array([
                [(0, height),
                 (width//4, height//2),
                 (3*width//4, height//2),
                 (width, height)]
            ], dtype=np.int32)
            
            # Create mask for ROI
            mask = np.zeros_like(edges)
            cv2.fillPoly(mask, roi_vertices, 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # Hough transform for line detection
            lines = cv2.HoughLinesP(
                masked_edges,
                rho=1,
                theta=np.pi/180,
                threshold=self.lane_detection_params['hough_threshold'],
                minLineLength=self.lane_detection_params['min_line_length'],
                maxLineGap=self.lane_detection_params['max_line_gap']
            )
            
            if lines is None:
                logger.debug("No lane lines detected")
                return []
            
            # Separate left and right lanes
            left_lines = []
            right_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                
                if slope < 0:
                    left_lines.append(line[0])
                else:
                    right_lines.append(line[0])
            
            # Average the lines
            lanes = []
            if left_lines:
                left_avg = self._average_lines(left_lines)
                lanes.append(left_avg)
            if right_lines:
                right_avg = self._average_lines(right_lines)
                lanes.append(right_avg)
            
            logger.info(f"Detected {len(lanes)} lane lines")
            return lanes
            
        except Exception as e:
            logger.error(f"Error in lane detection: {str(e)}")
            return []
    
    def detect_boundaries(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect road boundaries in the image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of boundary contours
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, self.boundary_params['blur_kernel'], 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, self.boundary_params['edge_threshold'])
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            valid_contours = [
                contour for contour in contours
                if cv2.contourArea(contour) > self.boundary_params['min_contour_area']
            ]
            
            logger.info(f"Detected {len(valid_contours)} road boundaries")
            return valid_contours
            
        except Exception as e:
            logger.error(f"Error in boundary detection: {str(e)}")
            return []
    
    def detect_signs(self, image: np.ndarray) -> List[Dict]:
        """Detect traffic signs in the image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of detected signs with their properties
        """
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for different signs
            red_lower = np.array([0, 100, 100])
            red_upper = np.array([10, 255, 255])
            blue_lower = np.array([100, 100, 100])
            blue_upper = np.array([130, 255, 255])
            
            # Create masks for different colors
            red_mask = cv2.inRange(hsv, red_lower, red_upper)
            blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
            
            # Combine masks
            combined_mask = cv2.bitwise_or(red_mask, blue_mask)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            signs = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.sign_params['min_area'] <= area <= self.sign_params['max_area']:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    if self.sign_params['min_aspect_ratio'] <= aspect_ratio <= self.sign_params['max_aspect_ratio']:
                        signs.append({
                            'contour': contour,
                            'bbox': (x, y, w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio
                        })
            
            logger.info(f"Detected {len(signs)} traffic signs")
            return signs
            
        except Exception as e:
            logger.error(f"Error in sign detection: {str(e)}")
            return []
    
    def _average_lines(self, lines: List[Tuple[int, int, int, int]]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Average multiple lines into a single line.
        
        Args:
            lines: List of lines as (x1, y1, x2, y2) tuples
            
        Returns:
            Averaged line as (start_point, end_point) tuple or None if invalid
        """
        if not lines:
            return None
            
        try:
            # Convert to numpy array for easier computation
            lines = np.array(lines)
            
            # Calculate slopes, handling vertical lines
            dx = lines[:, 2] - lines[:, 0]
            dy = lines[:, 3] - lines[:, 1]
            
            # Filter out vertical or near-vertical lines
            valid_lines = np.abs(dx) > 1e-6
            if not np.any(valid_lines):
                return None
                
            slopes = dy[valid_lines] / dx[valid_lines]
            intercepts = lines[valid_lines, 1] - slopes * lines[valid_lines, 0]
            
            # Check for valid slopes and intercepts
            valid_indices = ~np.isnan(slopes) & ~np.isnan(intercepts)
            if not np.any(valid_indices):
                return None
                
            avg_slope = np.mean(slopes[valid_indices])
            avg_intercept = np.mean(intercepts[valid_indices])
            
            # Calculate start and end points
            y1 = int(lines[:, 1].max())
            y2 = int(lines[:, 3].min())
            
            # Ensure y1 > y2 to maintain consistent line direction
            if y1 <= y2:
                y1, y2 = y2, y1
            
            # Calculate x coordinates
            x1 = int((y1 - avg_intercept) / avg_slope)
            x2 = int((y2 - avg_intercept) / avg_slope)
            
            return ((x1, y1), (x2, y2))
            
        except Exception as e:
            logger.error(f"Error in averaging lines: {str(e)}")
            return None
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect road infrastructure in the image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Dictionary containing detected road infrastructure
        """
        try:
            # Detect lanes, boundaries, and signs
            lanes = self.detect_lanes(image)
            boundaries = self.detect_boundaries(image)
            signs = self.detect_signs(image)
            
            # Convert boundaries to list format
            boundary_list = []
            for boundary in boundaries:
                boundary_list.append(boundary.reshape(-1, 2).tolist())
            
            # Convert signs to list format
            sign_list = []
            for sign in signs:
                x, y, w, h = sign['bbox']
                sign_list.append([x, y, w, h])
            
            return {
                'lanes': lanes,
                'boundaries': boundary_list,
                'signs': sign_list
            }
            
        except Exception as e:
            logger.error(f"Error in road detection: {str(e)}")
            return {
                'lanes': [],
                'boundaries': [],
                'signs': []
            }
    
    def process_frame(self, image: np.ndarray) -> Dict:
        """Process a single frame to detect all road infrastructure.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Dictionary containing all detected infrastructure
        """
        try:
            lanes = self.detect_lanes(image)
            boundaries = self.detect_boundaries(image)
            signs = self.detect_signs(image)
            
            return {
                'lanes': lanes,
                'boundaries': boundaries,
                'signs': signs
            }
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return {
                'lanes': [],
                'boundaries': [],
                'signs': []
            }
    
    def cleanup(self):
        """Clean up resources."""
        pass 