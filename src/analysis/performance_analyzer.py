import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Data class for performance metrics."""
    frame_rate: float
    detection_time: float
    tracking_time: float
    fusion_time: float
    decision_time: float
    total_time: float
    memory_usage: float
    cpu_usage: float
    gpu_usage: Optional[float] = None
    detection_accuracy: float = 0.0
    tracking_accuracy: float = 0.0
    false_positives: int = 0
    false_negatives: int = 0
    missed_detections: int = 0
    incorrect_tracks: int = 0

class PerformanceAnalyzer:
    """Analyzes system performance and suggests improvements."""
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.metrics_history: List[PerformanceMetrics] = []
        self.timing_data = defaultdict(list)
        self.resource_usage = defaultdict(list)
        self.detection_stats = defaultdict(int)
        self.tracking_stats = defaultdict(int)
        self.window_size = 100  # frames for moving average
        logger.info("Performance analyzer initialized")
    
    def update(self, metrics: PerformanceMetrics):
        """Update performance metrics.
        
        Args:
            metrics: Current performance metrics
        """
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.metrics_history) > self.window_size:
            self.metrics_history.pop(0)
        
        # Update timing data
        self.timing_data['detection'].append(metrics.detection_time)
        self.timing_data['tracking'].append(metrics.tracking_time)
        self.timing_data['fusion'].append(metrics.fusion_time)
        self.timing_data['decision'].append(metrics.decision_time)
        self.timing_data['total'].append(metrics.total_time)
        
        # Update resource usage
        self.resource_usage['memory'].append(metrics.memory_usage)
        self.resource_usage['cpu'].append(metrics.cpu_usage)
        if metrics.gpu_usage is not None:
            self.resource_usage['gpu'].append(metrics.gpu_usage)
        
        # Update detection and tracking stats
        self.detection_stats['false_positives'] += metrics.false_positives
        self.detection_stats['false_negatives'] += metrics.false_negatives
        self.detection_stats['missed_detections'] += metrics.missed_detections
        self.tracking_stats['incorrect_tracks'] += metrics.incorrect_tracks
    
    def get_performance_summary(self) -> Dict:
        """Get summary of current performance metrics.
        
        Returns:
            Dictionary containing performance summary
        """
        if not self.metrics_history:
            return {}
        
        # Calculate moving averages
        avg_metrics = {
            'frame_rate': np.mean([m.frame_rate for m in self.metrics_history]),
            'detection_time': np.mean([m.detection_time for m in self.metrics_history]),
            'tracking_time': np.mean([m.tracking_time for m in self.metrics_history]),
            'fusion_time': np.mean([m.fusion_time for m in self.metrics_history]),
            'decision_time': np.mean([m.decision_time for m in self.metrics_history]),
            'total_time': np.mean([m.total_time for m in self.metrics_history]),
            'memory_usage': np.mean([m.memory_usage for m in self.metrics_history]),
            'cpu_usage': np.mean([m.cpu_usage for m in self.metrics_history]),
            'detection_accuracy': np.mean([m.detection_accuracy for m in self.metrics_history]),
            'tracking_accuracy': np.mean([m.tracking_accuracy for m in self.metrics_history])
        }
        
        # Add GPU usage if available
        if self.metrics_history[0].gpu_usage is not None:
            avg_metrics['gpu_usage'] = np.mean([m.gpu_usage for m in self.metrics_history])
        
        # Add cumulative stats
        avg_metrics.update({
            'total_false_positives': self.detection_stats['false_positives'],
            'total_false_negatives': self.detection_stats['false_negatives'],
            'total_missed_detections': self.detection_stats['missed_detections'],
            'total_incorrect_tracks': self.tracking_stats['incorrect_tracks']
        })
        
        return avg_metrics
    
    def analyze_bottlenecks(self) -> List[Dict]:
        """Analyze system bottlenecks.
        
        Returns:
            List of bottleneck descriptions and suggestions
        """
        bottlenecks = []
        
        # Get current performance metrics
        metrics = self.get_performance_summary()
        
        # Analyze frame rate
        if metrics['frame_rate'] < 30:
            bottlenecks.append({
                'component': 'Overall System',
                'issue': 'Low frame rate',
                'current_value': f"{metrics['frame_rate']:.1f} FPS",
                'target_value': '30 FPS',
                'suggestions': [
                    'Optimize detection model',
                    'Reduce processing resolution',
                    'Implement parallel processing'
                ]
            })
        
        # Analyze detection time
        if metrics['detection_time'] > 50:  # ms
            bottlenecks.append({
                'component': 'Object Detection',
                'issue': 'Slow detection time',
                'current_value': f"{metrics['detection_time']:.1f} ms",
                'target_value': '50 ms',
                'suggestions': [
                    'Use lighter model architecture',
                    'Implement model quantization',
                    'Optimize preprocessing pipeline'
                ]
            })
        
        # Analyze tracking performance
        if metrics['tracking_accuracy'] < 0.8:
            bottlenecks.append({
                'component': 'Object Tracking',
                'issue': 'Low tracking accuracy',
                'current_value': f"{metrics['tracking_accuracy']:.2f}",
                'target_value': '0.8',
                'suggestions': [
                    'Improve track association algorithm',
                    'Add motion prediction',
                    'Implement track confidence scoring'
                ]
            })
        
        # Analyze resource usage
        if metrics['cpu_usage'] > 80:
            bottlenecks.append({
                'component': 'CPU Usage',
                'issue': 'High CPU utilization',
                'current_value': f"{metrics['cpu_usage']:.1f}%",
                'target_value': '80%',
                'suggestions': [
                    'Optimize CPU-intensive operations',
                    'Implement multi-threading',
                    'Use hardware acceleration'
                ]
            })
        
        if metrics['memory_usage'] > 1000:  # MB
            bottlenecks.append({
                'component': 'Memory Usage',
                'issue': 'High memory consumption',
                'current_value': f"{metrics['memory_usage']:.1f} MB",
                'target_value': '1000 MB',
                'suggestions': [
                    'Implement memory pooling',
                    'Optimize data structures',
                    'Reduce buffer sizes'
                ]
            })
        
        return bottlenecks
    
    def get_optimization_suggestions(self) -> List[Dict]:
        """Generate optimization suggestions based on performance analysis.
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        metrics = self.get_performance_summary()
        
        # Detection optimization suggestions
        if metrics['detection_time'] > 50:
            suggestions.append({
                'area': 'Detection',
                'priority': 'High',
                'suggestion': 'Implement model quantization to reduce inference time',
                'expected_improvement': '30-50% reduction in detection time'
            })
        
        if metrics['detection_accuracy'] < 0.8:
            suggestions.append({
                'area': 'Detection',
                'priority': 'Medium',
                'suggestion': 'Fine-tune model on specific domain data',
                'expected_improvement': '5-10% increase in detection accuracy'
            })
        
        # Tracking optimization suggestions
        if metrics['tracking_accuracy'] < 0.8:
            suggestions.append({
                'area': 'Tracking',
                'priority': 'High',
                'suggestion': 'Implement Kalman filter for motion prediction',
                'expected_improvement': '15-20% increase in tracking accuracy'
            })
        
        if metrics['total_incorrect_tracks'] > 100:
            suggestions.append({
                'area': 'Tracking',
                'priority': 'Medium',
                'suggestion': 'Improve track association algorithm',
                'expected_improvement': 'Reduction in incorrect tracks'
            })
        
        # Resource optimization suggestions
        if metrics['cpu_usage'] > 80:
            suggestions.append({
                'area': 'Resource Usage',
                'priority': 'High',
                'suggestion': 'Implement multi-threading for parallel processing',
                'expected_improvement': '20-30% reduction in CPU usage'
            })
        
        if metrics['memory_usage'] > 1000:
            suggestions.append({
                'area': 'Resource Usage',
                'priority': 'Medium',
                'suggestion': 'Implement memory pooling for dynamic allocation',
                'expected_improvement': '30-40% reduction in memory usage'
            })
        
        return suggestions
    
    def generate_report(self) -> Dict:
        """Generate comprehensive performance report.
        
        Returns:
            Dictionary containing performance report
        """
        return {
            'metrics': self.get_performance_summary(),
            'bottlenecks': self.analyze_bottlenecks(),
            'suggestions': self.get_optimization_suggestions(),
            'timestamp': time.time()
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.metrics_history.clear()
        self.timing_data.clear()
        self.resource_usage.clear()
        self.detection_stats.clear()
        self.tracking_stats.clear()
        logger.info("Performance analyzer cleaned up") 