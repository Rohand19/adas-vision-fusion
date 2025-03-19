# Advanced Driver Assistance System (ADAS)

A comprehensive ADAS implementation that combines computer vision, radar processing, and sensor fusion to provide advanced safety features for vehicles.

## Features

### 1. Sensor Processing
- **Camera Processing**
  - Image preprocessing and normalization
  - Resolution adjustment
  - Color space conversion
  - Frame rate optimization

- **Radar Processing**
  - Point cloud processing
  - Velocity estimation
  - Field of view generation
  - Road infrastructure detection

### 2. Object Detection and Tracking
- **Object Detection**
  - YOLOv8-based object detection
  - Multiple object class support
  - Confidence scoring
  - Real-time processing

- **Object Tracking**
  - Contour-based tracking
  - Motion prediction
  - Track association
  - Velocity estimation

- **Road Detection**
  - Lane detection
  - Road boundary detection
  - Traffic sign recognition
  - Infrastructure mapping

### 3. Sensor Fusion
- Multi-sensor data fusion
- Confidence-weighted fusion
- Position and velocity estimation
- Track association

### 4. Safety Systems
- **Collision Avoidance**
  - Time-to-collision estimation
  - Risk assessment
  - Warning generation
  - Evasive maneuver planning

- **Safety Features**
  - Adaptive Cruise Control (ACC)
  - Emergency braking
  - Lane keeping assistance
  - Blind spot monitoring

### 5. Performance Analysis
- Real-time performance monitoring
- Bottleneck detection
- Resource usage tracking
- Optimization suggestions

## Project Structure

```
adas/
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── camera_processor.py    # Camera image processing
│   │   └── radar_processor.py     # Radar data processing
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── object_detector.py     # YOLOv8 object detection
│   │   ├── object_tracker.py      # Contour-based tracking
│   │   └── road_detector.py       # Road infrastructure detection
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── sensor_fusion.py       # Multi-sensor fusion
│   ├── decision/
│   │   ├── __init__.py
│   │   └── collision_avoidance.py # Collision risk assessment
│   ├── safety/
│   │   ├── __init__.py
│   │   └── safety_systems.py      # ACC and safety features
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── performance_analyzer.py # Performance monitoring
│   └── visualization/
│       ├── __init__.py
│       └── visualizer.py          # Results visualization
├── models/
│   └── yolov8n.pt                 # YOLOv8 model weights
├── data/
│   ├── test_video.mp4             # Test video data
│   └── radar_data.json            # Test radar data
├── docs/
│   ├── architecture.md            # System architecture
│   └── api.md                     # API documentation
├── tests/
│   └── test_*.py                  # Unit tests
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
└── main_nuscenes.py              # Main entry point
```

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- PyTorch
- Ultralytics (YOLOv8)
- SciPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Rohand19/adas-vision-fusion.git
cd adas-vision-fusion
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the YOLOv8 model:
```bash
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt
```

## Usage

1. Run the main script:
```bash
python src/main_nuscenes.py
```

2. The system will:
   - Initialize all components
   - Process camera and radar data
   - Detect and track objects
   - Generate safety warnings
   - Display visualization
   - Provide performance metrics

## Performance Metrics

- Frame Rate: ~30 FPS
- Detection Time: ~30ms
- Tracking Time: ~5ms
- Fusion Time: ~2ms
- Decision Time: ~1ms
- Total Processing Time: ~38ms

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## Acknowledgments

- YOLOv8 by Ultralytics
- nuScenes dataset
- OpenCV community
- PyTorch team 