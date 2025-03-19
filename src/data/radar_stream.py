import numpy as np
import logging
from pathlib import Path
from typing import Optional, List
import json
import time

logger = logging.getLogger(__name__)

class RadarStream:
    def __init__(self, data_path: str = None):
        """
        Initialize the radar stream.
        
        Args:
            data_path: Path to radar data file or directory
        """
        self.data_path = Path(data_path) if data_path else None
        self.current_frame = 0
        self.data = None
        self.frame_rate = 10  # Hz
        self.last_frame_time = 0
        logger.info(f"Radar stream initialized with data path: {data_path}")

    def start(self) -> bool:
        """
        Start the radar stream.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.data_path is None:
                logger.error("No data path provided")
                return False
            
            if self.data_path.is_file():
                # Load from file
                self.data = self._load_from_file(self.data_path)
            elif self.data_path.is_dir():
                # Load from directory
                self.data = self._load_from_directory(self.data_path)
            else:
                logger.error(f"Invalid data path: {self.data_path}")
                return False
            
            if self.data is None:
                return False
            
            self.current_frame = 0
            self.last_frame_time = time.time()
            logger.info("Radar stream started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting radar stream: {str(e)}")
            return False

    def _load_from_file(self, file_path: Path) -> Optional[np.ndarray]:
        """
        Load radar data from a file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            np.ndarray: Loaded data, or None if failed
        """
        try:
            if file_path.suffix == '.npy':
                return np.load(file_path)
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                return np.array(data)
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading data file: {str(e)}")
            return None

    def _load_from_directory(self, dir_path: Path) -> Optional[np.ndarray]:
        """
        Load radar data from a directory.
        
        Args:
            dir_path: Path to the data directory
            
        Returns:
            np.ndarray: Loaded data, or None if failed
        """
        try:
            # Look for data files
            data_files = list(dir_path.glob('*.npy')) + list(dir_path.glob('*.json'))
            if not data_files:
                logger.error(f"No data files found in {dir_path}")
                return None
            
            # Load the first file
            return self._load_from_file(data_files[0])
            
        except Exception as e:
            logger.error(f"Error loading data directory: {str(e)}")
            return None

    def get_data(self) -> Optional[np.ndarray]:
        """
        Get radar data for the current frame.
        
        Returns:
            np.ndarray: Radar data, or None if failed
        """
        try:
            if self.data is None:
                logger.error("Radar stream not started")
                return None
            
            # Check frame rate
            current_time = time.time()
            if current_time - self.last_frame_time < 1.0 / self.frame_rate:
                return None
            
            # Get data for current frame
            if self.current_frame >= len(self.data):
                self.current_frame = 0  # Loop back to start
            
            frame_data = self.data[self.current_frame]
            self.current_frame += 1
            self.last_frame_time = current_time
            
            return frame_data
            
        except Exception as e:
            logger.error(f"Error getting radar data: {str(e)}")
            return None

    def get_frame_rate(self) -> int:
        """
        Get the frame rate.
        
        Returns:
            int: Frames per second
        """
        return self.frame_rate

    def stop(self):
        """Stop the radar stream."""
        self.data = None
        self.current_frame = 0
        logger.info("Radar stream stopped")

    def __del__(self):
        """Clean up resources."""
        self.stop() 