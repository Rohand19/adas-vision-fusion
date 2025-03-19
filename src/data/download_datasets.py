import os
import sys
import logging
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    def __init__(self):
        """Initialize the dataset downloader."""
        self.data_dir = Path(__file__).parent.parent.parent / 'data'
        self.data_dir.mkdir(exist_ok=True)
        self.nuscenes_dir = self.data_dir / 'nuscenes'
        self.nuscenes_dir.mkdir(exist_ok=True)
        
        # nuScenes dataset URLs (mini dataset for testing)
        self.urls = {
            'camera': 'https://www.nuscenes.org/data/v1.0-mini.tgz',
            'radar': 'https://www.nuscenes.org/data/v1.0-mini.tgz'
        }
        
        logger.info("Dataset downloader initialized")

    def download_file(self, url: str, filename: str):
        """
        Download a file with progress bar.
        
        Args:
            url: URL to download from
            filename: Name to save the file as
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
                    
        except Exception as e:
            logger.error(f"Error downloading {filename}: {str(e)}")
            raise

    def extract_archive(self, archive_path: str, extract_path: str):
        """
        Extract a zip/tar archive.
        
        Args:
            archive_path: Path to the archive file
            extract_path: Path to extract to
        """
        try:
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
            elif archive_path.endswith('.tgz'):
                import tarfile
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(extract_path)
            else:
                raise ValueError(f"Unsupported archive format: {archive_path}")
                
        except Exception as e:
            logger.error(f"Error extracting {archive_path}: {str(e)}")
            raise

    def prepare_dataset(self):
        """Download and prepare the nuScenes dataset."""
        try:
            logger.info("Starting dataset preparation")
            
            # Download camera data
            camera_archive = self.nuscenes_dir / 'camera_data.tgz'
            if not camera_archive.exists():
                logger.info("Downloading camera data...")
                self.download_file(self.urls['camera'], str(camera_archive))
            
            # Download radar data
            radar_archive = self.nuscenes_dir / 'radar_data.tgz'
            if not radar_archive.exists():
                logger.info("Downloading radar data...")
                self.download_file(self.urls['radar'], str(radar_archive))
            
            # Extract archives
            logger.info("Extracting archives...")
            self.extract_archive(str(camera_archive), str(self.nuscenes_dir / 'camera'))
            self.extract_archive(str(radar_archive), str(self.nuscenes_dir / 'radar'))
            
            # Clean up archives
            logger.info("Cleaning up...")
            camera_archive.unlink()
            radar_archive.unlink()
            
            logger.info("Dataset preparation completed successfully")
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            raise

def main():
    """Main function to download and prepare the dataset."""
    try:
        downloader = DatasetDownloader()
        downloader.prepare_dataset()
    except Exception as e:
        logger.error(f"Failed to prepare dataset: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 