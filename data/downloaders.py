"""
Data downloaders for various sources including AWS S3 and local datasets.

This module provides utilities to download and manage plant bloom datasets
from cloud storage and other sources.
"""

import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import requests
from pathlib import Path
import zipfile
import tarfile
from typing import Optional, List, Dict, Any, Callable
import pandas as pd
from tqdm import tqdm
import logging


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3Downloader:
    """
    Download plant bloom datasets from AWS S3.
    
    Handles authentication, bucket access, and progressive download
    of large datasets with proper error handling.
    """
    
    def __init__(
        self,
        bucket_name: str,
        region: str = 'us-west-2',
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None
    ):
        """
        Initialize S3 downloader.
        
        Args:
            bucket_name: Name of the S3 bucket
            region: AWS region
            aws_access_key_id: AWS access key (optional, can use env vars)
            aws_secret_access_key: AWS secret key (optional, can use env vars)
            aws_session_token: AWS session token (optional, for temporary credentials)
        """
        self.bucket_name = bucket_name
        self.region = region
        
        # Initialize S3 client
        try:
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region
            )
            self.s3_client = session.client('s3')
            
            # Test connection
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Successfully connected to S3 bucket: {bucket_name}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure your credentials.")
            self.s3_client = None
        except ClientError as e:
            logger.error(f"Error connecting to S3: {e}")
            self.s3_client = None
    
    def list_objects(self, prefix: str = '') -> List[Dict[str, Any]]:
        """
        List objects in the S3 bucket with given prefix.
        
        Args:
            prefix: S3 prefix to filter objects
            
        Returns:
            List of object metadata dictionaries
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return []
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            objects = response.get('Contents', [])
            logger.info(f"Found {len(objects)} objects with prefix '{prefix}'")
            return objects
            
        except ClientError as e:
            logger.error(f"Error listing objects: {e}")
            return []
    
    def download_file(
        self,
        s3_key: str,
        local_path: str,
        progress_callback: Optional[Callable] = None
    ) -> bool:
        """
        Download a single file from S3.
        
        Args:
            s3_key: S3 object key
            local_path: Local file path to save
            progress_callback: Optional callback for progress updates
            
        Returns:
            bool: Success status
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download with progress tracking
            file_size = self._get_object_size(s3_key)
            
            def progress_hook(bytes_transferred):
                if progress_callback and file_size > 0:
                    progress_callback(bytes_transferred / file_size)
            
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                local_path,
                Callback=progress_hook
            )
            
            logger.info(f"Successfully downloaded: {s3_key} -> {local_path}")
            return True
            
        except ClientError as e:
            logger.error(f"Error downloading {s3_key}: {e}")
            return False
    
    def download_dataset(
        self,
        prefix: str,
        local_dir: str,
        file_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Download an entire dataset from S3.
        
        Args:
            prefix: S3 prefix for the dataset
            local_dir: Local directory to save files
            file_patterns: Optional list of file patterns to filter
            
        Returns:
            Dict with download statistics
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return {'success': False, 'error': 'S3 client not initialized'}
        
        # List all objects
        objects = self.list_objects(prefix)
        
        if file_patterns:
            objects = [
                obj for obj in objects
                if any(pattern in obj['Key'] for pattern in file_patterns)
            ]
        
        # Download files with progress bar
        downloaded = 0
        failed = 0
        total_size = 0
        
        with tqdm(total=len(objects), desc="Downloading dataset") as pbar:
            for obj in objects:
                s3_key = obj['Key']
                local_path = os.path.join(local_dir, s3_key.replace(prefix, '').lstrip('/'))
                
                if self.download_file(s3_key, local_path):
                    downloaded += 1
                    total_size += obj.get('Size', 0)
                else:
                    failed += 1
                
                pbar.update(1)
        
        stats = {
            'success': True,
            'downloaded': downloaded,
            'failed': failed,
            'total_size_mb': total_size / (1024 * 1024),
            'local_dir': local_dir
        }
        
        logger.info(f"Download complete: {downloaded} files, {stats['total_size_mb']:.2f} MB")
        return stats
    
    def _get_object_size(self, s3_key: str) -> int:
        """Get the size of an S3 object."""
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return response.get('ContentLength', 0)
        except ClientError:
            return 0


class LocalDataLoader:
    """
    Load and manage local plant bloom datasets.
    
    Handles various local data formats and provides utilities
    for organizing and validating local datasets.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize local data loader.
        
        Args:
            data_dir: Root directory for local datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_archive(self, archive_path: str, extract_dir: Optional[str] = None) -> str:
        """
        Extract ZIP or TAR archives.
        
        Args:
            archive_path: Path to archive file
            extract_dir: Directory to extract to (default: same as archive)
            
        Returns:
            str: Path to extracted directory
        """
        archive_path = Path(archive_path)
        
        if extract_dir is None:
            extract_dir = archive_path.parent / archive_path.stem
        else:
            extract_dir = Path(extract_dir)
        
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        if archive_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif archive_path.suffix.lower() in ['.tar', '.tar.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
        
        logger.info(f"Extracted {archive_path} to {extract_dir}")
        return str(extract_dir)
    
    def scan_directory(self, directory: str, extensions: List[str] = None) -> Dict[str, List[str]]:
        """
        Scan directory for image files and create file inventory.
        
        Args:
            directory: Directory to scan
            extensions: List of file extensions to include
            
        Returns:
            Dict with file inventory by extension
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        directory = Path(directory)
        inventory = {ext: [] for ext in extensions}
        inventory['all'] = []
        
        for ext in extensions:
            files = list(directory.rglob(f'*{ext}')) + list(directory.rglob(f'*{ext.upper()}'))
            inventory[ext] = [str(f) for f in files]
            inventory['all'].extend(inventory[ext])
        
        logger.info(f"Found {len(inventory['all'])} image files in {directory}")
        return inventory
    
    def create_annotations_csv(
        self,
        image_dir: str,
        output_path: str,
        label_mapping: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create annotations CSV from directory structure.
        
        Assumes directory structure like:
        image_dir/
        ├── bud/
        ├── early_bloom/
        ├── full_bloom/
        ├── late_bloom/
        └── dormant/
        
        Args:
            image_dir: Root directory with class subdirectories
            output_path: Path to save annotations CSV
            label_mapping: Optional mapping from folder names to bloom stages
            
        Returns:
            str: Path to created annotations file
        """
        image_dir = Path(image_dir)
        annotations = []
        
        # Default label mapping
        if label_mapping is None:
            label_mapping = {
                'bud': 'bud',
                'early_bloom': 'early_bloom',
                'full_bloom': 'full_bloom',
                'late_bloom': 'late_bloom',
                'dormant': 'dormant'
            }
        
        # Scan subdirectories
        for class_dir in image_dir.iterdir():
            if class_dir.is_dir() and class_dir.name in label_mapping:
                bloom_stage = label_mapping[class_dir.name]
                
                # Find all images in class directory
                for img_file in class_dir.iterdir():
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        relative_path = img_file.relative_to(image_dir)
                        
                        annotations.append({
                            'image_path': str(relative_path),
                            'bloom_stage': bloom_stage,
                            'stage': 'train',  # Default to train, can be split later
                            'plant_id': f'plant_{len(annotations):04d}',
                            'timestamp': '2024-01-01'  # Placeholder
                        })
        
        # Create DataFrame and save
        df = pd.DataFrame(annotations)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Created annotations file with {len(annotations)} entries: {output_path}")
        return output_path
    
    def split_dataset(
        self,
        annotations_file: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ) -> str:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            annotations_file: Path to annotations CSV
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducibility
            
        Returns:
            str: Path to updated annotations file
        """
        df = pd.read_csv(annotations_file)
        
        # Stratified split by bloom_stage
        from sklearn.model_selection import train_test_split
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            stratify=df['bloom_stage'],
            random_state=random_seed
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            stratify=temp_df['bloom_stage'],
            random_state=random_seed
        )
        
        # Update stage column
        train_df['stage'] = 'train'
        val_df['stage'] = 'val'
        test_df['stage'] = 'test'
        
        # Combine and save
        final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        final_df.to_csv(annotations_file, index=False)
        
        logger.info(f"Dataset split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return annotations_file


class URLDownloader:
    """
    Download datasets from URLs (HTTP/HTTPS).
    
    Useful for downloading public datasets from research repositories.
    """
    
    def __init__(self):
        pass
    
    def download_file(
        self,
        url: str,
        local_path: str,
        chunk_size: int = 8192
    ) -> bool:
        """
        Download a file from URL with progress tracking.
        
        Args:
            url: URL to download from
            local_path: Local path to save file
            chunk_size: Size of chunks for streaming download
            
        Returns:
            bool: Success status
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with open(local_path, 'wb') as f, tqdm(
                desc=os.path.basename(local_path),
                total=total_size,
                unit='B',
                unit_scale=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Downloaded: {url} -> {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return False