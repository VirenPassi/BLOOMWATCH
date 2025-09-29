#!/usr/bin/env python3
"""
MODIS-enabled dataset classes for plant bloom detection.

This module extends the PlantBloomDataset to include MODIS MOD13Q1 data
(NDVI and EVI arrays) for enhanced plant bloom detection and analysis.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict, Any, List, Union
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)


class MODISPlantBloomDataset(Dataset):
    """
    Enhanced PlantBloomDataset that integrates MODIS MOD13Q1 data.
    
    This dataset combines traditional plant images with MODIS satellite data
    (NDVI and EVI arrays) to provide multi-modal input for bloom detection.
    
    Args:
        data_dir (str): Path to the plant image dataset directory
        processed_dir (str): Path to processed MODIS .npy files directory
        annotations_file (str): Path to CSV file containing image paths and labels
        transform (Optional[Callable]): Transform to apply to images
        target_transform (Optional[Callable]): Transform to apply to labels
        modis_transform (Optional[Callable]): Transform to apply to MODIS data
        stage (str): Dataset stage - 'train', 'val', or 'test'
        target_size (Optional[Tuple[int, int]]): Target size for resizing MODIS arrays
        normalize_modis (bool): Whether to normalize MODIS values to [0,1]
        include_metadata (bool): Whether to include MODIS metadata
        filter_by_date (Optional[str]): Filter MODIS data by date (YYYY-MM-DD format)
        filter_by_region (Optional[Dict]): Filter by bounding box {'min_lon': float, 'min_lat': float, 'max_lon': float, 'max_lat': float}
    """
    
    # Bloom stage mappings (inherited from original dataset)
    BLOOM_STAGES = {
        'bud': 0,
        'early_bloom': 1, 
        'full_bloom': 2,
        'late_bloom': 3,
        'dormant': 4
    }
    
    STAGE_NAMES = {v: k for k, v in BLOOM_STAGES.items()}
    
    def __init__(
        self,
        data_dir: str,
        processed_dir: str = "data/processed/MODIS",
        annotations_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        modis_transform: Optional[Callable] = None,
        stage: str = 'train',
        target_size: Optional[Tuple[int, int]] = None,
        normalize_modis: bool = True,
        include_metadata: bool = True,
        filter_by_date: Optional[str] = None,
        filter_by_region: Optional[Dict[str, float]] = None
    ):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.modis_transform = modis_transform
        self.stage = stage
        self.target_size = target_size
        self.normalize_modis = normalize_modis
        self.include_metadata = include_metadata
        self.filter_by_date = filter_by_date
        self.filter_by_region = filter_by_region
        
        # Load annotations if provided
        if annotations_file and os.path.exists(annotations_file):
            self.annotations = self._load_annotations(annotations_file)
            if stage in ['train', 'val', 'test']:
                self.annotations = self.annotations[
                    self.annotations['stage'] == stage
                ].reset_index(drop=True)
        else:
            self.annotations = None
            logger.warning("No annotations file provided. Dataset will work with MODIS data only.")
        
        # Load MODIS data
        self.modis_files = self._load_modis_files()
        
        # Create mapping between plant images and MODIS data
        self.sample_mapping = self._create_sample_mapping()
        
        logger.info(f"Initialized MODISPlantBloomDataset with {len(self.sample_mapping)} samples")
        logger.info(f"Found {len(self.modis_files)} MODIS files")
    
    def _load_annotations(self, annotations_file: str) -> pd.DataFrame:
        """Load and validate annotations from CSV file."""
        df = pd.read_csv(annotations_file)
        
        # Validate required columns
        required_cols = ['image_path', 'bloom_stage']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate bloom stages
        invalid_stages = set(df['bloom_stage']) - set(self.BLOOM_STAGES.keys())
        if invalid_stages:
            raise ValueError(f"Invalid bloom stages: {invalid_stages}")
        
        return df
    
    def list_processed_files(self) -> List[Path]:
        """
        List all available processed MODIS .npy files.
        
        Returns:
            List of Path objects for NDVI .npy files
        """
        if not self.processed_dir.exists():
            logger.warning(f"Processed directory {self.processed_dir} does not exist")
            return []
        
        ndvi_files = list(self.processed_dir.glob("*_ndvi.npy"))
        logger.info(f"Found {len(ndvi_files)} NDVI files")
        return ndvi_files
    
    def get_metadata(self, filename: str) -> Dict[str, Any]:
        """
        Read metadata from corresponding .json file.
        
        Args:
            filename: Base filename (without _ndvi.npy extension)
            
        Returns:
            Dictionary containing metadata
        """
        metadata_file = self.processed_dir / f"{filename}_metadata.json"
        
        if not metadata_file.exists():
            logger.warning(f"Metadata file not found: {metadata_file}")
            return {}
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            logger.error(f"Error reading metadata file {metadata_file}: {e}")
            return {}
    
    def _load_modis_files(self) -> List[Dict[str, Any]]:
        """Load and validate MODIS files."""
        modis_files = []
        ndvi_files = self.list_processed_files()
        
        for ndvi_file in ndvi_files:
            # Get corresponding EVI file
            base_name = ndvi_file.stem.replace('_ndvi', '')
            evi_file = self.processed_dir / f"{base_name}_evi.npy"
            
            if not evi_file.exists():
                logger.warning(f"EVI file not found for {base_name}")
                continue
            
            # Load metadata
            metadata = self.get_metadata(base_name)
            
            # Apply filters if specified
            if self._should_include_file(metadata):
                modis_files.append({
                    'base_name': base_name,
                    'ndvi_file': ndvi_file,
                    'evi_file': evi_file,
                    'metadata': metadata
                })
        
        return modis_files
    
    def _should_include_file(self, metadata: Dict[str, Any]) -> bool:
        """Check if a file should be included based on filters."""
        if not metadata:
            return True
        
        # Filter by date if specified
        if self.filter_by_date:
            file_date = metadata.get('filename', '')
            if self.filter_by_date not in file_date:
                return False
        
        # Filter by region if specified
        if self.filter_by_region:
            # This would require additional metadata about bounding box
            # For now, we'll include all files
            pass
        
        return True
    
    def _create_sample_mapping(self) -> List[Dict[str, Any]]:
        """Create mapping between samples and MODIS data."""
        samples = []
        
        if self.annotations is not None:
            # Map plant images to MODIS data
            for idx, row in self.annotations.iterrows():
                # For now, assign MODIS data randomly
                # In a real implementation, you'd match by location/time
                modis_idx = idx % len(self.modis_files) if self.modis_files else None
                
                samples.append({
                    'type': 'plant_image',
                    'image_path': self.data_dir / row['image_path'],
                    'bloom_stage': row['bloom_stage'],
                    'plant_id': row.get('plant_id', 'unknown'),
                    'timestamp': row.get('timestamp', 'unknown'),
                    'modis_idx': modis_idx
                })
        else:
            # MODIS-only mode
            for idx, modis_data in enumerate(self.modis_files):
                samples.append({
                    'type': 'modis_only',
                    'modis_idx': idx,
                    'bloom_stage': None  # No labels available
                })
        
        # If we have both annotations and MODIS files, also create MODIS-only samples
        if self.annotations is not None and self.modis_files:
            for idx, modis_data in enumerate(self.modis_files):
                samples.append({
                    'type': 'modis_only',
                    'modis_idx': idx,
                    'bloom_stage': None
                })
        
        return samples
    
    def _load_modis_arrays(self, modis_idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Load NDVI and EVI arrays from .npy files.
        
        Args:
            modis_idx: Index of MODIS data in self.modis_files
            
        Returns:
            Tuple of (ndvi_tensor, evi_tensor, metadata)
        """
        if modis_idx >= len(self.modis_files):
            raise IndexError(f"MODIS index {modis_idx} out of range")
        
        modis_data = self.modis_files[modis_idx]
        
        try:
            # Load NDVI array
            ndvi_array = np.load(modis_data['ndvi_file'])
            
            # Load EVI array
            evi_array = np.load(modis_data['evi_file'])
            
            # Validate arrays
            if ndvi_array.shape != evi_array.shape:
                raise ValueError(f"NDVI and EVI shapes don't match: {ndvi_array.shape} vs {evi_array.shape}")
            
            # Normalize to [0, 1] if requested
            if self.normalize_modis:
                ndvi_array = self._normalize_modis_array(ndvi_array)
                evi_array = self._normalize_modis_array(evi_array)
            
            # Convert to tensors
            ndvi_tensor = torch.FloatTensor(ndvi_array)
            evi_tensor = torch.FloatTensor(evi_array)
            
            # Add channel dimension for CNN compatibility
            if len(ndvi_tensor.shape) == 2:
                ndvi_tensor = ndvi_tensor.unsqueeze(0)  # Add channel dimension
                evi_tensor = evi_tensor.unsqueeze(0)
            
            # Resize if target size specified
            if self.target_size:
                ndvi_tensor = F.interpolate(
                    ndvi_tensor.unsqueeze(0), 
                    size=self.target_size, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                evi_tensor = F.interpolate(
                    evi_tensor.unsqueeze(0), 
                    size=self.target_size, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # Apply MODIS transform if provided
            if self.modis_transform:
                ndvi_tensor = self.modis_transform(ndvi_tensor)
                evi_tensor = self.modis_transform(evi_tensor)
            
            metadata = modis_data['metadata'].copy()
            metadata.update({
                'ndvi_shape': ndvi_tensor.shape,
                'evi_shape': evi_tensor.shape,
                'ndvi_min': float(torch.min(ndvi_tensor)),
                'ndvi_max': float(torch.max(ndvi_tensor)),
                'evi_min': float(torch.min(evi_tensor)),
                'evi_max': float(torch.max(evi_tensor))
            })
            
            return ndvi_tensor, evi_tensor, metadata
            
        except Exception as e:
            logger.error(f"Error loading MODIS arrays for {modis_data['base_name']}: {e}")
            # Return dummy data to prevent crashes
            dummy_shape = (1, 2400, 2400) if not self.target_size else (1, *self.target_size)
            dummy_ndvi = torch.zeros(dummy_shape)
            dummy_evi = torch.zeros(dummy_shape)
            return dummy_ndvi, dummy_evi, {}
    
    def _normalize_modis_array(self, array: np.ndarray) -> np.ndarray:
        """
        Normalize MODIS array to [0, 1] range.
        
        Args:
            array: Input MODIS array
            
        Returns:
            Normalized array
        """
        # Handle NaN values
        valid_mask = ~np.isnan(array)
        
        if not np.any(valid_mask):
            # All values are NaN
            return np.zeros_like(array)
        
        valid_data = array[valid_mask]
        
        # Normalize to [0, 1]
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        
        if max_val == min_val:
            # All valid values are the same
            normalized = np.zeros_like(array)
            normalized[valid_mask] = 0.5
        else:
            normalized = np.zeros_like(array)
            normalized[valid_mask] = (valid_data - min_val) / (max_val - min_val)
        
        return normalized
    
    def _load_image(self, img_path: Path) -> Image.Image:
        """Load an image from file path."""
        if img_path.exists():
            return Image.open(img_path).convert('RGB')
        else:
            # Create dummy image for development
            logger.warning(f"Image not found: {img_path}. Creating dummy image.")
            dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            return Image.fromarray(dummy_img)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sample_mapping)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]], 
                                           Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            If plant image mode: (image_tensor, ndvi_tensor, evi_tensor, label, metadata)
            If MODIS-only mode: (ndvi_tensor, evi_tensor, metadata)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.sample_mapping[idx]
        
        # Load MODIS data
        if sample['modis_idx'] is not None:
            ndvi_tensor, evi_tensor, modis_metadata = self._load_modis_arrays(sample['modis_idx'])
        else:
            # No MODIS data available
            dummy_shape = (1, 2400, 2400) if not self.target_size else (1, *self.target_size)
            ndvi_tensor = torch.zeros(dummy_shape)
            evi_tensor = torch.zeros(dummy_shape)
            modis_metadata = {}
        
        if sample['type'] == 'plant_image':
            # Plant image mode
            image = self._load_image(sample['image_path'])
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Get label
            label = self.BLOOM_STAGES[sample['bloom_stage']]
            if self.target_transform:
                label = self.target_transform(label)
            
            # Prepare metadata
            metadata = {
                'image_path': str(sample['image_path']),
                'bloom_stage_name': sample['bloom_stage'],
                'plant_id': sample['plant_id'],
                'timestamp': sample['timestamp'],
                'modis_metadata': modis_metadata
            }
            
            return image, ndvi_tensor, evi_tensor, label, metadata
            
        else:
            # MODIS-only mode
            metadata = {
                'modis_metadata': modis_metadata,
                'sample_type': 'modis_only'
            }
            
            return ndvi_tensor, evi_tensor, metadata
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced datasets."""
        if self.annotations is None:
            # Return equal weights if no annotations
            return torch.ones(len(self.BLOOM_STAGES))
        
        stage_counts = self.annotations['bloom_stage'].value_counts()
        total_samples = len(self.annotations)
        
        weights = []
        for stage_name in self.BLOOM_STAGES.keys():
            count = stage_counts.get(stage_name, 1)
            weight = total_samples / (len(self.BLOOM_STAGES) * count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'total_samples': len(self.sample_mapping),
            'modis_files': len(self.modis_files),
            'stage': self.stage,
            'target_size': self.target_size,
            'normalize_modis': self.normalize_modis
        }
        
        if self.annotations is not None:
            stats.update({
                'bloom_stage_distribution': self.annotations['bloom_stage'].value_counts().to_dict(),
                'unique_plants': self.annotations.get('plant_id', pd.Series()).nunique()
            })
        
        return stats
    
    def get_modis_statistics(self) -> Dict[str, Any]:
        """Get MODIS-specific statistics."""
        if not self.modis_files:
            return {'error': 'No MODIS files available'}
        
        ndvi_shapes = []
        evi_shapes = []
        ndvi_mins = []
        ndvi_maxs = []
        evi_mins = []
        evi_maxs = []
        
        for modis_data in self.modis_files[:10]:  # Sample first 10 files
            try:
                ndvi_array = np.load(modis_data['ndvi_file'])
                evi_array = np.load(modis_data['evi_file'])
                
                ndvi_shapes.append(ndvi_array.shape)
                evi_shapes.append(evi_array.shape)
                
                valid_ndvi = ndvi_array[~np.isnan(ndvi_array)]
                valid_evi = evi_array[~np.isnan(evi_array)]
                
                if len(valid_ndvi) > 0:
                    ndvi_mins.append(np.min(valid_ndvi))
                    ndvi_maxs.append(np.max(valid_ndvi))
                
                if len(valid_evi) > 0:
                    evi_mins.append(np.min(valid_evi))
                    evi_maxs.append(np.max(valid_evi))
                    
            except Exception as e:
                logger.warning(f"Error processing {modis_data['base_name']}: {e}")
        
        return {
            'total_modis_files': len(self.modis_files),
            'sample_ndvi_shapes': ndvi_shapes[:5],
            'sample_evi_shapes': evi_shapes[:5],
            'ndvi_range': [min(ndvi_mins) if ndvi_mins else 0, max(ndvi_maxs) if ndvi_maxs else 0],
            'evi_range': [min(evi_mins) if evi_mins else 0, max(evi_maxs) if evi_maxs else 0]
        }


class MODISOnlyDataset(MODISPlantBloomDataset):
    """
    Simplified dataset that only contains MODIS data (no plant images).
    
    This is useful for MODIS-only analysis or when plant images are not available.
    """
    
    def __init__(self, processed_dir: str = "data/processed/MODIS", **kwargs):
        # Initialize without plant image data
        super().__init__(
            data_dir="",  # No plant images
            processed_dir=processed_dir,
            annotations_file=None,  # No annotations
            **kwargs
        )
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Get MODIS data only."""
        sample = self.sample_mapping[idx]
        
        if sample['modis_idx'] is not None:
            ndvi_tensor, evi_tensor, modis_metadata = self._load_modis_arrays(sample['modis_idx'])
        else:
            dummy_shape = (1, 2400, 2400) if not self.target_size else (1, *self.target_size)
            ndvi_tensor = torch.zeros(dummy_shape)
            evi_tensor = torch.zeros(dummy_shape)
            modis_metadata = {}
        
        metadata = {
            'modis_metadata': modis_metadata,
            'sample_type': 'modis_only'
        }
        
        return ndvi_tensor, evi_tensor, metadata
