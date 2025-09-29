"""
Dataset classes for plant bloom detection.

This module contains the main dataset class for loading and processing
plant bloom images with their corresponding bloom stage labels.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from typing import Optional, Callable, Tuple, Dict, Any
import numpy as np


class PlantBloomDataset(Dataset):
    """
    Dataset class for plant bloom stage detection.
    
    This dataset handles loading plant images and their corresponding
    bloom stage labels. Supports various bloom stages like bud, early_bloom,
    full_bloom, late_bloom, and dormant.
    
    Args:
        data_dir (str): Path to the dataset directory
        annotations_file (str): Path to CSV file containing image paths and labels
        transform (Optional[Callable]): Transform to apply to images
        target_transform (Optional[Callable]): Transform to apply to labels
        stage (str): Dataset stage - 'train', 'val', or 'test'
    """
    
    # Bloom stage mappings
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
        annotations_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        stage: str = 'train'
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.stage = stage
        
        # Load annotations
        self.annotations = self._load_annotations(annotations_file)
        
        # Filter by stage if specified
        if stage in ['train', 'val', 'test']:
            self.annotations = self.annotations[
                self.annotations['stage'] == stage
            ].reset_index(drop=True)
    
    def _load_annotations(self, annotations_file: str) -> pd.DataFrame:
        """
        Load and validate annotations from CSV file.
        
        Expected CSV format:
        image_path,bloom_stage,stage,plant_id,timestamp
        """
        if not os.path.exists(annotations_file):
            # Create dummy annotations for development
            print(f"Warning: {annotations_file} not found. Creating dummy data.")
            return self._create_dummy_annotations()
        
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
    
    def _create_dummy_annotations(self) -> pd.DataFrame:
        """Create dummy annotations for development and testing."""
        import random
        
        # Create dummy data with different stages
        data = []
        stages = ['train', 'val', 'test']
        bloom_stages = list(self.BLOOM_STAGES.keys())
        
        for i in range(100):  # 100 dummy samples
            stage = random.choice(stages)
            bloom_stage = random.choice(bloom_stages)
            data.append({
                'image_path': f'dummy_image_{i:03d}.jpg',
                'bloom_stage': bloom_stage,
                'stage': stage,
                'plant_id': f'plant_{i // 10:03d}',
                'timestamp': f'2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}'
            })
        
        return pd.DataFrame(data)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            Tuple containing:
                - image (torch.Tensor): Processed image tensor
                - label (int): Bloom stage label
                - metadata (Dict): Additional metadata
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get annotation
        ann = self.annotations.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.data_dir, ann['image_path'])
        image = self._load_image(img_path)
        
        # Get label
        label = self.BLOOM_STAGES[ann['bloom_stage']]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        # Prepare metadata
        metadata = {
            'image_path': ann['image_path'],
            'bloom_stage_name': ann['bloom_stage'],
            'plant_id': ann.get('plant_id', 'unknown'),
            'timestamp': ann.get('timestamp', 'unknown')
        }
        
        return image, label, metadata
    
    def _load_image(self, img_path: str) -> Image.Image:
        """
        Load an image from file path.
        
        If image doesn't exist (dummy data), create a random image.
        """
        if os.path.exists(img_path):
            return Image.open(img_path).convert('RGB')
        else:
            # Create dummy image for development
            import numpy as np
            dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            return Image.fromarray(dummy_img)
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced datasets.
        
        Returns:
            torch.Tensor: Class weights for loss function
        """
        stage_counts = self.annotations['bloom_stage'].value_counts()
        total_samples = len(self.annotations)
        
        weights = []
        for stage_name in self.BLOOM_STAGES.keys():
            count = stage_counts.get(stage_name, 1)  # Avoid division by zero
            weight = total_samples / (len(self.BLOOM_STAGES) * count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'total_samples': len(self.annotations),
            'bloom_stage_distribution': self.annotations['bloom_stage'].value_counts().to_dict(),
            'unique_plants': self.annotations.get('plant_id', pd.Series()).nunique(),
            'stage': self.stage
        }
        return stats


class TimeSeriesBloomDataset(PlantBloomDataset):
    """
    Extended dataset for time-series analysis of plant blooming.
    
    This dataset groups images by plant_id and timestamp to enable
    temporal analysis of blooming patterns.
    """
    
    def __init__(self, *args, sequence_length: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_length = sequence_length
        self.sequences = self._create_sequences()
    
    def _create_sequences(self):
        """Create sequences of images for the same plant over time."""
        sequences = []
        
        # Group by plant_id and sort by timestamp
        if 'plant_id' in self.annotations.columns and 'timestamp' in self.annotations.columns:
            grouped = self.annotations.groupby('plant_id')
            
            for plant_id, group in grouped:
                group_sorted = group.sort_values('timestamp').reset_index(drop=True)
                
                # Create overlapping sequences
                for i in range(len(group_sorted) - self.sequence_length + 1):
                    sequence_indices = list(range(i, i + self.sequence_length))
                    sequences.append({
                        'plant_id': plant_id,
                        'indices': sequence_indices,
                        'start_idx': group_sorted.index[i],
                        'end_idx': group_sorted.index[i + self.sequence_length - 1]
                    })
        
        return sequences
    
    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences) if self.sequences else super().__len__()
    
    def __getitem__(self, idx: int):
        """Get a sequence of images and labels."""
        if not self.sequences:
            return super().__getitem__(idx)
        
        sequence_info = self.sequences[idx]
        
        images = []
        labels = []
        metadata_list = []
        
        for seq_idx in sequence_info['indices']:
            img, label, metadata = super().__getitem__(seq_idx)
            images.append(img)
            labels.append(label)
            metadata_list.append(metadata)
        
        # Stack images and labels
        images = torch.stack(images)
        labels = torch.tensor(labels)
        
        sequence_metadata = {
            'plant_id': sequence_info['plant_id'],
            'sequence_length': self.sequence_length,
            'individual_metadata': metadata_list
        }
        
        return images, labels, sequence_metadata