#!/usr/bin/env python3
"""
Example usage of MODIS-enabled PlantBloomDataset.

This script demonstrates how to use the MODISPlantBloomDataset and MODISOnlyDataset
classes for training CNN models with MODIS satellite data.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add the data directory to the path
data_dir = Path(__file__).parent
sys.path.insert(0, str(data_dir))

from modis_dataset import MODISPlantBloomDataset, MODISOnlyDataset


def example_modis_only_dataset():
    """Example: Using MODIS-only dataset."""
    print("=== MODIS-Only Dataset Example ===")
    
    # Create dataset
    dataset = MODISOnlyDataset(
        processed_dir="data/processed/MODIS",
        target_size=(512, 512),  # Resize to 512x512 for CNN input
        normalize_modis=True,    # Normalize values to [0, 1]
        include_metadata=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Available MODIS files: {len(dataset.list_processed_files())}")
    
    # Get a sample
    if len(dataset) > 0:
        ndvi, evi, metadata = dataset[0]
        
        print(f"NDVI shape: {ndvi.shape}")
        print(f"EVI shape: {evi.shape}")
        print(f"NDVI range: {torch.min(ndvi):.3f} to {torch.max(ndvi):.3f}")
        print(f"EVI range: {torch.min(evi):.3f} to {torch.max(evi):.3f}")
        print(f"Metadata keys: {list(metadata.keys())}")
    
    # Get statistics
    stats = dataset.get_statistics()
    print(f"Dataset statistics: {stats}")
    
    modis_stats = dataset.get_modis_statistics()
    print(f"MODIS statistics: {modis_stats}")


def example_modis_plant_dataset():
    """Example: Using MODIS + Plant image dataset."""
    print("\n=== MODIS + Plant Image Dataset Example ===")
    
    # Simple transform to convert PIL to tensor
    def simple_transform(image):
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
        return torch.FloatTensor(img_array)
    
    # Create dataset
    dataset = MODISPlantBloomDataset(
        data_dir="data/images",  # Plant images directory
        processed_dir="data/processed/MODIS",
        annotations_file="data/annotations.csv",  # Optional
        transform=simple_transform,
        target_size=(256, 256),  # Resize MODIS data
        normalize_modis=True,
        include_metadata=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    if len(dataset) > 0:
        image, ndvi, evi, label, metadata = dataset[0]
        
        print(f"Image shape: {image.shape}")
        print(f"NDVI shape: {ndvi.shape}")
        print(f"EVI shape: {evi.shape}")
        print(f"Label: {label}")
        print(f"Bloom stage: {metadata.get('bloom_stage_name', 'unknown')}")
        print(f"Plant ID: {metadata.get('plant_id', 'unknown')}")
    
    # Get class weights for imbalanced datasets
    class_weights = dataset.get_class_weights()
    print(f"Class weights: {class_weights}")


def example_data_loader():
    """Example: Using dataset with PyTorch DataLoader."""
    print("\n=== DataLoader Example ===")
    
    # Create dataset
    dataset = MODISOnlyDataset(
        processed_dir="data/processed/MODIS",
        target_size=(224, 224),
        normalize_modis=True
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    print(f"DataLoader created with batch size 4")
    print(f"Number of batches: {len(dataloader)}")
    
    # Iterate through batches
    for batch_idx, (ndvi_batch, evi_batch, metadata_batch) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  NDVI batch shape: {ndvi_batch.shape}")
        print(f"  EVI batch shape: {evi_batch.shape}")
        print(f"  Batch size: {ndvi_batch.size(0)}")
        
        if batch_idx >= 2:  # Show only first 3 batches
            break


def example_cnn_training():
    """Example: CNN model architecture for MODIS data."""
    print("\n=== CNN Model Example ===")
    
    import torch.nn as nn
    
    class MODISCNN(nn.Module):
        """Simple CNN for MODIS data classification."""
        
        def __init__(self, num_classes=5, input_size=(224, 224)):
            super().__init__()
            
            # Input: 2 channels (NDVI + EVI)
            self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.5)
            
            # Calculate flattened size
            with torch.no_grad():
                dummy_input = torch.zeros(1, 2, *input_size)
                dummy_output = self._forward_features(dummy_input)
                self.flattened_size = dummy_output.numel()
            
            self.fc1 = nn.Linear(self.flattened_size, 512)
            self.fc2 = nn.Linear(512, num_classes)
            
        def _forward_features(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            return x
        
        def forward(self, ndvi, evi):
            # Concatenate NDVI and EVI along channel dimension
            x = torch.cat([ndvi, evi], dim=1)
            
            # Extract features
            x = self._forward_features(x)
            
            # Flatten and classify
            x = x.view(x.size(0), -1)
            x = self.dropout(torch.relu(self.fc1(x)))
            x = self.fc2(x)
            
            return x
    
    # Create model
    model = MODISCNN(num_classes=5, input_size=(224, 224))
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example forward pass
    batch_size = 2
    ndvi_batch = torch.randn(batch_size, 1, 224, 224)
    evi_batch = torch.randn(batch_size, 1, 224, 224)
    
    with torch.no_grad():
        output = model(ndvi_batch, evi_batch)
        print(f"Model output shape: {output.shape}")
        print(f"Predictions: {torch.softmax(output, dim=1)}")


def example_filtering():
    """Example: Filtering MODIS data by date and region."""
    print("\n=== Filtering Example ===")
    
    # Filter by date
    dataset_by_date = MODISOnlyDataset(
        processed_dir="data/processed/MODIS",
        filter_by_date="2022-01",  # Only January 2022 data
        normalize_modis=True
    )
    
    print(f"Dataset filtered by date: {len(dataset_by_date)} samples")
    
    # Filter by region (if metadata contains bounding box info)
    dataset_by_region = MODISOnlyDataset(
        processed_dir="data/processed/MODIS",
        filter_by_region={
            'min_lon': 70.0,
            'min_lat': 8.0,
            'max_lon': 90.0,
            'max_lat': 37.0
        },
        normalize_modis=True
    )
    
    print(f"Dataset filtered by region: {len(dataset_by_region)} samples")


def example_metadata_usage():
    """Example: Working with MODIS metadata."""
    print("\n=== Metadata Usage Example ===")
    
    dataset = MODISOnlyDataset(
        processed_dir="data/processed/MODIS",
        include_metadata=True
    )
    
    if len(dataset) > 0:
        ndvi, evi, metadata = dataset[0]
        
        print("Sample metadata:")
        for key, value in metadata.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key}: {value}")
        
        # Access specific metadata
        modis_meta = metadata.get('modis_metadata', {})
        if modis_meta:
            print(f"\nMODIS-specific metadata:")
            print(f"  Original filename: {modis_meta.get('filename', 'unknown')}")
            print(f"  NDVI range: {modis_meta.get('ndvi_min', 0):.3f} to {modis_meta.get('ndvi_max', 0):.3f}")
            print(f"  EVI range: {modis_meta.get('evi_min', 0):.3f} to {modis_meta.get('evi_max', 0):.3f}")


def main():
    """Run all examples."""
    print("MODIS Dataset Usage Examples")
    print("=" * 50)
    
    try:
        # Check if processed MODIS data exists
        processed_dir = Path("data/processed/MODIS")
        if not processed_dir.exists() or not list(processed_dir.glob("*.npy")):
            print("Warning: No processed MODIS data found.")
            print("Please run: python data/preprocess_modis.py --all")
            print("Skipping examples that require MODIS data...")
            return
        
        example_modis_only_dataset()
        example_modis_plant_dataset()
        example_data_loader()
        example_cnn_training()
        example_filtering()
        example_metadata_usage()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully! ✅")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
