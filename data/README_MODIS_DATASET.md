# MODIS-Enabled PlantBloomDataset

This document describes the MODIS-enabled dataset classes that integrate NASA MODIS MOD13Q1 satellite data with plant bloom detection for enhanced CNN training.

## Overview

The MODIS dataset classes extend the original `PlantBloomDataset` to include:
- **NDVI (Normalized Difference Vegetation Index)** arrays
- **EVI (Enhanced Vegetation Index)** arrays  
- **Metadata** from processed MODIS files
- **Flexible filtering** by date and region
- **PyTorch compatibility** for CNN training

## Classes

### 1. MODISPlantBloomDataset

Combines plant images with MODIS satellite data for multi-modal analysis.

```python
from data.modis_dataset import MODISPlantBloomDataset

dataset = MODISPlantBloomDataset(
    data_dir="data/images",                    # Plant images directory
    processed_dir="data/processed/MODIS",      # MODIS .npy files
    annotations_file="data/annotations.csv",   # Optional annotations
    transform=transform,                       # Image transforms
    target_size=(224, 224),                   # Resize MODIS arrays
    normalize_modis=True,                     # Normalize to [0,1]
    include_metadata=True                     # Include MODIS metadata
)
```

**Returns:** `(image_tensor, ndvi_tensor, evi_tensor, label, metadata)`

### 2. MODISOnlyDataset

Simplified dataset containing only MODIS data (no plant images).

```python
from data.modis_dataset import MODISOnlyDataset

dataset = MODISOnlyDataset(
    processed_dir="data/processed/MODIS",
    target_size=(512, 512),
    normalize_modis=True
)
```

**Returns:** `(ndvi_tensor, evi_tensor, metadata)`

## Key Features

### 1. Data Loading
- **Automatic .npy file discovery** from processed directory
- **Metadata loading** from corresponding .json files
- **Error handling** for missing or corrupted files
- **File deduplication** to avoid processing the same data twice

### 2. Preprocessing
- **Normalization**: NDVI/EVI values normalized to [0, 1] range
- **Resizing**: Arrays resized to target dimensions for CNN input
- **NaN handling**: Invalid pixels properly handled
- **Tensor conversion**: Automatic conversion to PyTorch tensors

### 3. Filtering
- **Date filtering**: Filter MODIS data by specific dates
- **Region filtering**: Filter by bounding box coordinates
- **Stage filtering**: Filter plant images by train/val/test stage

### 4. PyTorch Integration
- **DataLoader compatibility**: Works seamlessly with PyTorch DataLoader
- **Batch processing**: Supports batch loading and processing
- **Memory efficient**: Processes files on-demand

## Usage Examples

### Basic Usage

```python
# Create dataset
dataset = MODISOnlyDataset(
    processed_dir="data/processed/MODIS",
    target_size=(224, 224),
    normalize_modis=True
)

# Get a sample
ndvi, evi, metadata = dataset[0]
print(f"NDVI shape: {ndvi.shape}")  # torch.Size([1, 224, 224])
print(f"EVI shape: {evi.shape}")    # torch.Size([1, 224, 224])
```

### With DataLoader

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

for ndvi_batch, evi_batch, metadata_batch in dataloader:
    print(f"Batch shapes: {ndvi_batch.shape}, {evi_batch.shape}")
    # ndvi_batch: torch.Size([8, 1, 224, 224])
    # evi_batch: torch.Size([8, 1, 224, 224])
```

### CNN Model Integration

```python
import torch.nn as nn

class MODISCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)  # 2 channels: NDVI + EVI
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 56 * 56, num_classes)  # Adjust based on input size
    
    def forward(self, ndvi, evi):
        # Concatenate NDVI and EVI
        x = torch.cat([ndvi, evi], dim=1)  # [batch, 2, H, W]
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Usage
model = MODISCNN()
ndvi, evi, _ = dataset[0]
output = model(ndvi.unsqueeze(0), evi.unsqueeze(0))  # Add batch dimension
```

### Filtering Examples

```python
# Filter by date
dataset_jan = MODISOnlyDataset(
    processed_dir="data/processed/MODIS",
    filter_by_date="2022-01"  # Only January 2022 data
)

# Filter by region
dataset_region = MODISOnlyDataset(
    processed_dir="data/processed/MODIS",
    filter_by_region={
        'min_lon': 70.0,
        'min_lat': 8.0,
        'max_lon': 90.0,
        'max_lat': 37.0
    }
)
```

## Parameters

### MODISPlantBloomDataset

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | str | Required | Path to plant images directory |
| `processed_dir` | str | "data/processed/MODIS" | Path to MODIS .npy files |
| `annotations_file` | str | None | Path to CSV annotations file |
| `transform` | Callable | None | Transform for plant images |
| `target_transform` | Callable | None | Transform for labels |
| `modis_transform` | Callable | None | Transform for MODIS data |
| `stage` | str | "train" | Dataset stage (train/val/test) |
| `target_size` | Tuple[int, int] | None | Target size for MODIS arrays |
| `normalize_modis` | bool | True | Normalize MODIS values to [0,1] |
| `include_metadata` | bool | True | Include MODIS metadata |
| `filter_by_date` | str | None | Filter by date (YYYY-MM-DD) |
| `filter_by_region` | Dict | None | Filter by bounding box |

### MODISOnlyDataset

Same as above, but without `data_dir` and `annotations_file` parameters.

## Data Format

### Input Files
- **NDVI files**: `{base_name}_ndvi.npy`
- **EVI files**: `{base_name}_evi.npy`
- **Metadata files**: `{base_name}_metadata.json`

### Output Tensors
- **NDVI tensor**: `torch.FloatTensor` with shape `[1, H, W]`
- **EVI tensor**: `torch.FloatTensor` with shape `[1, H, W]`
- **Metadata**: Dictionary with processing information

### Metadata Structure

```json
{
  "filename": "MOD13Q1.A2022013.h31v08.061.2022025132825.hdf",
  "ndvi_shape": [2400, 2400],
  "evi_shape": [2400, 2400],
  "ndvi_min": -0.2,
  "ndvi_max": 0.9,
  "evi_min": -0.2,
  "evi_max": 0.8,
  "ndvi_valid_pixels": 5760000,
  "evi_valid_pixels": 5760000
}
```

## Helper Functions

### Dataset Methods

```python
# List available MODIS files
files = dataset.list_processed_files()

# Get metadata for specific file
metadata = dataset.get_metadata("MOD13Q1.A2022013.h31v08.061.2022025132825")

# Get dataset statistics
stats = dataset.get_statistics()

# Get MODIS-specific statistics
modis_stats = dataset.get_modis_statistics()

# Get class weights for imbalanced datasets
weights = dataset.get_class_weights()
```

## Error Handling

The dataset includes robust error handling:

- **Missing files**: Skips files with missing NDVI/EVI pairs
- **Corrupted data**: Returns dummy data to prevent crashes
- **Invalid metadata**: Provides empty metadata dictionary
- **Shape mismatches**: Logs warnings and continues processing

## Performance Tips

1. **Memory Management**: Process files on-demand to manage memory usage
2. **Batch Size**: Use appropriate batch sizes based on available RAM
3. **Target Size**: Resize MODIS arrays to reduce memory footprint
4. **Caching**: Consider caching frequently accessed data
5. **Workers**: Use `num_workers=0` on Windows to avoid multiprocessing issues

## Integration with Training Pipeline

```python
# Complete training example
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

# Create dataset
dataset = MODISOnlyDataset(
    processed_dir="data/processed/MODIS",
    target_size=(224, 224),
    normalize_modis=True
)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Create model
model = MODISCNN(num_classes=5)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    for ndvi_batch, evi_batch, metadata_batch in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(ndvi_batch, evi_batch)
        
        # Compute loss (you'll need labels for this)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()
```

## Troubleshooting

### Common Issues

1. **"No MODIS files found"**
   - Ensure processed directory contains .npy files
   - Run `python data/preprocess_modis.py --all` first

2. **Memory errors**
   - Reduce batch size
   - Use smaller target_size
   - Process fewer files at once

3. **Shape errors**
   - Check that NDVI and EVI files have matching shapes
   - Verify target_size parameter

4. **Slow loading**
   - Use SSD storage for faster I/O
   - Consider caching frequently accessed data
   - Reduce target_size for smaller arrays

## Dependencies

- `torch>=2.0.0`
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `PIL>=9.5.0`

## File Structure

```
data/
├── modis_dataset.py          # Main dataset classes
├── example_modis_usage.py    # Usage examples
├── processed/
│   └── MODIS/
│       ├── file1_ndvi.npy
│       ├── file1_evi.npy
│       ├── file1_metadata.json
│       └── ...
└── images/                   # Plant images (optional)
    └── plant_001.jpg
```
