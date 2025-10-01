# Mini BloomWatch Pipeline - Enhanced Version

## Overview
This enhanced mini pipeline supports both small synthetic datasets and larger expanded datasets with data augmentation, improved training, and better accuracy.

## Features
- **Dual Dataset Support**: Automatically detects and uses expanded dataset if available, falls back to mini synthetic dataset
- **Data Augmentation**: Random flips, rotations, brightness adjustments, and resize-crop for training
- **Enhanced Training**: 20 epochs for expanded dataset, early stopping, learning rate scheduling
- **CPU Optimized**: Runs efficiently on CPU with appropriate batch sizes
- **BloomWatch JSON Output**: Generates prediction JSON in BloomWatch API format

## Dataset Structure

### Mini Dataset (Synthetic)
```
data/
├── mini_images/
│   ├── bud/
│   ├── early_bloom/
│   ├── full_bloom/
│   ├── late_bloom/
│   └── dormant/
├── processed/MINI/
│   ├── tile_h31v08_t0.ndvi.npy
│   └── tile_h31v08_t0.evi.npy
└── mini_annotations.csv
```

### Expanded Dataset (Synthetic)
```
data/expanded_dataset/
├── plant_images/
│   ├── bud/ (50 images)
│   ├── early_bloom/ (50 images)
│   ├── full_bloom/ (50 images)
│   ├── late_bloom/ (50 images)
│   └── dormant/ (50 images)
├── ndvi/
│   ├── ndvi_000.npy
│   ├── evi_000.npy
│   └── ... (20 pairs)
└── metadata.csv
```

## Usage

### Quick Start
```powershell
# Run the complete pipeline
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_mini_pipeline.ps1
```

### Manual Steps
```bash
# 1. Generate expanded dataset (if needed)
python pipelines/generate_expanded_dataset.py

# 2. Run the enhanced pipeline
python pipelines/mini_bloomwatch.py
```

## Outputs

### Models
- **Mini Dataset**: `outputs/models/mini_bloomwatch.pt`
- **Expanded Dataset**: `outputs/models/expanded_bloomwatch.pt`

### Predictions
- **Mini Dataset**: `outputs/mini_prediction.json`
- **Expanded Dataset**: `outputs/expanded_prediction.json`

### Sample Prediction JSON
```json
{
  "predicted_class": 2,
  "confidence": 0.2488,
  "class_confidences": {
    "bud": 0.2232,
    "early_bloom": 0.2237,
    "full_bloom": 0.2488,
    "late_bloom": 0.2227,
    "dormant": 0.0817
  },
  "probabilities": [0.2232, 0.2237, 0.2488, 0.2227, 0.0817],
  "true_label": 0,
  "class_names": ["bud", "early_bloom", "full_bloom", "late_bloom", "dormant"],
  "inference_sample": {
    "image_path": "bud/img_035.png",
    "plant_id": "plant_0_003",
    "timestamp": "2024-02-10"
  },
  "model_path": "D:\\NASA(0)\\BloomWatch\\outputs\\models\\expanded_bloomwatch.pt",
  "dataset_type": "expanded",
  "training_epochs": 20
}
```

## Performance Results

### Mini Dataset (12 images)
- **Training Time**: ~7 seconds
- **Final Accuracy**: ~22% (limited by small dataset)
- **Model Size**: Small CNN with 5-channel input

### Expanded Dataset (250 images)
- **Training Time**: ~2.5 minutes
- **Final Accuracy**: ~40% (significant improvement)
- **Data Augmentation**: Applied during training
- **Early Stopping**: Prevents overfitting

## Technical Details

### Data Augmentation
- **Horizontal Flip**: 50% probability
- **Rotation**: ±15° with 50% probability
- **Brightness**: 0.8-1.2x factor with 50% probability
- **Resize-Crop**: 0.8-1.0x scale with random crop

### Model Architecture
- **Input**: 5-channel tensor (RGB + NDVI + EVI)
- **CNN Layers**: 2 conv blocks with max pooling
- **Classifier**: 2 fully connected layers with dropout
- **Output**: 5 classes (bloom stages)

### Training Configuration
- **Optimizer**: Adam with different learning rates
- **Mini Dataset**: lr=1e-3, 10 epochs
- **Expanded Dataset**: lr=1e-4, 20 epochs, weight decay
- **Scheduler**: StepLR for expanded dataset
- **Early Stopping**: 5 epochs patience

## Requirements
- Python 3.11+
- PyTorch 2.3.1 (CPU)
- NumPy 1.26.4
- Pillow 10.4.0
- psutil 5.9.8

## Hackathon Ready
This pipeline is ready for hackathon submission with:
- ✅ Automated end-to-end execution
- ✅ Scientific results (NDVI + plant images → CNN → predictions)
- ✅ BloomWatch API JSON format
- ✅ CPU-compatible (no GPU required)
- ✅ Demonstrable accuracy improvement with larger dataset
- ✅ Complete documentation and outputs
