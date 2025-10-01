# BloomWatch Flowers Classification Training Pipeline

This document explains how to use the fully automated ResNet50 training pipeline for flower classification.

## Pipeline Overview

The training pipeline implements a state-of-the-art approach for training a ResNet50 model on flower images with the following features:

1. **Two-Phase Training**:
   - Phase 1: Train only the classifier head with frozen backbone
   - Phase 2: Fine-tune the entire network with a smaller learning rate

2. **Advanced Data Augmentations**:
   - Training: RandomResizedCrop, RandomHorizontalFlip, RandomRotation, RandomAffine, ColorJitter, RandomErasing
   - Validation/Test: Only Resize and ImageNet normalization

3. **Automatic Device Detection**:
   - Uses GPU if available, otherwise falls back to CPU
   - Adjusts batch size automatically based on device capabilities

4. **Comprehensive Monitoring**:
   - Real-time training logs with loss, accuracy, and learning rate
   - Early stopping to prevent overfitting
   - Gradient clipping for stable training

5. **Automatic Evaluation & Reporting**:
   - Test set evaluation upon training completion
   - Detailed metrics saved in JSON format
   - Confusion matrix visualization
   - Markdown summary report

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- scikit-learn
- matplotlib
- seaborn
- Pillow
- numpy

All dependencies are listed in [requirements.txt](requirements.txt).

## Usage

### Quick Start

1. **Activate your virtual environment** (if using one):
   ```bash
   .venv\Scripts\Activate.ps1  # PowerShell
   # or
   .venv\Scripts\activate.bat   # Command Prompt
   ```

2. **Run the training pipeline**:
   ```bash
   python pipelines/train_flowers_resnet50.py
   ```

   Or use the PowerShell script:
   ```bash
   .\run_flowers_training.ps1
   ```

### Expected Output

The pipeline will create the following outputs in the `outputs/` directory:

- `models/flowers_resnet50_best.pt` - Best model checkpoint
- `flowers_final_metrics.json` - Detailed test metrics
- `flowers_training/confusion_matrix.png` - Confusion matrix visualization
- `flowers_training/training_curves.png` - Training/validation curves
- `flowers_summary.md` - Comprehensive training report

## Pipeline Configuration

The pipeline is configured with the following parameters:

### Dataset Handling
- **Physical splits**: `data/processed/real_flowers/{train,val,test}` (70/20/10)
- **Training augmentations**:
  - `RandomResizedCrop(224)`
  - `RandomHorizontalFlip()`
  - `RandomRotation(30°)`
  - `RandomAffine(degrees=30, translate=(0.1,0.1), scale=(0.8,1.2))`
  - `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)`
  - `RandomErasing(p=0.2)`
- **Validation/Test**: Only `Resize(224)` + ImageNet normalization
- **Normalization**: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]

### Model Setup
- **Architecture**: Pretrained ResNet50 (ImageNet weights)
- **Two-phase training**:
  - Phase 1: Freeze backbone, train classifier head (20 epochs)
  - Phase 2: Unfreeze backbone, fine-tune full network (30 epochs)
- **Optimizer**: AdamW
- **LR Scheduler**: CosineAnnealingLR (Phase 1), OneCycleLR (Phase 2)
- **Gradient clipping**: max norm = 1.0

### Training Loop
- **Epochs**: 20 (Phase 1) + 30 (Phase 2) = 50 total
- **Batch size**: 32 (GPU) / 16 (CPU)
- **EarlyStopping**: patience = 7 (Phase 1) / 10 (Phase 2)
- **Checkpoint**: `outputs/models/flowers_resnet50_best.pt` (overwrites old checkpoint)

## Expected Results

With this pipeline, you should achieve:
- **Test Accuracy**: 90-95% on the flower dataset
- **Training Time**: 30-60 minutes on GPU, 2-4 hours on CPU

The model will be ready for inference immediately after training completes.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in the script
   - Ensure no other GPU processes are running

2. **Missing Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Not Found**:
   - Ensure the dataset is in `data/processed/real_flowers/`
   - Verify train/val/test directories exist with class subdirectories

### Getting Help

If you encounter any issues, please check:
1. All dependencies are installed
2. The dataset is properly organized
3. There's sufficient disk space for outputs
4. GPU drivers are up to date (if using GPU)

## Customization

To modify the pipeline behavior:
- Edit `pipelines/train_flowers_resnet50.py` for training parameters
- Adjust augmentation strategies in the `AdvancedAugmentations` class
- Modify the model architecture in the `create_model` function
- Change output paths in the script as needed

The pipeline is designed to be modular and easy to customize for different requirements.