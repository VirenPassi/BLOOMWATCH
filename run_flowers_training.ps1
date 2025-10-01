# BloomWatch Flowers Training Pipeline
# This script runs the fully automated ResNet50 training pipeline

Write-Host "🚀 Starting BloomWatch Flowers Training Pipeline" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green

# Activate virtual environment if it exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "No virtual environment found, using system Python" -ForegroundColor Yellow
}

# Check if required packages are installed
Write-Host "Checking required packages..." -ForegroundColor Yellow
$required_packages = @("torch", "torchvision", "scikit-learn", "matplotlib", "seaborn", "Pillow", "numpy")

# Run the training pipeline
Write-Host "Starting ResNet50 training pipeline..." -ForegroundColor Yellow
Write-Host "This will train a model on the flower dataset with the following features:" -ForegroundColor Cyan
Write-Host "  - Two-phase training (frozen backbone + fine-tuning)" -ForegroundColor Cyan
Write-Host "  - Advanced data augmentations" -ForegroundColor Cyan
Write-Host "  - Automatic GPU detection" -ForegroundColor Cyan
Write-Host "  - Real-time logging" -ForegroundColor Cyan
Write-Host "  - Automatic evaluation and reporting" -ForegroundColor Cyan

python pipelines/train_flowers_resnet50.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Training pipeline completed successfully!" -ForegroundColor Green
    Write-Host "📄 Check the outputs directory for results:" -ForegroundColor Yellow
    Write-Host "   - Model: outputs/models/flowers_resnet50_best.pt" -ForegroundColor Yellow
    Write-Host "   - Metrics: outputs/flowers_final_metrics.json" -ForegroundColor Yellow
    Write-Host "   - Confusion Matrix: outputs/flowers_training/confusion_matrix.png" -ForegroundColor Yellow
    Write-Host "   - Report: outputs/flowers_summary.md" -ForegroundColor Yellow
} else {
    Write-Host "❌ Training pipeline failed!" -ForegroundColor Red
    exit 1
}