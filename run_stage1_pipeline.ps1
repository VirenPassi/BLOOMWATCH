# BloomWatch Stage-1 Pipeline Runner
# Automates complete dataset expansion and validation

Write-Host "=== BloomWatch Stage-1 Pipeline Runner ===" -ForegroundColor Cyan
Write-Host "Working directory: $(Get-Location)" -ForegroundColor Gray

# 1) Check Python availability
Write-Host "Checking Python installation..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Error "Python not found. Please install Python 3.11+ and try again."
    exit 1
}

# 2) Create/activate virtual environment
Write-Host "Setting up virtual environment..." -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

$venvActivate = ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
    Write-Host "Virtual environment activated" -ForegroundColor Green
} else {
    Write-Error "Failed to activate virtual environment"
    exit 1
}

# 3) Upgrade pip and install dependencies
Write-Host "Installing dependencies for Stage-1 pipeline..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 --extra-index-url https://pypi.org/simple
pip install numpy==1.26.4 pillow==10.4.0 psutil==5.9.8 scipy==1.11.4 pandas==2.1.4 matplotlib==3.7.2 seaborn==0.12.2 scikit-learn==1.3.2 requests==2.31.0

# 4) Run Stage-1 pipeline
Write-Host "Starting Stage-1 dataset expansion pipeline..." -ForegroundColor Cyan
python run_stage1_pipeline.py

if ($LASTEXITCODE -ne 0) {
    Write-Error "Stage-1 pipeline failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# 5) Display results
Write-Host "Stage-1 pipeline completed successfully!" -ForegroundColor Green
Write-Host "📊 Check outputs in .\outputs\" -ForegroundColor Cyan
Write-Host "- Stage-1 report: .\outputs\stage1_report.md" -ForegroundColor DarkGray
Write-Host "- MODIS data: .\data\raw\MODIS\stage1\" -ForegroundColor DarkGray
Write-Host "- Processed arrays: .\data\processed\MODIS\stage1\" -ForegroundColor DarkGray
Write-Host "- Plant images: .\data\expanded_dataset\plant_images\" -ForegroundColor DarkGray
Write-Host "- Training results: .\outputs\*_prediction.json" -ForegroundColor DarkGray
Write-Host "- Quality assurance: .\outputs\dataset_check.json" -ForegroundColor DarkGray

Write-Host "`n🎯 Next steps:" -ForegroundColor Yellow
Write-Host "1. Review stage1_report.md for scaling recommendations" -ForegroundColor White
Write-Host "2. Check learning curves and confusion matrix" -ForegroundColor White
Write-Host "3. Consider scaling to 10GB dataset if recommended" -ForegroundColor White
