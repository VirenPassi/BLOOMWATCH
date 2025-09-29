# BloomWatch Stage-2 Pipeline Runner
# Automates complete Stage-2 dataset expansion with fine-tuning and inference

param(
    [string]$InferenceImage = "",
    [switch]$FineTune = $false,
    [switch]$SkipDownload = $false,
    [switch]$SkipPreprocess = $false,
    [switch]$SkipSynthesis = $false
)

Write-Host "=== BloomWatch Stage-2 Pipeline Runner ===" -ForegroundColor Cyan
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
Write-Host "Installing dependencies for Stage-2 pipeline..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 --extra-index-url https://pypi.org/simple
pip install numpy==1.26.4 pillow==10.4.0 psutil==5.9.8 scipy==1.11.4 pandas==2.1.4 matplotlib==3.7.2 seaborn==0.12.2 scikit-learn==1.3.2 requests==2.31.0

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install dependencies."
    exit 1
}

# 4) Run Stage-2 pipeline
Write-Host "Starting Stage-2 dataset expansion pipeline..." -ForegroundColor Cyan

# Build command arguments
$args = @()

if ($InferenceImage) {
    $args += "--inference", $InferenceImage
    Write-Host "Running inference mode on: $InferenceImage" -ForegroundColor Yellow
} else {
    if ($FineTune) {
        $args += "--fine-tune"
        Write-Host "Fine-tuning enabled" -ForegroundColor Yellow
    }
    
    if ($SkipDownload) {
        $args += "--skip-download"
        Write-Host "Skipping MODIS download" -ForegroundColor Yellow
    }
    
    if ($SkipPreprocess) {
        $args += "--skip-preprocess"
        Write-Host "Skipping MODIS preprocessing" -ForegroundColor Yellow
    }
    
    if ($SkipSynthesis) {
        $args += "--skip-synthesis"
        Write-Host "Skipping plant synthesis" -ForegroundColor Yellow
    }
}

$command = "python run_stage2_pipeline.py " + ($args -join " ")
Write-Host "Command: $command" -ForegroundColor Gray

Invoke-Expression $command

if ($LASTEXITCODE -ne 0) {
    Write-Error "Stage-2 pipeline failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# 5) Display results
Write-Host "Stage-2 pipeline completed successfully!" -ForegroundColor Green
Write-Host "📊 Check outputs in .\outputs\" -ForegroundColor Cyan
Write-Host "- Stage-2 report: .\outputs\stage2_report.md" -ForegroundColor DarkGray
Write-Host "- MODIS data: .\data\raw\MODIS\stage2\" -ForegroundColor DarkGray
Write-Host "- Processed arrays: .\data\processed\MODIS\stage2\" -ForegroundColor DarkGray
Write-Host "- Plant images: .\data\expanded_dataset\plant_images\" -ForegroundColor DarkGray
Write-Host "- Training results: .\outputs\*stage2*_prediction.json" -ForegroundColor DarkGray
Write-Host "- Learning curves: .\outputs\learning_curves.png" -ForegroundColor DarkGray
Write-Host "- Confusion matrix: .\outputs\confusion_matrix.png" -ForegroundColor DarkGray

if ($InferenceImage) {
    Write-Host "- Inference result: .\outputs\inference_result_stage2.json" -ForegroundColor DarkGray
}

Write-Host "`n🎯 Next steps:" -ForegroundColor Yellow
Write-Host "1. Review stage2_report.md for scaling recommendations" -ForegroundColor White
Write-Host "2. Check learning curves and confusion matrix" -ForegroundColor White
Write-Host "3. Consider fine-tuning if not already enabled" -ForegroundColor White
Write-Host "4. Run inference on specific images if needed" -ForegroundColor White
