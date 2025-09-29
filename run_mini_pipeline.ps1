$ErrorActionPreference = 'Stop'

Write-Host "=== Mini BloomWatch Pipeline Runner ===" -ForegroundColor Cyan
Write-Host "Working directory: $(Get-Location)" -ForegroundColor DarkGray

# 1) Ensure Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  Write-Error "Python not found in PATH. Please install Python 3.11 and re-run."
  exit 1
}

# 2) Create/activate venv
if (-not (Test-Path ".venv")) {
  Write-Host "Creating virtual environment (.venv)" -ForegroundColor Cyan
  python -m venv .venv
}

$venvActivate = Join-Path ".venv" "Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
  Write-Error "Activation script not found at $venvActivate"
  exit 1
}

Write-Host "Activating virtual environment" -ForegroundColor Cyan
. $venvActivate

# 3) Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# 4) Install enhanced dependencies for improved accuracy and quality assurance
Write-Host "Installing enhanced dependencies" -ForegroundColor Cyan
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 --extra-index-url https://pypi.org/simple
pip install numpy==1.26.4 pillow==10.4.0 psutil==5.9.8 scipy==1.11.4 pandas==2.1.4 matplotlib==3.7.2 seaborn==0.12.2 scikit-learn==1.3.2

# 5) Generate expanded dataset if needed
if (-not (Test-Path "data\expanded_dataset\metadata.csv")) {
    Write-Host "Generating expanded synthetic dataset..." -ForegroundColor Cyan
    python pipelines/generate_expanded_dataset.py
}

# 6) Run the pipeline
Write-Host "Running enhanced pipeline" -ForegroundColor Cyan
python pipelines/mini_bloomwatch.py

if ($LASTEXITCODE -ne 0) {
  Write-Error "Mini pipeline failed with exit code $LASTEXITCODE"
  exit $LASTEXITCODE
}

Write-Host "Enhanced pipeline with quality assurance completed!" -ForegroundColor Green
Write-Host "📊 Outputs available in .\outputs" -ForegroundColor Cyan
Write-Host "- Models: .\outputs\models\*_bloomwatch.pt" -ForegroundColor DarkGray
Write-Host "- Predictions: .\outputs\*_prediction.json" -ForegroundColor DarkGray
Write-Host "- Learning curves: .\outputs\learning_curves.png" -ForegroundColor DarkGray
Write-Host "- Confusion matrix: .\outputs\confusion_matrix.png" -ForegroundColor DarkGray
Write-Host "- Dataset check: .\outputs\dataset_check.json" -ForegroundColor DarkGray


