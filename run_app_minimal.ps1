# BloomWatch Minimal Setup Script (Skips problematic dependencies)
param(
    [int]$Port = 8000,
    [switch]$Help = $false
)

if ($Help) {
    Write-Host "BloomWatch Minimal Setup Script"
    Write-Host "Usage: .\run_app_minimal.ps1 [-Port <port>] [-Help]"
    Write-Host "This version skips GDAL and other problematic dependencies"
    exit 0
}

Write-Host "BloomWatch Minimal Setup" -ForegroundColor Magenta
Write-Host "=========================" -ForegroundColor Magenta

# Check if we're in the right directory
if (-not (Test-Path "requirements.txt")) {
    Write-Host "ERROR: requirements.txt not found. Please run from BloomWatch project root." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "app\main.py")) {
    Write-Host "ERROR: app\main.py not found. Please run from BloomWatch project root." -ForegroundColor Red
    exit 1
}

Write-Host "SUCCESS: Found BloomWatch project files" -ForegroundColor Green

# Check Python
Write-Host "INFO: Checking Python installation..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    Write-Host "SUCCESS: Found Python: $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "ERROR: Python not found. Please install Python 3.9+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Create virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "INFO: Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
    Write-Host "SUCCESS: Virtual environment created" -ForegroundColor Green
}
else {
    Write-Host "SUCCESS: Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "INFO: Activating virtual environment..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "INFO: Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install minimal dependencies for FastAPI app to run
Write-Host "INFO: Installing minimal dependencies..." -ForegroundColor Cyan
$minimalDeps = @(
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.22.0",
    "pydantic>=2.0.0",
    "python-multipart>=0.0.6",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "pillow>=9.5.0",
    "psutil>=5.9.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "plotly>=5.14.0",
    "seaborn>=0.12.0",
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
    "pyyaml>=6.0"
)

foreach ($dep in $minimalDeps) {
    Write-Host "INFO: Installing $dep..." -ForegroundColor Cyan
    pip install $dep --quiet
}

Write-Host "SUCCESS: Minimal dependencies installed" -ForegroundColor Green

# Check if uvicorn is available
Write-Host "INFO: Checking if uvicorn is available..." -ForegroundColor Cyan
try {
    $UvicornVersion = uvicorn --version 2>&1
    Write-Host "SUCCESS: Found uvicorn: $UvicornVersion" -ForegroundColor Green
}
catch {
    Write-Host "ERROR: uvicorn not found. Installing it now..." -ForegroundColor Red
    pip install uvicorn[standard]
    Write-Host "SUCCESS: uvicorn installed" -ForegroundColor Green
}

# Launch the application
Write-Host ""
Write-Host "Launching BloomWatch FastAPI Application" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "The application will be available at:" -ForegroundColor Yellow
Write-Host "   Main App: http://127.0.0.1:$Port" -ForegroundColor Yellow
Write-Host "   API Docs: http://127.0.0.1:$Port/docs" -ForegroundColor Yellow
Write-Host "   Health Check: http://127.0.0.1:$Port/health" -ForegroundColor Yellow
Write-Host ""
Write-Host "Note: Some features may not work without full dependencies" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Magenta

# Start the application
python -m uvicorn app.main:app --host 127.0.0.1 --port $Port --reload
