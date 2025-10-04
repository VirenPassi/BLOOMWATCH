# BloomWatch Local Development Setup and Launch Script
param(
 [int]$Port = 8000,
 [switch]$Help = $false
)

if ($Help) {
 Write-Host "BloomWatch Local Development Setup Script"
 Write-Host "Usage: .\run_app_simple.ps1 [-Port <port>] [-Help]"
 Write-Host "Example: .\run_app_simple.ps1 -Port 8080"
 exit 0
}

Write-Host "BloomWatch Local Development Setup" -ForegroundColor Magenta
Write-Host "====================================" -ForegroundColor Magenta

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

# Install dependencies
Write-Host "INFO: Installing dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt
Write-Host "SUCCESS: Dependencies installed" -ForegroundColor Green

# Launch the application
Write-Host ""
Write-Host "Launching BloomWatch FastAPI Application" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "The application will be available at:" -ForegroundColor Yellow
Write-Host " Main App: http://127.0.0.1:$Port" -ForegroundColor Yellow
Write-Host " API Docs: http://127.0.0.1:$Port/docs" -ForegroundColor Yellow
Write-Host " Health Check: http://127.0.0.1:$Port/health" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Magenta

# Start the application
python -m uvicorn app.main:app --host 127.0.0.1 --port $Port --reload