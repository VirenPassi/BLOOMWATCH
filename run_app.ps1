# BloomWatch Local Development Setup and Launch Script
# This script sets up the virtual environment and launches the FastAPI app

param(
    [string]$PythonVersion = "3.9",
    [int]$Port = 8000,
    [switch]$SkipDependencies = $false,
    [switch]$Help = $false
)

# Show help if requested
if ($Help) {
    Write-Host @"
BloomWatch Local Development Setup Script

USAGE:
    .\run_app.ps1 [OPTIONS]

OPTIONS:
    -PythonVersion <version>    Python version to use (default: 3.9)
    -Port <port>                Port to run the app on (default: 8000)
    -SkipDependencies          Skip installing dependencies (for faster testing)
    -Help                      Show this help message

EXAMPLES:
    .\run_app.ps1                           # Run with defaults
    .\run_app.ps1 -Port 8080                # Run on port 8080
    .\run_app.ps1 -PythonVersion 3.11       # Use Python 3.11
    .\run_app.ps1 -SkipDependencies         # Skip dependency installation

"@
    exit 0
}

# Set error action preference
$ErrorActionPreference = "Stop"

# Color functions for better output
function Write-Success { param($Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-Info { param($Message) Write-Host "ℹ️  $Message" -ForegroundColor Cyan }
function Write-Warning { param($Message) Write-Host "⚠️  $Message" -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host "❌ $Message" -ForegroundColor Red }

# Script header
Write-Host @"
🌸 BloomWatch Local Development Setup
=====================================
Setting up and launching your BloomWatch FastAPI application...

"@ -ForegroundColor Magenta

# Check if we're in the right directory
$ProjectRoot = Get-Location
$RequirementsFile = Join-Path $ProjectRoot "requirements.txt"
$AppMainFile = Join-Path $ProjectRoot "app\main.py"

if (-not (Test-Path $RequirementsFile)) {
    Write-Error "requirements.txt not found in current directory. Please run this script from the BloomWatch project root."
    exit 1
}

if (-not (Test-Path $AppMainFile)) {
    Write-Error "app\main.py not found. Please run this script from the BloomWatch project root."
    exit 1
}

Write-Success "Found BloomWatch project files"

# Check PowerShell execution policy
$ExecutionPolicy = Get-ExecutionPolicy
if ($ExecutionPolicy -eq "Restricted") {
    Write-Warning "PowerShell execution policy is set to 'Restricted'"
    Write-Info "Attempting to set execution policy for current user..."
    try {
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
        Write-Success "Execution policy updated successfully"
    }
    catch {
        Write-Warning "Could not update execution policy. You may need to run:"
        Write-Host "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    }
}

# Check if Python is available
Write-Info "Checking Python installation..."
try {
    $PythonVersionInstalled = python --version 2>&1
    if ($PythonVersionInstalled -match "Python (\d+\.\d+)") {
        $InstalledVersion = $Matches[1]
        Write-Success "Found Python $InstalledVersion"
        
        # Check if version meets requirements
        $RequiredVersion = [version]$PythonVersion
        $CurrentVersion = [version]$InstalledVersion
        if ($CurrentVersion -lt $RequiredVersion) {
            Write-Warning "Python version $InstalledVersion is older than required $PythonVersion"
            Write-Info "Continuing anyway, but some features may not work properly"
        }
    }
    else {
        Write-Error "Python not found or version could not be determined"
        Write-Info "Please install Python $PythonVersion or later from https://python.org"
        exit 1
    }
}
catch {
    Write-Error "Python not found in PATH"
    Write-Info "Please install Python $PythonVersion or later from https://python.org"
    exit 1
}

# Check if pip is available
Write-Info "Checking pip installation..."
try {
    $PipVersion = pip --version 2>&1
    Write-Success "Found pip: $PipVersion"
}
catch {
    Write-Error "pip not found. Please ensure Python is properly installed with pip."
    exit 1
}

# Create virtual environment if it doesn't exist
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"

if (-not (Test-Path $VenvPath)) {
    Write-Info "Creating virtual environment..."
    try {
        python -m venv .venv
        Write-Success "Virtual environment created at $VenvPath"
    }
    catch {
        Write-Error "Failed to create virtual environment: $_"
        exit 1
    }
}
else {
    Write-Success "Virtual environment already exists at $VenvPath"
}

# Activate virtual environment
Write-Info "Activating virtual environment..."
try {
    if (Test-Path $VenvActivateScript) {
        & $VenvActivateScript
        Write-Success "Virtual environment activated"
    }
    else {
        Write-Error "Virtual environment activation script not found at $VenvActivateScript"
        exit 1
    }
}
catch {
    Write-Error "Failed to activate virtual environment: $_"
    exit 1
}

# Upgrade pip
Write-Info "Upgrading pip..."
try {
    python -m pip install --upgrade pip
    Write-Success "pip upgraded successfully"
}
catch {
    Write-Warning "Failed to upgrade pip: $_"
    Write-Info "Continuing with current pip version..."
}

# Install dependencies
if (-not $SkipDependencies) {
    Write-Info "Installing dependencies from requirements.txt..."
    try {
        pip install -r requirements.txt
        Write-Success "Dependencies installed successfully"
    }
    catch {
        Write-Error "Failed to install dependencies: $_"
        Write-Info "You can try running with -SkipDependencies to skip this step"
        exit 1
    }
}
else {
    Write-Warning "Skipping dependency installation (as requested)"
}

# Check if uvicorn is available
Write-Info "Checking if uvicorn is available..."
try {
    $UvicornVersion = uvicorn --version 2>&1
    Write-Success "Found uvicorn: $UvicornVersion"
}
catch {
    Write-Error "uvicorn not found. Please install it with: pip install uvicorn[standard]"
    exit 1
}

# Launch the application
Write-Host @"

🚀 Launching BloomWatch FastAPI Application
===========================================

"@ -ForegroundColor Green

Write-Info "Starting FastAPI server on port $Port..."
Write-Info "The application will be available at:"
Write-Host "   🌐 Main App: http://127.0.0.1:$Port" -ForegroundColor Yellow
Write-Host "   📚 API Docs: http://127.0.0.1:$Port/docs" -ForegroundColor Yellow
Write-Host "   🔍 Health Check: http://127.0.0.1:$Port/health" -ForegroundColor Yellow
Write-Host "   📖 ReDoc: http://127.0.0.1:$Port/redoc" -ForegroundColor Yellow

Write-Host @"

💡 Tips:
   - Press Ctrl+C to stop the server
   - The server will auto-reload when you make changes
   - Check the terminal for any error messages
   - Open your browser to view the application

"@ -ForegroundColor Cyan

Write-Host "=" * 60 -ForegroundColor Magenta

# Start the application
try {
    python -m uvicorn app.main:app --host 127.0.0.1 --port $Port --reload
}
catch {
    Write-Error "Failed to start the application: $_"
    Write-Info "You can try running manually with:"
    Write-Host "python -m uvicorn app.main:app --host 127.0.0.1 --port $Port --reload" -ForegroundColor Yellow
    exit 1
}