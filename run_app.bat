@echo off
REM BloomWatch Local Development Setup and Launch Script (CMD version)
REM This script sets up the virtual environment and launches the FastAPI app

setlocal enabledelayedexpansion

echo.
echo BloomWatch Local Development Setup
echo =====================================
echo Setting up and launching your BloomWatch FastAPI application...
echo.

REM Check if we're in the right directory
if not exist "requirements.txt" (
 echo requirements.txt not found in current directory.
 echo Please run this script from the BloomWatch project root.
 pause
 exit /b 1
)

if not exist "app\main.py" (
 echo app\main.py not found.
 echo Please run this script from the BloomWatch project root.
 pause
 exit /b 1
)

echo Found BloomWatch project files

REM Check if Python is available
echo ℹ Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
 echo Python not found in PATH
 echo Please install Python 3.9 or later from https://python.org
 pause
 exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python !PYTHON_VERSION!

REM Check if pip is available
echo ℹ Checking pip installation...
pip --version >nul 2>&1
if errorlevel 1 (
 echo pip not found. Please ensure Python is properly installed with pip.
 pause
 exit /b 1
)

echo Found pip

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
 echo ℹ Creating virtual environment...
 python -m venv .venv
 if errorlevel 1 (
 echo Failed to create virtual environment
 pause
 exit /b 1
 )
 echo Virtual environment created
) else (
 echo Virtual environment already exists
)

REM Activate virtual environment
echo ℹ Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
 echo Failed to activate virtual environment
 pause
 exit /b 1
)
echo Virtual environment activated

REM Upgrade pip
echo ℹ Upgrading pip...
python -m pip install --upgrade pip
echo pip upgraded

REM Install dependencies
echo ℹ Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
 echo Failed to install dependencies
 echo You can try installing them manually later
 pause
 exit /b 1
)
echo Dependencies installed successfully

REM Check if uvicorn is available
echo ℹ Checking if uvicorn is available...
uvicorn --version >nul 2>&1
if errorlevel 1 (
 echo uvicorn not found. Installing it now...
 pip install uvicorn[standard]
 if errorlevel 1 (
 echo Failed to install uvicorn
 pause
 exit /b 1
 )
)
echo uvicorn is available

REM Launch the application
echo.
echo Launching BloomWatch FastAPI Application
echo ===========================================
echo.
echo ℹ Starting FastAPI server on port 8000...
echo The application will be available at:
echo Main App: http://127.0.0.1:8000
echo API Docs: http://127.0.0.1:8000/docs
echo Health Check: http://127.0.0.1:8000/health
echo ReDoc: http://127.0.0.1:8000/redoc
echo.
echo Tips:
echo - Press Ctrl+C to stop the server
echo - The server will auto-reload when you make changes
echo - Check the terminal for any error messages
echo - Open your browser to view the application
echo.
echo ============================================================

REM Start the application
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

pause
