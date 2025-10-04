@echo off
title BloomWatch Flowers Training Pipeline

echo Starting BloomWatch Flowers Training Pipeline
echo =============================================

REM Check if virtual environment exists and activate it
if exist ".venv\Scripts\activate.bat" (
 echo Activating virtual environment...
 call .venv\Scripts\activate.bat
) else (
 echo No virtual environment found, using system Python
)

echo.
echo Starting ResNet50 training pipeline...
echo This will train a model on the flower dataset with the following features:
echo - Two-phase training (frozen backbone + fine-tuning)
echo - Advanced data augmentations
echo - Automatic GPU detection
echo - Real-time logging
echo - Automatic evaluation and reporting
echo.

python pipelines/train_flowers_resnet50.py

if %ERRORLEVEL% EQU 0 (
 echo.
 echo Training pipeline completed successfully!
 echo Check the outputs directory for results:
 echo - Model: outputs/models/flowers_resnet50_best.pt
 echo - Metrics: outputs/flowers_final_metrics.json
 echo - Confusion Matrix: outputs/flowers_training/confusion_matrix.png
 echo - Report: outputs/flowers_summary.md
) else (
 echo.
 echo Training pipeline failed!
 pause
 exit /b 1
)

echo.
echo Press any key to exit...
pause >nul