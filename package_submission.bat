@echo off
REM BloomWatch Final Submission Package Creator
REM This script creates a ZIP archive of the essential BloomWatch components

echo 🌸 BloomWatch Final Submission Package Creator
echo ============================================

REM Get the current directory
set SCRIPT_DIR=%~dp0
echo Working directory: %SCRIPT_DIR%

REM Create submission directory
echo Creating submission directory...
mkdir "%SCRIPT_DIR%BloomWatch_Submission" 2>nul

REM Copy essential files and directories
echo Copying essential components...

REM Core pipeline
xcopy "%SCRIPT_DIR%pipelines\bloomwatch_temporal_workflow.py" "%SCRIPT_DIR%BloomWatch_Submission\pipelines\" /Y /Q
xcopy "%SCRIPT_DIR%webapp\bloomwatch_explorer.py" "%SCRIPT_DIR%BloomWatch_Submission\webapp\" /Y /Q

REM Documentation
xcopy "%SCRIPT_DIR%README.md" "%SCRIPT_DIR%BloomWatch_Submission\" /Y /Q
xcopy "%SCRIPT_DIR%TEMPORAL_WORKFLOW.md" "%SCRIPT_DIR%BloomWatch_Submission\" /Y /Q
xcopy "%SCRIPT_DIR%BLOOMWATCH_SUBMISSION_SUMMARY.md" "%SCRIPT_DIR%BloomWatch_Submission\" /Y /Q
xcopy "%SCRIPT_DIR%requirements.txt" "%SCRIPT_DIR%BloomWatch_Submission\" /Y /Q

REM Model checkpoint
xcopy "%SCRIPT_DIR%outputs\models\stage2_transfer_learning_bloomwatch.pt" "%SCRIPT_DIR%BloomWatch_Submission\outputs\models\" /Y /Q

REM Final report
xcopy "%SCRIPT_DIR%outputs\final_bloomwatch_report.md" "%SCRIPT_DIR%BloomWatch_Submission\outputs\" /Y /Q

REM Create ZIP archive
echo Creating ZIP archive...
powershell -Command "Compress-Archive -Path '%SCRIPT_DIR%BloomWatch_Submission\*' -DestinationPath '%SCRIPT_DIR%BloomWatch_Final_Submission.zip' -Force"

echo.
echo ============================================
echo ✅ Final submission package created successfully!
echo Package: BloomWatch_Final_Submission.zip
echo ============================================

REM Pause to keep window open
pause