# BloomWatch Final QA Process Runner
# This script runs the complete final QA, reporting, cleanup, and submission process

Write-Host " BloomWatch Final QA Process Runner" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green

# Get the directory of this script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "Working directory: $ScriptDir"

# Run the final QA Python script
Write-Host "Running final QA process..." -ForegroundColor Yellow
python "$ScriptDir\final_qa_submission.py"

if ($LASTEXITCODE -eq 0) {
 Write-Host " Final QA process completed successfully!" -ForegroundColor Green
} else {
 Write-Host " Final QA process failed with exit code $LASTEXITCODE" -ForegroundColor Red
 exit $LASTEXITCODE
}

Write-Host "=====================================" -ForegroundColor Green
Write-Host "Process complete!" -ForegroundColor Green