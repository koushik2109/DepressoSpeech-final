@echo off
REM DepressoSpeech - Full Trimodal Pipeline (Windows)
REM Runs: organize data -> train trimodal model

echo ==========================================
echo   DepressoSpeech - Full Trimodal Pipeline
echo ==========================================
echo.

cd /d "%~dp0\.."

REM Step 1: Organize Data
if not "%SOURCE_DIR%"=="" (
    echo [Step 1/2] Organizing patient data from %SOURCE_DIR%...
    call windows\organize_data.bat
) else (
    echo [Step 1/2] Skipping data organization (SOURCE_DIR not set)
    echo   To organize: set SOURCE_DIR=C:\path\to\parent && windows\run_trimodal_pipeline.bat
)

REM Step 2: Trimodal Training
echo.
echo [Step 2/2] Training trimodal model...
call windows\train_trimodal.bat

echo.
echo ==========================================
echo   Full trimodal pipeline complete!
echo.
echo   Next steps:
echo     Inference notebook: notebooks\04_inference_pipeline.ipynb
echo ==========================================
pause
