@echo off
REM DepressoSpeech - Trimodal Training Pipeline (Windows)
REM Trains Audio + Video + Text trimodal fusion model
REM Supports both regression (PHQ-8 score) and binary classification (depressed/not)
REM
REM Usage:
REM   Regression (default):    windows\train_trimodal.bat
REM   Classification mode:     set CLASSIFICATION=1 && windows\train_trimodal.bat
REM   With NPZ cache:          set NPZ_CACHE=data\npz && windows\train_trimodal.bat

echo ==========================================
echo   DepressoSpeech - Trimodal Training
echo ==========================================

cd /d "%~dp0\.."
call .venv\Scripts\activate.bat

if "%CONFIG%"=="" set CONFIG=configs\trimodal_v2_config.yaml
if "%PATIENTS_DIR%"=="" set PATIENTS_DIR=data\raw
if "%LABELS_CSV%"=="" set LABELS_CSV=data\labels.csv
if "%CLASSIFICATION%"=="" set CLASSIFICATION=0
if "%NPZ_CACHE%"=="" set NPZ_CACHE=

echo Configuration:
echo   Config:          %CONFIG%
echo   Patients Dir:    %PATIENTS_DIR%
echo   Labels CSV:      %LABELS_CSV%
echo   Classification:  %CLASSIFICATION%
if "%NPZ_CACHE%"=="" (
    echo   NPZ Cache:       (disabled, loads raw CSVs)
) else (
    echo   NPZ Cache:       %NPZ_CACHE%
)
echo.

REM Check GPU
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'Device: CPU')"

REM Ensure checkpoint dirs
if not exist "checkpoints" mkdir checkpoints

REM Auto-build NPZ cache if requested but not present
if not "%NPZ_CACHE%"=="" (
    if not exist "%NPZ_CACHE%" (
        echo.
        echo [NPZ] Cache directory not found. Building from raw CSVs...
        python scripts\verify_npz.py --build --raw-dir %PATIENTS_DIR% --output-dir %NPZ_CACHE% --labels %LABELS_CSV%
        echo [NPZ] Cache built.
    )
)

REM Assemble optional npz_cache flag
set NPZ_FLAG=
if not "%NPZ_CACHE%"=="" set NPZ_FLAG=--npz_cache %NPZ_CACHE%

echo.
if "%CLASSIFICATION%"=="1" (
    echo Starting trimodal CLASSIFIER training...
    echo ------------------------------------------
    python scripts\train_trimodal_classifier.py --config %CONFIG% --data_dir %PATIENTS_DIR% --labels %LABELS_CSV% %NPZ_FLAG%
    echo ------------------------------------------
    echo   Classification training complete!
    echo   Artifacts saved:
    echo     Best Model:  checkpoints\best_classifier.pt
    echo     Latest:      checkpoints\latest_classifier.pt
) else (
    echo Starting trimodal REGRESSION training...
    echo ------------------------------------------
    python scripts\train_trimodal.py --config %CONFIG% --patients-dir %PATIENTS_DIR%
    echo ------------------------------------------
    echo   Regression training complete!
    echo   Artifacts saved:
    echo     Best Model:  checkpoints\best_trimodal.pt
    echo     Stage 1:     checkpoints\best_trimodal_text.pt
    echo     Stage 2:     checkpoints\best_trimodal_audio_text.pt
)
echo ==========================================
pause
