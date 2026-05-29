@echo off
REM DepressoSpeech - FastAPI Server (Windows)
REM Runs the REST API for inference (keeps running on host)
REM Endpoints: POST /predict, POST /predict/batch, GET /health

echo ==========================================
echo   DepressoSpeech - API Server
echo ==========================================

cd /d "%~dp0\.."
call .venv\Scripts\activate.bat

if "%CONFIG%"=="" set CONFIG=configs\inference_config.yaml
if "%HOST%"=="" set HOST=0.0.0.0
if "%PORT%"=="" set PORT=8000
if "%LOG_LEVEL%"=="" set LOG_LEVEL=info

if not exist "checkpoints\best_model.pt" (
    echo ERROR: Model checkpoint not found at checkpoints\best_model.pt
    echo Please train the model first: windows\train.bat
    exit /b 1
)

echo Server Configuration:
echo   Config:    %CONFIG%
echo   Host:      %HOST%
echo   Port:      %PORT%
echo   Log Level: %LOG_LEVEL%
echo.
echo Endpoints:
echo   POST http://%HOST%:%PORT%/predict              - Single file prediction (audio-only)
echo   POST http://%HOST%:%PORT%/predict/batch        - Batch prediction
echo   POST http://%HOST%:%PORT%/predict/multimodal   - Multimodal prediction (audio+video+text)
echo   GET  http://%HOST%:%PORT%/health               - Health check
echo.
echo Classification mode: num_classes=2 in config enables binary depression detection
echo   Output: depression_probability, predicted_label (0/1), phq8_score (scaled)
echo.
echo Starting server (Ctrl+C to stop)...
echo ------------------------------------------

python scripts\serve.py --host %HOST% --port %PORT% --config %CONFIG% --log-level %LOG_LEVEL%
