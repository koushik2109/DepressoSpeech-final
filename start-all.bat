@echo off
REM ################################################################################
REM  DepressoSpeech - Start All Services (Windows)
REM  This script starts Frontend, Backend, ML Model, and Swagger services
REM
REM  Usage:
REM    start-all.bat                    Start all services
REM    start-all.bat --no-deps          Start without installing dependencies
REM    start-all.bat --kill-only        Only kill existing processes
REM
REM  Services will run on:
REM    - Frontend:  http://localhost:5173
REM    - Backend:   http://localhost:8000
REM    - ML Model:  http://localhost:8001
REM    - Swagger:   http://localhost:8080
REM ################################################################################

setlocal enabledelayedexpansion

REM Colors (ANSI codes work on Windows 10+)
set "GREEN=[32m"
set "RED=[31m"
set "YELLOW=[33m"
set "BLUE=[34m"
set "NC=[0m"

REM Configuration
set "SCRIPT_DIR=%~dp0"
set "FRONTEND_DIR=%SCRIPT_DIR%Depression-UI"
set "BACKEND_DIR=%SCRIPT_DIR%backend"
set "MODEL_DIR=%SCRIPT_DIR%ModelV2"
set "SWAGGER_DIR=%SCRIPT_DIR%swagger"

set FRONTEND_PORT=5173
set BACKEND_PORT=8000
set MODEL_PORT=8001
set SWAGGER_PORT=8080

set INSTALL_DEPS=true
set KILL_ONLY=false

REM Parse arguments
:parse_args
if "%~1"=="" goto :end_parse_args
if "%~1"=="--no-deps" (
    set INSTALL_DEPS=false
    shift
    goto :parse_args
)
if "%~1"=="--kill-only" (
    set KILL_ONLY=true
    shift
    goto :parse_args
)
echo Unknown option: %~1
echo Usage: %~0 [--no-deps] [--kill-only]
exit /b 1
:end_parse_args

REM ################################################################################
REM Helper Functions
REM ################################################################################

echo.
echo %BLUE%================================================================%NC%
echo %BLUE%  DepressoSpeech - Start All Services (Windows)%NC%
echo %BLUE%================================================================%NC%
echo.

REM Check directories exist
if not exist "%FRONTEND_DIR%" (
    echo %RED%ERROR: Frontend directory not found: %FRONTEND_DIR%%NC%
    exit /b 1
)
if not exist "%BACKEND_DIR%" (
    echo %RED%ERROR: Backend directory not found: %BACKEND_DIR%%NC%
    exit /b 1
)
if not exist "%MODEL_DIR%" (
    echo %RED%ERROR: Model directory not found: %MODEL_DIR%%NC%
    exit /b 1
)

echo %GREEN%Running from: %SCRIPT_DIR%%NC%

REM Kill existing processes on ports
echo.
echo %BLUE%================================================================%NC%
echo %BLUE%  Checking and Clearing Ports%NC%
echo %BLUE%================================================================%NC%
echo.

call :kill_port %FRONTEND_PORT%
call :kill_port %BACKEND_PORT%
call :kill_port %MODEL_PORT%
call :kill_port %SWAGGER_PORT%

if "%KILL_ONLY%"=="true" (
    echo %GREEN%All ports cleared. Exiting.%NC%
    exit /b 0
)

REM Check prerequisites
echo.
echo %BLUE%================================================================%NC%
echo %BLUE%  Checking Prerequisites%NC%
echo %BLUE%================================================================%NC%
echo.

where node >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%ERROR: Node.js is not installed%NC%
    echo Please install Node.js from https://nodejs.org/
    exit /b 1
)
for /f "tokens=*" %%i in ('node --version') do echo %GREEN%Node.js found: %%i%NC%

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%ERROR: Python is not installed%NC%
    echo Please install Python 3
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do echo %GREEN%Python found: %%i%NC%

REM Install dependencies if needed
if "%INSTALL_DEPS%"=="true" (
    REM Frontend dependencies
    echo.
    echo %BLUE%================================================================%NC%
    echo %BLUE%  Installing Frontend Dependencies%NC%
    echo %BLUE%================================================================%NC%
    echo.
    if not exist "%FRONTEND_DIR%\node_modules" (
        echo %YELLOW%Running: npm install%NC%
        pushd "%FRONTEND_DIR%"
        npm install --legacy-peer-deps
        popd
        echo %GREEN%Frontend dependencies installed%NC%
    ) else (
        echo %GREEN%Frontend dependencies already installed%NC%
    )

    REM Backend dependencies
    echo.
    echo %BLUE%================================================================%NC%
    echo %BLUE%  Installing Backend Dependencies%NC%
    echo %BLUE%================================================================%NC%
    echo.
    pip show fastapi >nul 2>&1
    if %errorlevel% neq 0 (
        echo %YELLOW%Running: pip install -r requirements.txt%NC%
        pushd "%BACKEND_DIR%"
        pip install -r requirements.txt
        popd
        echo %GREEN%Backend dependencies installed%NC%
    ) else (
        echo %GREEN%Backend dependencies already installed%NC%
    )

    REM Model dependencies
    echo.
    echo %BLUE%================================================================%NC%
    echo %BLUE%  Installing ML Model Dependencies%NC%
    echo %BLUE%================================================================%NC%
    echo.
    pip show slowapi >nul 2>&1
    if %errorlevel% neq 0 (
        echo %YELLOW%Running: pip install -r requirements.txt%NC%
        pushd "%MODEL_DIR%"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        pip install -r requirements.txt
        popd
        echo %GREEN%ML Model dependencies installed%NC%
    ) else (
        echo %GREEN%ML Model dependencies already installed%NC%
    )
) else (
    echo %YELLOW%Skipping dependency installation (--no-deps flag set)%NC%
)

REM Start all services
echo.
echo %BLUE%================================================================%NC%
echo %BLUE%  Starting Services%NC%
echo %BLUE%================================================================%NC%
echo.

REM Start Frontend
echo %YELLOW%Starting Frontend (Vite Dev Server) on port %FRONTEND_PORT%...%NC%
start "DepressoSpeech-Frontend" /min cmd /c "cd /d "%FRONTEND_DIR%" && npm run dev > "%TEMP%\frontend.log" 2>&1"
timeout /t 2 /nobreak >nul

REM Start Backend
echo %YELLOW%Starting Backend (FastAPI) on port %BACKEND_PORT%...%NC%
start "DepressoSpeech-Backend" /min cmd /c "cd /d "%BACKEND_DIR%" && python -m uvicorn main:app --host 0.0.0.0 --port %BACKEND_PORT% --reload > "%TEMP%\backend.log" 2>&1"
timeout /t 2 /nobreak >nul

REM Start ML Model
echo %YELLOW%Starting ML Model Server on port %MODEL_PORT%...%NC%
if exist "%MODEL_DIR%\.venv\Scripts\activate.bat" (
    start "DepressoSpeech-Model" /min cmd /c "cd /d "%MODEL_DIR%" && .venv\Scripts\activate.bat && python scripts\serve.py --port %MODEL_PORT% > "%TEMP%\model_serve.log" 2>&1"
) else (
    start "DepressoSpeech-Model" /min cmd /c "cd /d "%MODEL_DIR%" && python scripts\serve.py --port %MODEL_PORT% > "%TEMP%\model_serve.log" 2>&1"
)
timeout /t 3 /nobreak >nul

REM Start Swagger
echo %YELLOW%Starting Swagger Documentation on port %SWAGGER_PORT%...%NC%
start "DepressoSpeech-Swagger" /min cmd /c "cd /d "%SWAGGER_DIR%" && python "%SWAGGER_DIR%\serve.py" --port %SWAGGER_PORT% > "%TEMP%\swagger.log" 2>&1"
timeout /t 1 /nobreak >nul

REM Show final status
echo.
echo %BLUE%================================================================%NC%
echo %BLUE%  Service Status%NC%
echo %BLUE%================================================================%NC%
echo.

call :check_port %FRONTEND_PORT% "Frontend (Vite)"
call :check_port %BACKEND_PORT% "Backend (FastAPI)"
call :check_port %MODEL_PORT% "ML Model Server"
call :check_port %SWAGGER_PORT% "Swagger Documentation"

echo.
echo %BLUE%================================================================%NC%
echo %BLUE%  Access Points%NC%
echo %BLUE%================================================================%NC%
echo.
echo %GREEN%Frontend:        %NC%http://localhost:%FRONTEND_PORT%
echo %GREEN%Backend API:     %NC%http://localhost:%BACKEND_PORT%/api/v1
echo %GREEN%Backend Docs:    %NC%http://localhost:%BACKEND_PORT%/docs
echo %GREEN%Model API:       %NC%http://localhost:%MODEL_PORT%
echo %GREEN%Model Docs:      %NC%http://localhost:%MODEL_PORT%/docs
echo %GREEN%Swagger UI:      %NC%http://localhost:%SWAGGER_PORT%

echo.
echo %BLUE%================================================================%NC%
echo %BLUE%  Log Files%NC%
echo %BLUE%================================================================%NC%
echo.
echo Frontend:   %TEMP%\frontend.log
echo Backend:    %TEMP%\backend.log
echo Model:      %TEMP%\model_serve.log
echo Swagger:    %TEMP%\swagger.log

echo.
echo %YELLOW%Press any key to stop all services and exit...%NC%
pause >nul

REM Kill all services
echo.
echo %RED%Shutting down all services...%NC%

REM Kill by window title
taskkill /FI "WindowTitle eq DepressoSpeech-Frontend*" /T /F >nul 2>&1
taskkill /FI "WindowTitle eq DepressoSpeech-Backend*" /T /F >nul 2>&1
taskkill /FI "WindowTitle eq DepressoSpeech-Model*" /T /F >nul 2>&1
taskkill /FI "WindowTitle eq DepressoSpeech-Swagger*" /T /F >nul 2>&1

REM Also try killing by port
call :kill_port %FRONTEND_PORT%
call :kill_port %BACKEND_PORT%
call :kill_port %MODEL_PORT%
call :kill_port %SWAGGER_PORT%

echo %GREEN%All services stopped.%NC%
exit /b 0

REM ################################################################################
REM  Subroutines
REM ################################################################################

:kill_port
set port=%~1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%port% " ^| findstr "LISTENING"') do (
    echo %YELLOW%Killing process on port %port% (PID: %%a)...%NC%
    taskkill /PID %%a /F >nul 2>&1
)
echo %GREEN%Port %port% is free%NC%
exit /b 0

:check_port
set port=%~1
set name=%~2
netstat -aon | findstr ":%port% " | findstr "LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%  [OK]  %name% is running on port %port%%NC%
) else (
    echo %RED%  [!!]  %name% failed to start on port %port%%NC%
)
exit /b 0
