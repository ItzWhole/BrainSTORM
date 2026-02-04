@echo off
title STORM Microscopy Analysis Launcher
color 0A

:MAIN
cls
echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║                STORM Microscopy Analysis v3                  ║
echo  ║              Deep Learning Z-Height Prediction               ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.
echo  [1] Launch STORM Analysis
echo  [2] Check System Requirements
echo  [3] View Documentation
echo  [4] Exit
echo.
set /p choice="Select an option (1-4): "

if "%choice%"=="1" goto LAUNCH
if "%choice%"=="2" goto CHECK_SYSTEM
if "%choice%"=="3" goto DOCS
if "%choice%"=="4" goto EXIT
goto MAIN

:LAUNCH
cls
echo Starting STORM Analysis...
echo.

REM Check WSL
wsl --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: WSL not found
    echo Please install WSL2 with Ubuntu 20.04
    pause
    goto MAIN
)

REM Check GPU
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  WARNING: No NVIDIA GPU detected - CPU mode only
) else (
    echo ✅ NVIDIA GPU detected
)

echo ✅ WSL2 detected
echo.
echo Launching application...

REM Get current directory and convert to WSL path
set "current_dir=%~dp0"
set "wsl_path=/mnt/c%current_dir:C:=%"
set "wsl_path=%wsl_path:\=/%"

wsl -d Ubuntu-20.04 -e bash -c "cd '%wsl_path%' && chmod +x STORM_Analysis && ./STORM_Analysis"

if %errorlevel% neq 0 (
    echo.
    echo ❌ Failed to start application
    echo Check that Ubuntu 20.04 is installed in WSL2
    pause
)
goto MAIN

:CHECK_SYSTEM
cls
echo System Requirements Check
echo ========================
echo.

REM Check Windows version
for /f "tokens=4-5 delims=. " %%i in ('ver') do set VERSION=%%i.%%j
echo Windows Version: %VERSION%

REM Check WSL
wsl --version >nul 2>&1
if %errorlevel% neq 0 (
    echo WSL2: ❌ Not installed
    echo   Install with: wsl --install -d Ubuntu-20.04
) else (
    echo WSL2: ✅ Installed
)

REM Check Ubuntu
wsl -d Ubuntu-20.04 -e echo "Ubuntu check" >nul 2>&1
if %errorlevel% neq 0 (
    echo Ubuntu 20.04: ❌ Not found in WSL2
) else (
    echo Ubuntu 20.04: ✅ Available
)

REM Check NVIDIA
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo NVIDIA GPU: ❌ Not detected
    echo   Install drivers from: https://www.nvidia.com/drivers/
) else (
    echo NVIDIA GPU: ✅ Detected
)

echo.
pause
goto MAIN

:DOCS
cls
echo Documentation
echo =============
echo.
echo Opening README.txt...
start notepad README.txt
goto MAIN

:EXIT
exit