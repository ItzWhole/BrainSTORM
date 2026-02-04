@echo off
title STORM Analysis Launcher

echo ========================================
echo    STORM Microscopy Analysis v3
echo ========================================
echo.

echo Checking WSL...
wsl --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: WSL not found. Please install WSL2 first.
    pause
    exit /b 1
)

echo Checking GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo WARNING: No NVIDIA GPU detected - using CPU mode
) else (
    echo GPU detected
)

echo.
echo Starting STORM Analysis...
echo Please wait, this may take 1-2 minutes...
echo.

REM Change to script directory and launch
pushd "%~dp0"
wsl -d Ubuntu-20.04 -- bash -c "cd \"$(wslpath -a '%CD%')\" && chmod +x STORM_Analysis && ./STORM_Analysis"
popd

if errorlevel 1 (
    echo.
    echo Failed to start. Check that Ubuntu 20.04 is installed in WSL2.
    pause
)