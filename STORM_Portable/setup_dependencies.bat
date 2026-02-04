@echo off
echo Setting up STORM Analysis dependencies...
echo.

REM Check if WSL is available
wsl --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: WSL not found. Please install WSL2 first.
    echo Run: wsl --install -d Ubuntu-20.04
    pause
    exit /b 1
)

echo Installing Python dependencies in WSL...
wsl -d Ubuntu-20.04 -e bash -c "
    echo 'Updating package lists...'
    sudo apt update
    
    echo 'Installing Python and pip...'
    sudo apt install -y python3 python3-pip python3-venv
    
    echo 'Installing system dependencies...'
    sudo apt install -y python3-tk libgl1-mesa-glx
    
    echo 'Creating virtual environment...'
    python3 -m venv storm_env
    
    echo 'Installing Python packages...'
    source storm_env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    echo 'Setup complete!'
"

echo.
echo Dependencies installed successfully!
echo You can now run STORM_Launcher.bat
pause