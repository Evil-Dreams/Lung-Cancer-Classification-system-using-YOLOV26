@echo off
echo ===========================================
echo  Lung Cancer Detection Application Setup
echo ===========================================
echo.

echo [1/5] Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [2/5] Creating necessary directories...
if not exist "static\uploads" mkdir "static\uploads"
if not exist "static\results" mkdir "static\results"
if not exist "checkpoints" mkdir "checkpoints"

if not exist "checkpoints\model_best.pth" (
    echo [WARNING] Model file not found in checkpoints folder.
    echo Please make sure to place 'model_best.pth' in the 'checkpoints' folder.
    pause
)

echo [3/5] Setting up virtual environment...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
)

call venv\Scripts\activate

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

echo [4/5] Installing dependencies...
echo Upgrading pip...
python -m pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Failed to upgrade pip, continuing anyway...
)

echo Installing requirements...
pip install -r requirements-web.txt --no-cache-dir
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo [5/5] Starting the application...
echo.
echo ===========================================
echo  Application is starting...
echo  Open your browser and go to: http://localhost:5000
echo  Press Ctrl+C to stop the server
echo ===========================================
echo.

python app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to start the application
    pause
    exit /b 1
)

pause
