@echo off
REM Face Recognition Explainable AI - Complete Startup Script
REM This script activates the virtual environment and runs the interactive demo

echo ================================================================================
echo 🚀 FACE RECOGNITION EXPLAINABLE AI - STARTUP SCRIPT
echo ================================================================================

REM Change to project directory
cd /d "C:\Users\deepa rajesh\OneDrive\Desktop\faceauth"

echo 📁 Current Directory: %CD%

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if activation was successful
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    echo Please ensure the virtual environment exists at: .venv\Scripts\activate.bat
    pause
    exit /b 1
)

echo ✅ Virtual environment activated

REM Check Python and pip
echo 🐍 Python version:
python --version

echo 📦 Checking key packages...
python -c "import flask, torch, cv2, PIL; print('✅ Core packages available')" 2>nul
if errorlevel 1 (
    echo ⚠️  Some packages might be missing, installing requirements...
    pip install flask streamlit torch torchvision opencv-python pillow plotly matplotlib pandas
)

echo ================================================================================
echo 🚀 STARTING INTERACTIVE DEMO
echo ================================================================================
echo.
echo Choose an option:
echo 1. Full Demo (Dashboard + Study Server)
echo 2. Dashboard Only
echo 3. Study Server Only
echo 4. Test Mode
echo 5. Setup Only
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo 🚀 Starting full demo...
    python interactive_demo.py --mode full
) else if "%choice%"=="2" (
    echo 🌐 Starting dashboard only...
    python interactive_demo.py --mode dashboard
) else if "%choice%"=="3" (
    echo 📊 Starting study server only...
    python interactive_demo.py --mode study
) else if "%choice%"=="4" (
    echo 🧪 Running in test mode...
    python interactive_demo.py --test
) else if "%choice%"=="5" (
    echo 📦 Setting up demo data only...
    python interactive_demo.py --setup-only
) else (
    echo ❌ Invalid choice, starting full demo...
    python interactive_demo.py --mode full
)

echo.
echo ================================================================================
echo 🎯 ACCESS POINTS:
echo ================================================================================
echo 🌐 Main Dashboard: http://localhost:5000
echo 📊 Study Interface: http://localhost:5001
echo 📈 Streamlit (if enabled): http://localhost:8501
echo ================================================================================
echo.
echo Press any key to exit...
pause >nul