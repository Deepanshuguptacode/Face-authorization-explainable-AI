# Face Recognition Explainable AI - PowerShell Startup Script
# This script activates the virtual environment and runs the interactive demo

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "🚀 FACE RECOGNITION EXPLAINABLE AI - STARTUP SCRIPT" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan

# Change to project directory
Set-Location "C:\Users\deepa rajesh\OneDrive\Desktop\faceauth"
Write-Host "📁 Current Directory: $(Get-Location)" -ForegroundColor Yellow

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow

if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .\.venv\Scripts\Activate.ps1
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "❌ Virtual environment not found at .venv\Scripts\Activate.ps1" -ForegroundColor Red
    Write-Host "Please create a virtual environment first:" -ForegroundColor Red
    Write-Host "python -m venv .venv" -ForegroundColor White
    exit 1
}

# Check Python and packages
Write-Host "🐍 Python version:" -ForegroundColor Yellow
python --version

Write-Host "📦 Checking key packages..." -ForegroundColor Yellow
try {
    python -c "import flask, torch, cv2, PIL; print('✅ Core packages available')"
} catch {
    Write-Host "⚠️  Some packages might be missing, installing requirements..." -ForegroundColor Yellow
    pip install flask streamlit torch torchvision opencv-python pillow plotly matplotlib pandas
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "🚀 STARTING INTERACTIVE DEMO" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Choose an option:" -ForegroundColor White
Write-Host "1. Full Demo (Dashboard + Study Server)" -ForegroundColor Cyan
Write-Host "2. Dashboard Only" -ForegroundColor Cyan
Write-Host "3. Study Server Only" -ForegroundColor Cyan
Write-Host "4. Test Mode" -ForegroundColor Cyan
Write-Host "5. Setup Only" -ForegroundColor Cyan
Write-Host ""

$choice = Read-Host "Enter your choice (1-5)"

switch ($choice) {
    "1" {
        Write-Host "🚀 Starting full demo..." -ForegroundColor Green
        python interactive_demo.py --mode full
    }
    "2" {
        Write-Host "🌐 Starting dashboard only..." -ForegroundColor Green
        python interactive_demo.py --mode dashboard
    }
    "3" {
        Write-Host "📊 Starting study server only..." -ForegroundColor Green
        python interactive_demo.py --mode study
    }
    "4" {
        Write-Host "🧪 Running in test mode..." -ForegroundColor Green
        python interactive_demo.py --test
    }
    "5" {
        Write-Host "📦 Setting up demo data only..." -ForegroundColor Green
        python interactive_demo.py --setup-only
    }
    default {
        Write-Host "❌ Invalid choice, starting full demo..." -ForegroundColor Yellow
        python interactive_demo.py --mode full
    }
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "🎯 ACCESS POINTS:" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "🌐 Main Dashboard: http://localhost:5000" -ForegroundColor White
Write-Host "📊 Study Interface: http://localhost:5001" -ForegroundColor White
Write-Host "📈 Streamlit (if enabled): http://localhost:8501" -ForegroundColor White
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = Read-Host