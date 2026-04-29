@echo off
REM Medical Segmentation Training Pipeline for Windows
REM Complete training setup following project guidelines

echo 🔬 Medical Segmentation Training Pipeline
echo =========================================

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Create necessary directories
echo Creating output directories...
mkdir checkpoints\2d_axi 2>nul
mkdir checkpoints\2d_cor 2>nul
mkdir checkpoints\2d_sag 2>nul
mkdir checkpoints\3d_fullres 2>nul

REM Check if processed data exists
if not exist "data\processed\Task01_BrainTumour" (
    echo ❌ Error: Processed data not found!
    echo Expected: data\processed\Task01_BrainTumour\
    echo Please run data preparation first:
    echo   python scripts\prepare.py --raw_dir .\data\raw\Task01_BrainTumour --save_dir .\data\processed
    pause
    exit /b 1
)

echo ✅ Found processed data

REM Training options
set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=config\2d_axi.py
echo Using config: %CONFIG%

REM Launch training
echo 🚀 Starting training...
echo Config: %CONFIG%
cd src
python train.py ..\%CONFIG%

echo ✅ Training completed! Check checkpoints\ directory for results.
pause