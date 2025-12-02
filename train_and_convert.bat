@echo off
REM Complete training and conversion pipeline for 2048 AI (Windows)

echo ==================================
echo 2048 AI Training Pipeline
echo ==================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed
    exit /b 1
)

echo Step 1: Installing dependencies...
pip install -r python\requirements.txt

if errorlevel 1 (
    echo Error: Failed to install dependencies
    exit /b 1
)

echo.
echo Step 2: Training the model...
echo This will take approximately 20-40 minutes...
python python\train_dqn.py

if errorlevel 1 (
    echo Error: Training failed
    exit /b 1
)

echo.
echo Step 3: Converting model to ONNX...
python python\convert_to_onnx.py

if errorlevel 1 (
    echo Error: Model conversion failed
    exit /b 1
)

echo.
echo ==================================
echo √ Training and conversion complete!
echo ==================================
echo.
echo Next steps:
echo 1. Open index.html in your web browser
echo 2. Click 'Play AI' button or press 'A' key
echo 3. Watch your AI play 2048!
echo.

pause
