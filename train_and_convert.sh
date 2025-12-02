#!/bin/bash

# Complete training and conversion pipeline for 2048 AI

echo "=================================="
echo "2048 AI Training Pipeline"
echo "=================================="
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed"
    exit 1
fi

echo "Step 1: Installing dependencies..."
pip install -r python/requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo ""
echo "Step 2: Training the model..."
echo "This will take approximately 20-40 minutes..."
python python/train_dqn.py

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo ""
echo "Step 3: Converting model to ONNX..."
python python/convert_to_onnx.py

if [ $? -ne 0 ]; then
    echo "Error: Model conversion failed"
    exit 1
fi

echo ""
echo "=================================="
echo "✓ Training and conversion complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Open index.html in your web browser"
echo "2. Click 'Play AI' button or press 'A' key"
echo "3. Watch your AI play 2048!"
echo ""
