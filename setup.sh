#!/bin/bash

# Setup virtual environment
echo "🔄 Setting up Python venv..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "✅ Environment is ready! Activate with: source venv/bin/activate"
