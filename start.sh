#!/bin/bash

# FPL Analytics App Startup Script

echo "🚀 Starting FPL Analytics App..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/Update dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.template .env
    echo "Please edit .env file with your configuration"
fi

# Create logs directory
mkdir -p logs

# Start the application
echo "🎯 Starting Streamlit app..."
streamlit run simple_app.py --server.port 8501 --server.address 0.0.0.0