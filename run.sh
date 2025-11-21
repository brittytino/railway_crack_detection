#!/bin/bash
# Startup script for Railway Crack Detection System

echo "🚂 Railway Crack Detection System"
echo "=================================="
echo ""

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    echo "✓ Activating virtual environment..."
    source venv/bin/activate
else
    echo "✗ Virtual environment not found!"
    echo "  Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if sample data exists
if [ ! -d "data/raw/healthy" ] || [ -z "$(ls -A data/raw/healthy)" ]; then
    echo ""
    echo "⚠ No sample data found. Generating sample audio files..."
    python3 generate_sample_data.py
    echo ""
fi

# Display available commands
echo ""
echo "Available commands:"
echo "  1) streamlit run app.py          - Launch web application"
echo "  2) jupyter notebook notebooks/   - Open Jupyter notebooks"
echo "  3) python3 generate_sample_data.py - Generate sample data"
echo ""

# Ask user what to do
read -p "Select option (1-3) or press Enter to launch Streamlit: " choice

case $choice in
    1|"")
        echo ""
        echo "🚀 Launching Streamlit application..."
        echo "   Opening http://localhost:8501"
        echo ""
        streamlit run app.py
        ;;
    2)
        echo ""
        echo "📓 Starting Jupyter Notebook..."
        jupyter notebook notebooks/
        ;;
    3)
        echo ""
        echo "🔊 Generating sample audio data..."
        python3 generate_sample_data.py
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac
