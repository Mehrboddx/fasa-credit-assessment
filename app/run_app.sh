#!/bin/bash
# Run the Fasanara Credit Assessment Streamlit App
# Usage: ./run_app.sh

echo ""
echo "========================================"
echo "Fasanara Credit Assessment App"
echo "========================================"
echo ""

# Check if venv exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: Virtual environment not found"
    echo "Please run from project root"
fi

echo ""
echo "Starting Streamlit app..."
echo "Opening browser at http://localhost:8501"
echo "Press Ctrl+C to stop"
echo ""

streamlit run app/app.py
