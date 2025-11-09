#!/bin/bash
# Simple script to run the DConfusion Streamlit app

echo "🚀 Starting DConfusion Web App..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null && ! python3 -c "import streamlit" &> /dev/null
then
    echo "❌ Streamlit not found. Installing dependencies..."
    pip3 install -r requirements-app.txt
    echo ""
fi

# Run the app
echo "✅ Launching app at http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

# Try streamlit command first, fall back to python3 -m streamlit
if command -v streamlit &> /dev/null
then
    streamlit run app/streamlit_app.py
else
    python3 -m streamlit run app/streamlit_app.py
fi
