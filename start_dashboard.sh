#!/bin/bash
# Start the YOLO Video Detection Dashboard

echo "🎯 Starting YOLO Video Detection Dashboard..."
echo ""

# Check if Flask is installed
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Flask not found. Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Start the Flask server
echo "🚀 Starting Flask server..."
echo "📊 Dashboard will be available at: http://localhost:5000"
echo ""
python3 app.py






