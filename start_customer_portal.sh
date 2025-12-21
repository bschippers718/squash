#!/bin/bash
# Start the Customer Video Upload Portal

echo "🎬 Starting VideoAI Customer Portal..."
echo ""

# Check if Flask is installed
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Flask not found. Installing dependencies..."
    pip3 install flask werkzeug
fi

# Create required directories
mkdir -p uploads customer_results templates/customer

# Start the Flask server
echo "🚀 Starting server..."
echo "📊 Customer portal available at: http://localhost:5001"
echo ""
echo "Customers can:"
echo "  • Upload videos via drag-and-drop"
echo "  • Track processing progress in real-time"
echo "  • Download annotated videos and analysis data"
echo ""
python3 customer_app.py







