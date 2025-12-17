# Use Python slim image with system libraries
FROM python:3.12-slim

# Install system dependencies for OpenCV
# Note: libgl1-mesa-glx was renamed to libgl1 in newer Debian versions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir flask werkzeug google-generativeai ultralytics opencv-python-headless

# Copy application code
COPY . .

# Set environment variables
ENV OPENCV_IO_ENABLE_OPENEXR=0
ENV QT_QPA_PLATFORM=offscreen
ENV OPENCV_DISABLE_OPENCL=1
ENV OPENCV_VIDEOIO_PRIORITY_LIST=FFMPEG
ENV YOLO_CONFIG_DIR=/tmp/ultralytics

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "customer_app.py"]

