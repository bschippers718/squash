#!/bin/bash

# Google Cloud VM Setup Script for YOLO Video Processing
# Run this script on your newly created VM

set -e

echo "🚀 Setting up VM for YOLO video processing..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.8+ and pip
echo "🐍 Installing Python and pip..."
sudo apt install -y python3 python3-pip python3-venv

# Install system dependencies for OpenCV
echo "📹 Installing OpenCV dependencies..."
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# Install CUDA toolkit (for GPU acceleration)
echo "🎮 Installing CUDA toolkit..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.1-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-0-local_12.0.1-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv yolo_env
source yolo_env/bin/activate

# Install Python dependencies
echo "📚 Installing Python packages..."
pip install --upgrade pip
pip install ultralytics
pip install opencv-python
pip install google-generativeai
pip install numpy
pip install pillow

# Download YOLOv8 model
echo "🤖 Downloading YOLOv8 model..."
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Create directories
echo "📁 Creating directories..."
mkdir -p ~/videos
mkdir -p ~/detection_results
mkdir -p ~/scripts

# Set up GPU monitoring
echo "📊 Setting up GPU monitoring..."
sudo apt install -y nvidia-smi

# Create startup script
cat > ~/start_processing.sh << 'EOF'
#!/bin/bash
cd ~/yolo_env
source bin/activate
cd ~/scripts
python batch_process_videos.py
EOF

chmod +x ~/start_processing.sh

echo "✅ VM setup complete!"
echo ""
echo "Next steps:"
echo "1. Upload your videos: gcloud compute scp --recurse ./video/ yolo-video-processor:~/videos/"
echo "2. Upload processing scripts: gcloud compute scp *.py yolo-video-processor:~/scripts/"
echo "3. SSH into VM: gcloud compute ssh yolo-video-processor"
echo "4. Run processing: ~/start_processing.sh"
echo ""
echo "To check GPU status: nvidia-smi"

