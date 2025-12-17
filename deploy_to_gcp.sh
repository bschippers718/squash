#!/bin/bash

# Google Cloud VM Deployment Script
# This script creates and configures a VM for YOLO video processing

set -e

# Configuration
VM_NAME="yolo-video-processor"
ZONE="us-central1-a"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT="1"
DISK_SIZE="50"
IMAGE_FAMILY="ubuntu-2004-lts"
IMAGE_PROJECT="ubuntu-os-cloud"

echo "🚀 Creating Google Cloud VM for YOLO Video Processing"
echo "=================================================="

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it first:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with gcloud. Please run:"
    echo "   gcloud auth login"
    exit 1
fi

# Set default project (you may need to change this)
echo "📋 Current project: $(gcloud config get-value project)"
read -p "Is this the correct project? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please set the correct project with: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable compute.googleapis.com

# Create the VM
echo "🖥️  Creating VM instance..."
gcloud compute instances create $VM_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator=type=$GPU_TYPE,count=$GPU_COUNT \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --boot-disk-size=${DISK_SIZE}GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --restart-on-failure \
    --scopes=https://www.googleapis.com/auth/cloud-platform

echo "⏳ Waiting for VM to be ready..."
sleep 30

# Upload setup script
echo "📤 Uploading setup script..."
gcloud compute scp setup_vm.sh $VM_NAME:~/ --zone=$ZONE

# Upload processing scripts
echo "📤 Uploading processing scripts..."
gcloud compute scp batch_process_videos.py $VM_NAME:~/scripts/ --zone=$ZONE
gcloud compute scp detect_video.py $VM_NAME:~/scripts/ --zone=$ZONE

# Upload requirements
echo "📤 Uploading requirements..."
gcloud compute scp requirements.txt $VM_NAME:~/ --zone=$ZONE

# Upload YOLO model
echo "📤 Uploading YOLO model..."
gcloud compute scp yolov8n.pt $VM_NAME:~/ --zone=$ZONE

# Create videos directory
echo "📁 Creating directories..."
gcloud compute ssh $VM_NAME --zone=$ZONE --command="mkdir -p ~/videos ~/scripts ~/detection_results"

# Run setup script
echo "🔧 Running VM setup..."
gcloud compute ssh $VM_NAME --zone=$ZONE --command="chmod +x ~/setup_vm.sh && ~/setup_vm.sh"

echo ""
echo "✅ VM setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Upload your videos:"
echo "   gcloud compute scp --recurse ./video/ $VM_NAME:~/videos/ --zone=$ZONE"
echo ""
echo "2. SSH into the VM:"
echo "   gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "3. Run batch processing:"
echo "   cd ~/scripts && python batch_process_videos.py"
echo ""
echo "4. Download results:"
echo "   gcloud compute scp --recurse $VM_NAME:~/detection_results/ ./results/ --zone=$ZONE"
echo ""
echo "5. Stop the VM when done (to save costs):"
echo "   gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo ""
echo "6. Delete the VM when no longer needed:"
echo "   gcloud compute instances delete $VM_NAME --zone=$ZONE"
echo ""
echo "💰 Cost optimization tips:"
echo "- Use preemptible instances for 60-80% cost savings"
echo "- Stop the VM when not processing videos"
echo "- Consider spot instances for even more savings"

