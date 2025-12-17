# Google Cloud VM Setup for Video Analysis

## Overview
This guide will help you set up a Google Cloud VM optimized for YOLOv8 video processing to expedite your video analysis pipeline.

## Recommended VM Configuration

### For Single Video Processing:
- **Machine Type**: `n1-standard-4` (4 vCPUs, 15 GB RAM)
- **GPU**: `NVIDIA T4` (optional but recommended for faster inference)
- **Disk**: 50 GB SSD persistent disk
- **OS**: Ubuntu 20.04 LTS

### For Batch Processing:
- **Machine Type**: `n1-standard-8` (8 vCPUs, 30 GB RAM) 
- **GPU**: `NVIDIA T4` or `NVIDIA V100` (for heavy processing)
- **Disk**: 100 GB SSD persistent disk
- **OS**: Ubuntu 20.04 LTS

## Cost Estimates (approximate)
- n1-standard-4 + T4 GPU: ~$0.50-0.80/hour
- n1-standard-8 + T4 GPU: ~$0.80-1.20/hour
- Remember to stop the VM when not in use!

## Setup Steps

### 1. Create the VM
```bash
# Create VM with GPU support
gcloud compute instances create yolo-video-processor \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --restart-on-failure
```

### 2. Install Dependencies
The setup script will handle this automatically, but here's what gets installed:
- Python 3.8+
- CUDA toolkit (for GPU acceleration)
- OpenCV
- Ultralytics YOLO
- All required Python packages

### 3. Upload Your Videos
```bash
# Upload videos to VM
gcloud compute scp --recurse ./video/ yolo-video-processor:~/videos/
```

### 4. Run Processing
```bash
# SSH into VM
gcloud compute ssh yolo-video-processor

# Run batch processing
python batch_process_videos.py
```

## Performance Benefits
- **GPU Acceleration**: 3-5x faster inference with T4 GPU
- **High CPU/Memory**: Process multiple videos simultaneously
- **Fast Storage**: SSD for quick video I/O
- **Scalable**: Can upgrade to more powerful instances as needed

## Cost Optimization Tips
1. **Preemptible Instances**: Use preemptible VMs for 60-80% cost savings
2. **Auto-shutdown**: Set up automatic shutdown after processing
3. **Spot Instances**: Use spot instances for even more savings
4. **Right-sizing**: Start with smaller instances and scale up if needed

