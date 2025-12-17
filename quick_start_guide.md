# Quick Start Guide: Google Cloud VM for Video Analysis

## 🚀 One-Command Setup

```bash
# Make sure you're in your project directory
cd /Users/benschippersmini/Desktop/yolo_env

# Run the deployment script
./deploy_to_gcp.sh
```

## 📋 What This Does

1. **Creates a GPU-enabled VM** with NVIDIA T4 GPU
2. **Installs all dependencies** (Python, CUDA, OpenCV, YOLO)
3. **Uploads your scripts** and model files
4. **Sets up the environment** for batch processing

## 💰 Cost Estimate
- **VM with T4 GPU**: ~$0.50-0.80/hour
- **Processing time**: Depends on video length
- **Example**: 10-minute video ≈ 2-3 minutes processing time

## 🎬 After Setup

### Upload Your Videos
```bash
gcloud compute scp --recurse ./video/ yolo-video-processor:~/videos/ --zone=us-central1-a
```

### Start Processing
```bash
gcloud compute ssh yolo-video-processor --zone=us-central1-a
cd ~/scripts
python batch_process_videos.py
```

### Download Results
```bash
gcloud compute scp --recurse yolo-video-processor:~/detection_results/ ./results/ --zone=us-central1-a
```

## 🛑 Cost Management

### Stop VM (saves money)
```bash
gcloud compute instances stop yolo-video-processor --zone=us-central1-a
```

### Start VM again
```bash
gcloud compute instances start yolo-video-processor --zone=us-central1-a
```

### Delete VM (when done)
```bash
gcloud compute instances delete yolo-video-processor --zone=us-central1-a
```

## 📊 Performance Benefits

- **3-5x faster** processing with GPU
- **Batch processing** multiple videos
- **Automatic progress tracking**
- **Comprehensive output formats**

## 🔧 Troubleshooting

### Check GPU Status
```bash
gcloud compute ssh yolo-video-processor --zone=us-central1-a --command="nvidia-smi"
```

### View Processing Logs
```bash
gcloud compute ssh yolo-video-processor --zone=us-central1-a --command="tail -f ~/processing.log"
```

### Restart Processing
```bash
gcloud compute ssh yolo-video-processor --zone=us-central1-a --command="cd ~/scripts && python batch_process_videos.py"
```

## 📁 Output Files

Each video generates:
- `*_annotated.mp4` - Video with bounding boxes
- `*_detection_data.json` - Full detection data
- `*_compact_gemini.json` - Summary for AI analysis
- `*_ultra_compact.json` - Minimal summary
- `*_detection_summary.txt` - Human-readable summary

## 🎯 Pro Tips

1. **Start small**: Test with one video first
2. **Monitor costs**: Check your GCP billing dashboard
3. **Use preemptible**: Add `--preemptible` flag for 60-80% cost savings
4. **Batch upload**: Upload all videos at once
5. **Download results**: Don't forget to download before stopping VM

