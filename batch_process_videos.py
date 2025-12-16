#!/usr/bin/env python3
"""
Batch Video Processing Script for Google Cloud VM
Processes multiple videos with YOLOv8 detection and generates comprehensive results.
"""

import os

# Disable OpenCV GUI features for headless environments
# These must be set BEFORE importing cv2
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OPENCV_DISABLE_OPENCL'] = '1'

import cv2
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import torch

def check_gpu():
    """Check if GPU is available and display info"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU Available: {gpu_name} ({gpu_memory:.1f} GB)")
        return True
    else:
        print("⚠️  No GPU detected, using CPU")
        return False

def process_video(video_path, model, output_dir, use_gpu=True):
    """Process a single video file"""
    print(f"\n🎬 Processing: {os.path.basename(video_path)}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video at {video_path}")
        return None
    
    # Extract video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   📊 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer for annotated video
    output_video_path = os.path.join(output_dir, f"{video_name}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Initialize detection data storage
    detection_data = {
        "video_info": {
            "input_path": video_path,
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "processing_timestamp": datetime.now().isoformat(),
            "gpu_used": use_gpu
        },
        "detections": []
    }
    
    frame_count = 0
    start_time = time.time()
    
    print(f"   🔄 Processing {total_frames} frames...")
    
    # Process frames
    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            break
            
        frame_count += 1
        
        # Run YOLOv8 inference
        if use_gpu:
            results = model(frame, conf=0.4, iou=0.5, device=0)  # Use GPU
        else:
            results = model(frame, conf=0.4, iou=0.5, device='cpu')  # Use CPU
        
        result = results[0]
        
        # Extract detection information
        frame_detections = {
            "frame_number": frame_count,
            "timestamp_seconds": frame_count / fps,
            "objects": []
        }
        
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detection_info = {
                    "object_id": i,
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": {
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "width": x2 - x1, "height": y2 - y1
                    }
                }
                frame_detections["objects"].append(detection_info)
        
        detection_data["detections"].append(frame_detections)
        
        # Create annotated frame
        annotated_frame = result.plot()
        out.write(annotated_frame)
        
        # Progress update
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps_processed = frame_count / elapsed
            remaining = (total_frames - frame_count) / fps_processed
            print(f"   📈 Progress: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) - {fps_processed:.1f} FPS - {remaining:.1f}s remaining")
    
    # Cleanup
    cap.release()
    out.release()
    
    processing_time = time.time() - start_time
    print(f"   ✅ Completed in {processing_time:.1f}s ({frame_count/processing_time:.1f} FPS)")
    
    # Save detection data
    json_output_path = os.path.join(output_dir, f"{video_name}_detection_data.json")
    with open(json_output_path, 'w') as f:
        json.dump(detection_data, f, indent=2)
    
    # Create compact summaries
    class_counts = {}
    for frame_data in detection_data["detections"]:
        for obj in frame_data["objects"]:
            class_name = obj["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Compact JSON
    compact_data = {
        "video_name": video_name,
        "total_frames": total_frames,
        "fps": fps,
        "duration_seconds": total_frames / fps,
        "processing_time_seconds": processing_time,
        "detection_summary": class_counts
    }
    
    compact_json_path = os.path.join(output_dir, f"{video_name}_compact_gemini.json")
    with open(compact_json_path, 'w') as f:
        json.dump(compact_data, f, indent=2)
    
    # Ultra compact
    ultra_compact_data = {
        "video": video_name,
        "duration": f"{total_frames/fps:.1f}s",
        "processing_time": f"{processing_time:.1f}s",
        "objects": class_counts
    }
    
    ultra_compact_path = os.path.join(output_dir, f"{video_name}_ultra_compact.json")
    with open(ultra_compact_path, 'w') as f:
        json.dump(ultra_compact_data, f, indent=2)
    
    # Summary text
    summary_path = os.path.join(output_dir, f"{video_name}_detection_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("YOLO Video Detection Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Input Video: {video_path}\n")
        f.write(f"Total Frames: {total_frames}\n")
        f.write(f"FPS: {fps}\n")
        f.write(f"Duration: {total_frames/fps:.2f} seconds\n")
        f.write(f"Resolution: {width}x{height}\n")
        f.write(f"Processing Time: {processing_time:.2f} seconds\n")
        f.write(f"Processing Speed: {frame_count/processing_time:.2f} FPS\n")
        f.write(f"GPU Used: {use_gpu}\n\n")
        
        f.write("Detection Summary by Class:\n")
        f.write("-" * 30 + "\n")
        for class_name, count in sorted(class_counts.items()):
            f.write(f"{class_name}: {count} detections\n")
        
        f.write(f"\nOutput Files:\n")
        f.write(f"- Annotated Video: {output_video_path}\n")
        f.write(f"- Detection Data: {json_output_path}\n")
        f.write(f"- Compact JSON: {compact_json_path}\n")
        f.write(f"- Ultra Compact: {ultra_compact_path}\n")
        f.write(f"- Summary: {summary_path}\n")
    
    return {
        "video_name": video_name,
        "processing_time": processing_time,
        "total_frames": total_frames,
        "fps_processed": frame_count / processing_time,
        "detections": sum(class_counts.values())
    }

def main():
    """Main batch processing function"""
    print("🎯 YOLO Batch Video Processing")
    print("=" * 50)
    
    # Check GPU availability
    use_gpu = check_gpu()
    
    # Load YOLO model
    print("\n🤖 Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    print("✅ Model loaded successfully")
    
    # Setup directories
    video_dir = Path("~/videos").expanduser()
    output_dir = Path("~/detection_results").expanduser()
    output_dir.mkdir(exist_ok=True)
    
    # Find video files
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.m4v']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
        video_files.extend(video_dir.glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"❌ No video files found in {video_dir}")
        print("   Supported formats: .mp4, .mov, .avi, .mkv, .m4v")
        return
    
    print(f"\n📁 Found {len(video_files)} video files to process")
    
    # Process each video
    results = []
    total_start_time = time.time()
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"📹 Video {i}/{len(video_files)}")
        
        result = process_video(str(video_path), model, str(output_dir), use_gpu)
        if result:
            results.append(result)
    
    # Final summary
    total_time = time.time() - total_start_time
    total_frames = sum(r['total_frames'] for r in results)
    total_detections = sum(r['detections'] for r in results)
    
    print(f"\n🎉 Batch Processing Complete!")
    print(f"{'='*50}")
    print(f"📊 Total Videos: {len(results)}")
    print(f"⏱️  Total Time: {total_time:.1f}s")
    print(f"🎬 Total Frames: {total_frames:,}")
    print(f"🔍 Total Detections: {total_detections:,}")
    print(f"⚡ Average Speed: {total_frames/total_time:.1f} FPS")
    print(f"📁 Results saved to: {output_dir}")
    
    # Save batch summary
    batch_summary = {
        "batch_info": {
            "total_videos": len(results),
            "total_processing_time": total_time,
            "total_frames": total_frames,
            "total_detections": total_detections,
            "average_fps": total_frames / total_time,
            "gpu_used": use_gpu,
            "timestamp": datetime.now().isoformat()
        },
        "video_results": results
    }
    
    summary_path = output_dir / "batch_processing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    print(f"📋 Batch summary: {summary_path}")

if __name__ == "__main__":
    main()

