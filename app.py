#!/usr/bin/env python3
"""
Flask web application for YOLO Video Detection Dashboard
"""

from flask import Flask, render_template, jsonify, send_from_directory
import json
import os
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

# Configuration
DETECTION_RESULTS_DIR = Path("detection_results")
VIDEO_DIR = Path("video")

def load_detection_data():
    """Load all detection results from the detection_results directory"""
    results = []
    
    if not DETECTION_RESULTS_DIR.exists():
        return results
    
    # Find all compact_gemini.json files
    for json_file in DETECTION_RESULTS_DIR.glob("*_compact_gemini.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Get video name
            video_name = data.get('video_name', json_file.stem.replace('_compact_gemini', ''))
            
            # Check for annotated video
            annotated_video = DETECTION_RESULTS_DIR / f"{video_name}_annotated.mp4"
            has_annotated_video = annotated_video.exists()
            
            # Get summary text if available
            summary_file = DETECTION_RESULTS_DIR / f"{video_name}_detection_summary.txt"
            summary_text = None
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary_text = f.read()
            
            result = {
                "video_name": video_name,
                "total_frames": data.get("total_frames", 0),
                "fps": data.get("fps", 0),
                "duration_seconds": data.get("duration_seconds", 0),
                "processing_time_seconds": data.get("processing_time_seconds", 0),
                "detection_summary": data.get("detection_summary", {}),
                "has_annotated_video": has_annotated_video,
                "summary_text": summary_text,
                "total_detections": sum(data.get("detection_summary", {}).values())
            }
            
            results.append(result)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    # Sort by video name
    results.sort(key=lambda x: x['video_name'])
    return results

def get_detection_timeline(video_name):
    """Get detection timeline data for a specific video"""
    detection_file = DETECTION_RESULTS_DIR / f"{video_name}_detection_data.json"
    
    if not detection_file.exists():
        return None
    
    try:
        with open(detection_file, 'r') as f:
            data = json.load(f)
        
        # Extract timeline data (sample every N frames for performance)
        timeline = []
        detections = data.get("detections", [])
        
        # Sample every 10th frame to reduce data size
        for i in range(0, len(detections), 10):
            frame_data = detections[i]
            frame_num = frame_data.get("frame_number", 0)
            timestamp = frame_data.get("timestamp_seconds", 0)
            objects = frame_data.get("objects", [])
            
            # Count objects by class
            class_counts = {}
            for obj in objects:
                class_name = obj.get("class", "unknown")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            timeline.append({
                "frame": frame_num,
                "timestamp": timestamp,
                "detections": class_counts,
                "total_objects": len(objects)
            })
        
        return timeline
    except Exception as e:
        print(f"Error loading timeline for {video_name}: {e}")
        return None

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/videos')
def get_videos():
    """API endpoint to get all processed videos"""
    videos = load_detection_data()
    return jsonify({
        "success": True,
        "videos": videos,
        "total": len(videos)
    })

@app.route('/api/video/<video_name>')
def get_video_details(video_name):
    """Get detailed information about a specific video"""
    videos = load_detection_data()
    video = next((v for v in videos if v['video_name'] == video_name), None)
    
    if not video:
        return jsonify({"success": False, "error": "Video not found"}), 404
    
    # Get timeline data
    timeline = get_detection_timeline(video_name)
    video['timeline'] = timeline
    
    return jsonify({
        "success": True,
        "video": video
    })

@app.route('/api/video/<video_name>/timeline')
def get_video_timeline(video_name):
    """Get timeline data for a video"""
    timeline = get_detection_timeline(video_name)
    
    if timeline is None:
        return jsonify({"success": False, "error": "Timeline not found"}), 404
    
    return jsonify({
        "success": True,
        "timeline": timeline
    })

@app.route('/videos/<path:filename>')
def serve_video(filename):
    """Serve annotated video files"""
    return send_from_directory(DETECTION_RESULTS_DIR, filename)

@app.route('/api/stats')
def get_stats():
    """Get overall statistics"""
    videos = load_detection_data()
    
    if not videos:
        return jsonify({
            "success": True,
            "stats": {
                "total_videos": 0,
                "total_frames": 0,
                "total_detections": 0,
                "total_duration": 0
            }
        })
    
    total_frames = sum(v.get('total_frames', 0) for v in videos)
    total_detections = sum(v.get('total_detections', 0) for v in videos)
    total_duration = sum(v.get('duration_seconds', 0) for v in videos)
    
    # Aggregate detection classes across all videos
    all_classes = {}
    for video in videos:
        for class_name, count in video.get('detection_summary', {}).items():
            all_classes[class_name] = all_classes.get(class_name, 0) + count
    
    return jsonify({
        "success": True,
        "stats": {
            "total_videos": len(videos),
            "total_frames": total_frames,
            "total_detections": total_detections,
            "total_duration": total_duration,
            "all_classes": all_classes
        }
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("🎬 YOLO Video Detection Dashboard")
    print("=" * 50)
    print("Starting Flask server...")
    print("Open your browser to: http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)








