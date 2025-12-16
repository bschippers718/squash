import os

# Disable OpenCV GUI features for headless environments
# These must be set BEFORE importing cv2
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OPENCV_DISABLE_OPENCL'] = '1'

import cv2
import json
import os
from datetime import datetime
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # You can change this to yolov8s.pt for better accuracy 


# Open the video file
video_path = "video/padel.mov"
cap = cv2.VideoCapture(video_path)

# Extract video name without extension for file naming
video_name = os.path.splitext(os.path.basename(video_path))[0]

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

# Get video properties for output video
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create output directory
output_dir = "detection_results"
os.makedirs(output_dir, exist_ok=True)

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
        "processing_timestamp": datetime.now().isoformat()
    },
    "detections": []
}

frame_count = 0
print(f"Processing video: {total_frames} frames at {fps} FPS")
print(f"Output will be saved to: {output_dir}/")

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame_count += 1
        
        # Run YOLOv8 inference on the frame with custom parameters
        results = model(frame, conf=0.4, iou=0.5)  # Lower confidence, adjust IoU
        
        # Get detection results
        result = results[0]
        
        # Extract detection information
        frame_detections = {
            "frame_number": frame_count,
            "timestamp_seconds": frame_count / fps,
            "objects": []
        }
        
        # Process each detection
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                # Get class name and confidence
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detection_info = {
                    "object_id": i,
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": x2 - x1,
                        "height": y2 - y1
                    }
                }
                frame_detections["objects"].append(detection_info)
        
        # Add frame detections to overall data
        detection_data["detections"].append(frame_detections)
        
        # Visualize the results on the frame
        annotated_frame = result.plot()
        
        # Write annotated frame to output video
        out.write(annotated_frame)
        
        # Display progress
        if frame_count % 30 == 0:  # Print every 30 frames
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
        
        # Display the annotated frame in real-time
        cv2.imshow("YOLOv8 Real-Time Detection - Squash2", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Save detection data to JSON file
json_output_path = os.path.join(output_dir, f"{video_name}_detection_data.json")
with open(json_output_path, 'w') as f:
    json.dump(detection_data, f, indent=2)

# Create compact Gemini-friendly JSON file
compact_data = {
    "video_name": video_name,
    "total_frames": total_frames,
    "fps": fps,
    "duration_seconds": total_frames / fps,
    "detection_summary": {}
}

# Count detections by class for compact format
class_counts = {}
for frame_data in detection_data["detections"]:
    for obj in frame_data["objects"]:
        class_name = obj["class"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

compact_data["detection_summary"] = class_counts

# Save compact JSON
compact_json_path = os.path.join(output_dir, f"{video_name}_compact_gemini.json")
with open(compact_json_path, 'w') as f:
    json.dump(compact_data, f, indent=2)

# Create ultra-compact version for Gemini
ultra_compact_data = {
    "video": video_name,
    "duration": f"{total_frames/fps:.1f}s",
    "objects": class_counts
}

ultra_compact_path = os.path.join(output_dir, f"{video_name}_ultra_compact.json")
with open(ultra_compact_path, 'w') as f:
    json.dump(ultra_compact_data, f, indent=2)

# Create a summary text file
summary_path = os.path.join(output_dir, f"{video_name}_detection_summary.txt")
with open(summary_path, 'w') as f:
    f.write("YOLO Video Detection Summary\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Input Video: {video_path}\n")
    f.write(f"Total Frames: {total_frames}\n")
    f.write(f"FPS: {fps}\n")
    f.write(f"Duration: {total_frames/fps:.2f} seconds\n")
    f.write(f"Resolution: {width}x{height}\n\n")
    
    # Count total detections by class
    class_counts = {}
    for frame_data in detection_data["detections"]:
        for obj in frame_data["objects"]:
            class_name = obj["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    f.write("Detection Summary by Class:\n")
    f.write("-" * 30 + "\n")
    for class_name, count in sorted(class_counts.items()):
        f.write(f"{class_name}: {count} detections\n")
    
    f.write(f"\nOutput Files:\n")
    f.write(f"- Annotated Video: {output_video_path}\n")
    f.write(f"- Detection Data (JSON): {json_output_path}\n")
    f.write(f"- Compact Gemini JSON: {compact_json_path}\n")
    f.write(f"- Ultra Compact JSON: {ultra_compact_path}\n")
    f.write(f"- Summary: {summary_path}\n")

print(f"\nVideo processing finished!")
print(f"Results saved to: {output_dir}/")
print(f"- Annotated video: {output_video_path}")
print(f"- Detection data: {json_output_path}")
print(f"- Compact Gemini JSON: {compact_json_path}")
print(f"- Ultra Compact JSON: {ultra_compact_path}")
print(f"- Summary: {summary_path}")