#!/usr/bin/env python3
"""
Video Processing Module
Reusable video processing functions for YOLO object detection.
"""

import cv2
import json
import os
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import Counter
from ultralytics import YOLO
import torch

# Global model instance (loaded once)
_model = None

# Color name mapping for common colors (HSV ranges)
# HSV: Hue (0-180), Saturation (0-255), Value (0-255)
COLOR_NAMES = {
    'red': ([0, 70, 50], [10, 255, 255]),
    'red2': ([160, 70, 50], [180, 255, 255]),  # Red wraps around in HSV
    'orange': ([10, 70, 50], [25, 255, 255]),
    'yellow': ([25, 70, 50], [35, 255, 255]),
    'green': ([35, 70, 50], [85, 255, 255]),
    'cyan': ([85, 70, 50], [100, 255, 255]),
    'blue': ([100, 70, 50], [130, 255, 255]),
    'purple': ([130, 30, 30], [160, 255, 255]),  # Expanded purple detection
    'violet': ([140, 30, 50], [165, 255, 200]),  # Added violet/dark purple
    'pink': ([160, 30, 50], [180, 255, 255]),
    'white': ([0, 0, 180], [180, 40, 255]),
    'black': ([0, 0, 0], [180, 255, 45]),  # Reduced upper value threshold
    'gray': ([0, 0, 45], [180, 40, 180]),
    'dark_purple': ([120, 20, 30], [160, 150, 120]),  # Dark purple/maroon
}

def get_dominant_color(image, bbox):
    """
    Extract the dominant color from a player's upper body (shirt area).
    Returns color name and hex code.
    """
    x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
    
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Focus on upper body (shirt area) - top 40% of bounding box
    height = y2 - y1
    shirt_y2 = y1 + int(height * 0.4)
    
    # Add some margin from edges
    width = x2 - x1
    margin_x = int(width * 0.15)
    
    # Extract shirt region
    shirt_region = image[y1:shirt_y2, x1+margin_x:x2-margin_x]
    
    if shirt_region.size == 0:
        return "unknown", "#808080"
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(shirt_region, cv2.COLOR_BGR2HSV)
    
    # Also get the average BGR color for analysis
    avg_bgr = cv2.mean(shirt_region)[:3]  # BGR
    avg_hsv = cv2.mean(hsv)[:3]  # HSV averages
    
    # Find dominant color by checking each color range
    color_counts = {}
    
    for color_name, (lower, upper) in COLOR_NAMES.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        count = cv2.countNonZero(mask)
        
        # Combine red ranges
        if color_name == 'red2':
            color_counts['red'] = color_counts.get('red', 0) + count
        # Combine purple variants
        elif color_name in ['violet', 'dark_purple']:
            color_counts['purple'] = color_counts.get('purple', 0) + count
        else:
            color_counts[color_name] = color_counts.get(color_name, 0) + count
    
    if not color_counts:
        return "unknown", "#808080"
    
    # Get the dominant color
    dominant = max(color_counts, key=color_counts.get)
    
    # Special handling: if "black" but has purple/blue hue, it might be dark purple
    hex_color = "#{:02x}{:02x}{:02x}".format(int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))
    
    # Check if the "black" is actually a dark color with hue
    if dominant == 'black':
        # If there's noticeable blue/red component and it's not pure black
        b, g, r = avg_bgr
        # Check if blue > green (purple-ish) or red > green (maroon-ish)
        if b > g + 10 or r > g + 10:
            # Check saturation - if there's some color saturation, it's not pure black
            if avg_hsv[1] > 20:  # Saturation > 20
                if b > r:
                    dominant = 'purple'
                else:
                    dominant = 'maroon'
        # Also check the hue if value is low but saturation exists
        if avg_hsv[1] > 15 and avg_hsv[2] > 30:
            hue = avg_hsv[0]
            if 120 <= hue <= 160:  # Purple/violet hue range
                dominant = 'purple'
            elif 100 <= hue < 120:  # Blue-purple
                dominant = 'purple'
    
    return dominant, hex_color

def detect_player_colors(video_path, detection_data, num_samples=10):
    """
    Detect the dominant shirt color for each player by sampling frames.
    Returns color info for player 1 (left) and player 2 (right).
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detections = detection_data.get('detections', [])
    
    if not detections:
        cap.release()
        return None
    
    # Sample frames evenly throughout the video
    sample_interval = max(1, len(detections) // num_samples)
    
    p1_colors = []
    p2_colors = []
    p1_hexes = []
    p2_hexes = []
    
    for i in range(0, len(detections), sample_interval):
        frame_data = detections[i]
        frame_num = frame_data.get('frame_number', i + 1)
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        success, frame = cap.read()
        
        if not success:
            continue
        
        # Get person detections
        persons = [obj for obj in frame_data.get('objects', []) 
                   if obj['class'] == 'person' and obj['confidence'] >= 0.5]
        
        if len(persons) < 2:
            continue
        
        # Sort by size (largest first)
        persons.sort(key=lambda x: (x['bbox']['x2'] - x['bbox']['x1']) * 
                                    (x['bbox']['y2'] - x['bbox']['y1']), reverse=True)
        
        # Take top 2 and sort by x position (left = p1, right = p2)
        top_players = sorted(persons[:2], key=lambda x: x['bbox']['x1'])
        
        if len(top_players) >= 2:
            # Player 1 (left)
            color1, hex1 = get_dominant_color(frame, top_players[0]['bbox'])
            p1_colors.append(color1)
            p1_hexes.append(hex1)
            
            # Player 2 (right)
            color2, hex2 = get_dominant_color(frame, top_players[1]['bbox'])
            p2_colors.append(color2)
            p2_hexes.append(hex2)
    
    cap.release()
    
    if not p1_colors or not p2_colors:
        return None
    
    # Get most common color for each player
    p1_dominant = Counter(p1_colors).most_common(1)[0][0]
    p2_dominant = Counter(p2_colors).most_common(1)[0][0]
    
    # Get average hex color
    def avg_hex(hexes):
        if not hexes:
            return "#808080"
        r = sum(int(h[1:3], 16) for h in hexes) // len(hexes)
        g = sum(int(h[3:5], 16) for h in hexes) // len(hexes)
        b = sum(int(h[5:7], 16) for h in hexes) // len(hexes)
        return "#{:02x}{:02x}{:02x}".format(r, g, b)
    
    return {
        'player1': {
            'color_name': p1_dominant,
            'color_hex': avg_hex(p1_hexes),
            'description': f"Player in {p1_dominant}"
        },
        'player2': {
            'color_name': p2_dominant,
            'color_hex': avg_hex(p2_hexes),
            'description': f"Player in {p2_dominant}"
        }
    }

def score_back_of_court_frame(persons, frame_height, frame_width):
    """
    Score a frame for back-of-court view suitability.
    
    Back-of-court view characteristics:
    - Camera is behind players looking at front wall
    - Both players visible in lower portion of frame
    - Players at similar distance from camera (similar sizes)
    - Good horizontal separation
    - Players not cut off at frame edges
    
    Returns (score, sorted_players) where players are sorted left-to-right.
    """
    if len(persons) < 2:
        return 0, None
    
    # Sort by size (largest first) and take top 2
    persons_sorted = sorted(persons, key=lambda x: (x['bbox']['x2'] - x['bbox']['x1']) * 
                                                    (x['bbox']['y2'] - x['bbox']['y1']), reverse=True)
    top_players = persons_sorted[:2]
    
    p1_bbox = top_players[0]['bbox']
    p2_bbox = top_players[1]['bbox']
    
    # Calculate player metrics
    p1_center_y = (p1_bbox['y1'] + p1_bbox['y2']) / 2
    p2_center_y = (p2_bbox['y1'] + p2_bbox['y2']) / 2
    p1_center_x = (p1_bbox['x1'] + p1_bbox['x2']) / 2
    p2_center_x = (p2_bbox['x1'] + p2_bbox['x2']) / 2
    
    p1_area = (p1_bbox['x2'] - p1_bbox['x1']) * (p1_bbox['y2'] - p1_bbox['y1'])
    p2_area = (p2_bbox['x2'] - p2_bbox['x1']) * (p2_bbox['y2'] - p2_bbox['y1'])
    
    score = 0
    
    # 1. BACK-OF-COURT INDICATOR: Both players in lower portion of frame (below 40% from top)
    # In back-of-court view, players are closer to camera so they appear in lower portion
    if p1_center_y > frame_height * 0.4 and p2_center_y > frame_height * 0.4:
        score += 50  # Strong indicator of back-of-court view
        
        # Bonus if players are in the 50-80% vertical range (ideal back-of-court position)
        if p1_center_y > frame_height * 0.5 and p2_center_y > frame_height * 0.5:
            score += 20
    else:
        # If players are in upper portion, likely not back-of-court view - penalize
        score -= 30
    
    # 2. SIMILAR PLAYER SIZES: Both players at similar distance from camera
    # Size ratio close to 1.0 indicates similar distance from camera
    if max(p1_area, p2_area) > 0:
        size_ratio = min(p1_area, p2_area) / max(p1_area, p2_area)
        if size_ratio > 0.5:
            score += 30 * size_ratio  # Up to 30 points for similar sizes
        else:
            score -= 10  # Penalize very different sizes (one player much closer)
    
    # 3. HORIZONTAL SEPARATION: Players should be spread across court width
    horizontal_separation = abs(p1_center_x - p2_center_x)
    separation_ratio = horizontal_separation / frame_width
    if separation_ratio > 0.2:  # At least 20% of frame width apart
        score += 25 * min(separation_ratio, 0.6) / 0.6  # Up to 25 points
    
    # 4. PLAYERS NOT AT EDGES: Both players fully visible, not cut off
    edge_margin = frame_width * 0.05  # 5% margin from edges
    p1_in_bounds = (p1_bbox['x1'] > edge_margin and p1_bbox['x2'] < frame_width - edge_margin)
    p2_in_bounds = (p2_bbox['x1'] > edge_margin and p2_bbox['x2'] < frame_width - edge_margin)
    if p1_in_bounds and p2_in_bounds:
        score += 15
    elif p1_in_bounds or p2_in_bounds:
        score += 5
    
    # 5. HIGH DETECTION CONFIDENCE: Prefer clearly detected players
    avg_confidence = (top_players[0]['confidence'] + top_players[1]['confidence']) / 2
    score += avg_confidence * 10  # Up to 10 points for high confidence
    
    # 6. REASONABLE PLAYER SIZE: Players should be visible but not too close
    min_area = min(p1_area, p2_area)
    max_area = max(p1_area, p2_area)
    frame_area = frame_height * frame_width
    
    # Players should be between 1% and 15% of frame area each
    if min_area > frame_area * 0.01 and max_area < frame_area * 0.15:
        score += 15
    
    # Sort players left to right for consistent labeling
    sorted_players = sorted(top_players, key=lambda x: x['bbox']['x1'])
    
    return score, sorted_players


def extract_player_identification_frame(video_path, detection_data, output_path):
    """
    Extract a clear frame showing both players for identification.
    Prioritizes back-of-court camera views where both players are visible
    facing the front wall, making them easy to distinguish.
    
    Saves an image with both players boxed and labeled "LEFT" / "RIGHT".
    
    Returns the frame number used and player positions.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open video for player frame extraction: {video_path}")
        return None
    
    # Get video dimensions for scoring
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    best_frame_num = None
    best_score = -float('inf')
    best_players = None
    
    # Also track best single-player frame as fallback
    best_single_frame_num = None
    best_single_score = -float('inf')
    best_single_players = None
    
    # Get detections - try both 'detections' and 'frames' keys for compatibility
    detections = detection_data.get('detections', detection_data.get('frames', []))
    
    if not detections:
        print(f"Warning: No detection data available for player frame extraction")
        cap.release()
        return None
    
    # Scan through ALL frames to find the best back-of-court view
    # Sample frames throughout the video for efficiency
    sample_size = min(len(detections), 2000)  # Check up to 2000 frames
    step = max(1, len(detections) // sample_size)
    
    for idx in range(0, len(detections), step):
        frame_data = detections[idx]
        frame_num = frame_data.get('frame_number', idx)
        
        # Get person detections with lower confidence threshold for better recall
        persons = [obj for obj in frame_data.get('objects', []) 
                   if obj['class'] == 'person' and obj['confidence'] >= 0.3]
        
        if len(persons) >= 2:
            # Score this frame for back-of-court suitability
            score, sorted_players = score_back_of_court_frame(persons, frame_height, frame_width)
            
            if score > best_score and sorted_players is not None:
                best_score = score
                best_frame_num = frame_num
                best_players = sorted_players
        elif len(persons) == 1 and best_single_frame_num is None:
            # Track single player frame as fallback
            best_single_frame_num = frame_num
            best_single_players = persons
    
    # Use best 2-player frame, or fall back to single player frame
    if best_frame_num is None:
        if best_single_frame_num is not None:
            print(f"Warning: No frame with 2 players found, using single player frame")
            best_frame_num = best_single_frame_num
            best_players = best_single_players
        else:
            # Last resort: just use a frame from the middle of the video
            print(f"Warning: No player detections found, using middle frame as fallback")
            best_frame_num = total_frames // 2
            best_players = []
    
    if best_frame_num is None:
        cap.release()
        return None
    
    # Seek to the best frame and extract it
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, best_frame_num - 1))
    success, frame = cap.read()
    cap.release()
    
    if not success:
        print(f"Warning: Could not read frame {best_frame_num} from video")
        return None
    
    # Draw boxes around players with labels (if any detected)
    height, width = frame.shape[:2]
    
    for i, player in enumerate(best_players[:2]):  # Limit to 2 players max
        bbox = player.get('bbox', {})
        x1 = int(bbox.get('x1', 0))
        y1 = int(bbox.get('y1', 0))
        x2 = int(bbox.get('x2', width))
        y2 = int(bbox.get('y2', height))
        
        # Colors: green for left, pink for right (BGR format)
        # Green: (0, 255, 100), Pink: (200, 100, 255) which is #ff64c8 in RGB
        color = (0, 255, 100) if i == 0 else (200, 100, 255)
        label = "LEFT" if i == 0 else "RIGHT"
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        label_y = max(y1 - 15, label_size[1] + 10)
        cv2.rectangle(frame, (x1, label_y - label_size[1] - 10), 
                     (x1 + label_size[0] + 10, label_y + 5), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1 + 5, label_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    
    # If no players detected, add a text overlay
    if len(best_players) == 0:
        cv2.putText(frame, "Players not detected - please identify manually", 
                   (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Save the frame - ALWAYS save even if no players detected
    cv2.imwrite(output_path, frame)
    print(f"Saved player identification frame to {output_path} (players detected: {len(best_players)})")
    
    # Build return data
    result = {
        'frame_number': best_frame_num,
        'image_path': output_path,
        'score': best_score,
        'players_detected': len(best_players)
    }
    
    if len(best_players) >= 1:
        result['left_player'] = {
            'bbox': best_players[0].get('bbox', {}),
            'confidence': best_players[0].get('confidence', 0)
        }
    
    if len(best_players) >= 2:
        result['right_player'] = {
            'bbox': best_players[1].get('bbox', {}),
            'confidence': best_players[1].get('confidence', 0)
        }
    
    return result


def get_model():
    """Get or load the YOLO model (singleton pattern)"""
    global _model
    if _model is None:
        model_path = Path(__file__).parent / 'yolov8n.pt'
        _model = YOLO(str(model_path))
    return _model


def validate_detection_quality(detection_data, sport='squash'):
    """
    Check if detection data is reliable enough for analytics.
    
    Args:
        detection_data: Dict with video_info and detections
        sport: Sport type for sport-specific thresholds
    
    Returns:
        Dict with quality metrics and reliability assessment
    """
    detections = detection_data.get('detections', [])
    
    if not detections:
        return {
            'ball_detection_rate': 0,
            'player_detection_rate': 0,
            'quality_score': 0,
            'is_reliable': False,
            'issues': ['No detection data available']
        }
    
    total_frames = len(detections)
    
    # Count frames with ball detected
    ball_frames = 0
    for frame in detections:
        objects = frame.get('objects', [])
        if any(obj['class'] == 'sports ball' for obj in objects):
            ball_frames += 1
    
    # Count frames with 2+ players detected
    player_frames = 0
    for frame in detections:
        objects = frame.get('objects', [])
        player_count = sum(1 for obj in objects if obj['class'] == 'person')
        if player_count >= 2:
            player_frames += 1
    
    ball_rate = ball_frames / total_frames if total_frames > 0 else 0
    player_rate = player_frames / total_frames if total_frames > 0 else 0
    
    # Calculate quality score (weighted average)
    quality_score = (ball_rate * 0.4 + player_rate * 0.6) * 100
    
    # Sport-specific thresholds
    try:
        from sport_model_config import QUALITY_THRESHOLDS
        min_ball_rate = QUALITY_THRESHOLDS.get('min_ball_detection_rate', 0.10)
        min_player_rate = QUALITY_THRESHOLDS.get('min_player_detection_rate', 0.70)
    except ImportError:
        min_ball_rate = 0.10
        min_player_rate = 0.70
    
    is_reliable = ball_rate >= min_ball_rate and player_rate >= min_player_rate
    
    # Identify issues
    issues = []
    if ball_rate < min_ball_rate:
        issues.append(f'Low ball detection rate ({ball_rate*100:.1f}% < {min_ball_rate*100:.0f}%)')
    if player_rate < min_player_rate:
        issues.append(f'Low player detection rate ({player_rate*100:.1f}% < {min_player_rate*100:.0f}%)')
    if ball_rate < 0.05:
        issues.append('Ball rarely detected - consider better video quality or lighting')
    if player_rate < 0.50:
        issues.append('Players frequently not visible - check camera angle')
    
    return {
        'ball_detection_rate': round(ball_rate * 100, 1),
        'player_detection_rate': round(player_rate * 100, 1),
        'quality_score': round(quality_score, 1),
        'is_reliable': is_reliable,
        'issues': issues,
        'frames_analyzed': total_frames,
        'ball_frames': ball_frames,
        'player_frames': player_frames
    }

def check_gpu():
    """Check if GPU is available"""
    if torch.cuda.is_available():
        return True, torch.cuda.get_device_name(0)
    return False, "CPU"

def process_video(video_path, output_dir, progress_callback=None, sport='squash'):
    """
    Process a video file with YOLO object detection.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save results
        progress_callback: Optional callback function(progress_percent, message)
        sport: Sport type for sport-specific processing ('squash', 'padel', 'tennis', 'table_tennis')
    
    Returns:
        dict with processing results or None if failed
    """
    # Load sport-specific configuration
    try:
        from sport_model_config import get_sport_config, get_detection_thresholds
        sport_config = get_sport_config(sport)
        thresholds = get_detection_thresholds(sport)
        ball_conf = thresholds.get('ball', 0.3)
        player_conf = thresholds.get('player', 0.5)
    except ImportError:
        sport_config = {'name': sport.title()}
        ball_conf = 0.3
        player_conf = 0.5
    
    model = get_model()
    use_gpu, device_name = check_gpu()
    
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return {"error": f"Could not open video at {video_path}"}
    
    # Extract video name without extension
    video_name = Path(video_path).stem
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup video writer for annotated video
    output_video_path = os.path.join(output_dir, f"{video_name}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Initialize detection data storage with sport info
    detection_data = {
        "video_info": {
            "input_path": str(video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "processing_timestamp": datetime.now().isoformat(),
            "gpu_used": use_gpu,
            "device": device_name,
            "sport": sport,
            "sport_name": sport_config.get('name', sport.title())
        },
        "detections": []
    }
    
    frame_count = 0
    start_time = time.time()
    
    # Process frames
    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            break
            
        frame_count += 1
        
        # Run YOLOv8 inference
        device = 0 if use_gpu else 'cpu'
        results = model(frame, conf=0.4, iou=0.5, device=device, verbose=False)
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
                    "confidence": round(confidence, 3),
                    "bbox": {
                        "x1": round(x1, 1), "y1": round(y1, 1),
                        "x2": round(x2, 1), "y2": round(y2, 1)
                    }
                }
                frame_detections["objects"].append(detection_info)
        
        detection_data["detections"].append(frame_detections)
        
        # Create annotated frame
        annotated_frame = result.plot()
        out.write(annotated_frame)
        
        # Progress callback
        if progress_callback and frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            elapsed = time.time() - start_time
            fps_processed = frame_count / elapsed if elapsed > 0 else 0
            remaining = (total_frames - frame_count) / fps_processed if fps_processed > 0 else 0
            progress_callback(progress, f"Processing frame {frame_count}/{total_frames} ({fps_processed:.1f} FPS, {remaining:.0f}s remaining)")
    
    # Cleanup
    cap.release()
    out.release()
    
    processing_time = time.time() - start_time
    
    # Count detections by class
    class_counts = {}
    for frame_data in detection_data["detections"]:
        for obj in frame_data["objects"]:
            class_name = obj["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Validate detection quality
    quality_metrics = validate_detection_quality(detection_data, sport)
    
    # Save full detection data
    json_output_path = os.path.join(output_dir, f"{video_name}_detection_data.json")
    with open(json_output_path, 'w') as f:
        json.dump(detection_data, f)
    
    # Save compact summary
    compact_data = {
        "video_name": video_name,
        "total_frames": total_frames,
        "fps": fps,
        "duration_seconds": total_frames / fps,
        "processing_time_seconds": round(processing_time, 2),
        "detection_summary": class_counts
    }
    
    compact_json_path = os.path.join(output_dir, f"{video_name}_compact_gemini.json")
    with open(compact_json_path, 'w') as f:
        json.dump(compact_data, f, indent=2)
    
    # Save ultra compact
    ultra_compact_data = {
        "video": video_name,
        "duration": f"{total_frames/fps:.1f}s",
        "processing_time": f"{processing_time:.1f}s",
        "objects": class_counts
    }
    
    ultra_compact_path = os.path.join(output_dir, f"{video_name}_ultra_compact.json")
    with open(ultra_compact_path, 'w') as f:
        json.dump(ultra_compact_data, f, indent=2)
    
    # Save summary text
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
        f.write(f"Device: {device_name}\n\n")
        
        f.write("Detection Summary by Class:\n")
        f.write("-" * 30 + "\n")
        for class_name, count in sorted(class_counts.items()):
            f.write(f"{class_name}: {count} detections\n")
    
    # Extract player identification frame
    player_id_frame = None
    player_id_frame_path = os.path.join(output_dir, f"{video_name}_players.jpg")
    try:
        player_id_frame = extract_player_identification_frame(video_path, detection_data, player_id_frame_path)
        if player_id_frame:
            print(f"Extracted player identification frame from frame {player_id_frame['frame_number']}")
    except Exception as e:
        print(f"Warning: Could not extract player identification frame: {e}")
        player_id_frame = None
    
    # Detect player colors (as backup/additional info)
    player_colors = None
    try:
        player_colors = detect_player_colors(video_path, detection_data)
    except Exception as e:
        print(f"Warning: Could not detect player colors: {e}")
        player_colors = None
    
    # Generate sport-specific analytics
    squash_analytics = None
    try:
        from squash_analytics import analyze_squash_match
        # Pass sport parameter for sport-specific calculations
        squash_analytics = analyze_squash_match(detection_data, sport=sport)
        
        # Add player colors to analytics
        if player_colors and squash_analytics and 'error' not in squash_analytics:
            squash_analytics['player1']['color'] = player_colors['player1']
            squash_analytics['player2']['color'] = player_colors['player2']
        
        # Save sport analytics (keep filename for compatibility)
        analytics_path = os.path.join(output_dir, f"{video_name}_squash_analytics.json")
        with open(analytics_path, 'w') as f:
            json.dump(squash_analytics, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not generate {sport} analytics: {e}")
        import traceback
        traceback.print_exc()
        squash_analytics = None
    
    return {
        "success": True,
        "video_name": video_name,
        "sport": sport,
        "sport_name": sport_config.get('name', sport.title()),
        "processing_time": round(processing_time, 2),
        "total_frames": total_frames,
        "fps": fps,
        "duration_seconds": round(total_frames / fps, 2),
        "fps_processed": round(frame_count / processing_time, 2),
        "total_detections": sum(class_counts.values()),
        "detection_summary": class_counts,
        "quality_metrics": quality_metrics,
        "squash_analytics": squash_analytics,
        "player_colors": player_colors,
        "player_id_frame": player_id_frame,
        "output_files": {
            "annotated_video": output_video_path,
            "detection_data": json_output_path,
            "compact_json": compact_json_path,
            "ultra_compact": ultra_compact_path,
            "summary": summary_path,
            "player_id_frame": player_id_frame_path if player_id_frame else None
        }
    }

