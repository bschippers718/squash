#!/usr/bin/env python3
"""
Racket Sports Analytics Module
Calculates game-specific metrics from YOLO detection data.
Supports: Squash, Padel, Tennis, Table Tennis
"""

import math
from collections import defaultdict

# Import sport-specific configurations
try:
    from sport_model_config import (
        get_sport_config, get_t_position_ratio, get_zone_boundaries,
        pixels_to_meters, pixels_to_feet, get_enabled_metrics,
        get_detection_thresholds, calculate_quality_score, is_data_reliable,
        SPORT_CONFIGS
    )
except ImportError:
    # Fallback defaults if config not available
    SPORT_CONFIGS = {
        'squash': {
            'court_width_meters': 6.4,
            'court_length_meters': 9.75,
            't_position_ratio': 0.55,
            'ball_conf_threshold': 0.3,
            'player_conf_threshold': 0.5
        }
    }
    def get_sport_config(sport='squash'):
        return SPORT_CONFIGS.get(sport, SPORT_CONFIGS['squash'])
    def get_t_position_ratio(sport='squash', camera_angle='back'):
        return SPORT_CONFIGS.get(sport, SPORT_CONFIGS['squash']).get('t_position_ratio', 0.55)

def calculate_center(bbox):
    """Calculate center point of a bounding box"""
    return (
        (bbox['x1'] + bbox['x2']) / 2,
        (bbox['y1'] + bbox['y2']) / 2
    )

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_bbox_area(bbox):
    """Calculate area of bounding box"""
    width = bbox['x2'] - bbox['x1']
    height = bbox['y2'] - bbox['y1']
    return width * height

def identify_players(detections, video_width, video_height, min_confidence=0.5):
    """
    Identify and CONSISTENTLY TRACK the two main players across all frames.
    
    Uses a hybrid approach:
    1. Find the first frame with 2 well-separated players
    2. Establish Player 1 = left player, Player 2 = right player (based on x-position)
    3. For subsequent frames, use PROXIMITY TRACKING to maintain consistent identity
       (assign each detection to the player whose last position is closest)
    
    This ensures that when players cross paths or swap sides, their stats
    remain correctly attributed to the same person throughout the match.
    """
    frames_data = []
    
    # Track last known positions for each player (for proximity-based assignment)
    p1_last_pos = None  # Player 1 (initially on left)
    p2_last_pos = None  # Player 2 (initially on right)
    identity_established = False
    
    # Maximum distance a player can move between frames (sanity check)
    # At 30fps, a player sprinting at 8m/s moves ~0.27m per frame
    # On a 1920px wide video representing ~6.4m court, that's ~80px per frame
    # We'll be generous and allow 150px to account for frame skipping
    max_movement_per_frame = video_width * 0.15  # 15% of video width
    
    for frame in detections:
        frame_num = frame.get('frame_number', 0)
        timestamp = frame.get('timestamp_seconds', 0)
        objects = frame.get('objects', [])
        
        # Filter for person detections with good confidence
        persons = [
            obj for obj in objects 
            if obj['class'] == 'person' and obj['confidence'] >= min_confidence
        ]
        
        # Sort by area (largest first) - main players are usually larger
        persons.sort(key=lambda x: get_bbox_area(x['bbox']), reverse=True)
        
        # Take top 2 persons as potential players
        candidates = persons[:2] if len(persons) >= 2 else persons
        
        # Calculate centers for candidates
        candidate_data = []
        for p in candidates:
            center = calculate_center(p['bbox'])
            candidate_data.append({
                'center': center,
                'bbox': p['bbox'],
                'confidence': p['confidence']
            })
        
        frame_data = {
            'frame': frame_num,
            'timestamp': timestamp,
            'players': []
        }
        
        if not identity_established:
            # PHASE 1: Establish player identities from first good frame
            if len(candidate_data) >= 2:
                # Sort by x-position: left = Player 1, right = Player 2
                candidate_data.sort(key=lambda p: p['center'][0])
                
                # Check if players are well-separated (at least 20% of video width apart)
                separation = abs(candidate_data[1]['center'][0] - candidate_data[0]['center'][0])
                if separation > video_width * 0.2:
                    # Good separation - establish identities
                    p1_last_pos = candidate_data[0]['center']
                    p2_last_pos = candidate_data[1]['center']
                    identity_established = True
                    
                    # Add players to frame data with consistent IDs
                    frame_data['players'] = [
                        {'id': 0, 'center': p1_last_pos, 'bbox': candidate_data[0]['bbox'], 
                         'confidence': candidate_data[0]['confidence']},
                        {'id': 1, 'center': p2_last_pos, 'bbox': candidate_data[1]['bbox'], 
                         'confidence': candidate_data[1]['confidence']}
                    ]
                else:
                    # Players too close together - just use x-position for now
                    candidate_data.sort(key=lambda p: p['center'][0])
                    for i, c in enumerate(candidate_data):
                        frame_data['players'].append({
                            'id': i, 'center': c['center'], 'bbox': c['bbox'], 
                            'confidence': c['confidence']
                        })
            elif len(candidate_data) == 1:
                # Only one player - can't establish identity yet
                frame_data['players'].append({
                    'id': 0, 'center': candidate_data[0]['center'], 
                    'bbox': candidate_data[0]['bbox'], 
                    'confidence': candidate_data[0]['confidence']
                })
        else:
            # PHASE 2: Use proximity tracking to maintain consistent identity
            if len(candidate_data) >= 2:
                # Calculate distances from each candidate to each player's last position
                c0_to_p1 = calculate_distance(candidate_data[0]['center'], p1_last_pos)
                c0_to_p2 = calculate_distance(candidate_data[0]['center'], p2_last_pos)
                c1_to_p1 = calculate_distance(candidate_data[1]['center'], p1_last_pos)
                c1_to_p2 = calculate_distance(candidate_data[1]['center'], p2_last_pos)
                
                # Determine best assignment (minimize total distance)
                # Option A: c0 -> P1, c1 -> P2
                dist_option_a = c0_to_p1 + c1_to_p2
                # Option B: c0 -> P2, c1 -> P1
                dist_option_b = c0_to_p2 + c1_to_p1
                
                if dist_option_a <= dist_option_b:
                    # Candidate 0 is Player 1, Candidate 1 is Player 2
                    p1_data = candidate_data[0]
                    p2_data = candidate_data[1]
                else:
                    # Candidate 0 is Player 2, Candidate 1 is Player 1
                    p1_data = candidate_data[1]
                    p2_data = candidate_data[0]
                
                # Sanity check: if both players would have moved too far, 
                # fall back to x-position (might be a detection glitch)
                p1_movement = calculate_distance(p1_data['center'], p1_last_pos)
                p2_movement = calculate_distance(p2_data['center'], p2_last_pos)
                
                if p1_movement > max_movement_per_frame and p2_movement > max_movement_per_frame:
                    # Both moved too far - something's wrong, use x-position as fallback
                    candidate_data.sort(key=lambda p: p['center'][0])
                    p1_data = candidate_data[0]
                    p2_data = candidate_data[1]
                
                # Update last known positions
                p1_last_pos = p1_data['center']
                p2_last_pos = p2_data['center']
                
                frame_data['players'] = [
                    {'id': 0, 'center': p1_data['center'], 'bbox': p1_data['bbox'], 
                     'confidence': p1_data['confidence']},
                    {'id': 1, 'center': p2_data['center'], 'bbox': p2_data['bbox'], 
                     'confidence': p2_data['confidence']}
                ]
                
            elif len(candidate_data) == 1:
                # Only one player detected - assign to closest last position
                c = candidate_data[0]
                dist_to_p1 = calculate_distance(c['center'], p1_last_pos)
                dist_to_p2 = calculate_distance(c['center'], p2_last_pos)
                
                if dist_to_p1 <= dist_to_p2:
                    # This is Player 1
                    p1_last_pos = c['center']
                    frame_data['players'].append({
                        'id': 0, 'center': c['center'], 'bbox': c['bbox'], 
                        'confidence': c['confidence']
                    })
                else:
                    # This is Player 2
                    p2_last_pos = c['center']
                    frame_data['players'].append({
                        'id': 1, 'center': c['center'], 'bbox': c['bbox'], 
                        'confidence': c['confidence']
                    })
            # If no candidates, keep last known positions (no update needed)
        
        # Also track ball/racket
        balls = [obj for obj in objects if obj['class'] == 'sports ball']
        rackets = [obj for obj in objects if obj['class'] in ['tennis racket', 'baseball bat']]
        
        frame_data['ball'] = balls[0]['bbox'] if balls else None
        frame_data['ball_center'] = calculate_center(balls[0]['bbox']) if balls else None
        frame_data['rackets'] = len(rackets)
        
        frames_data.append(frame_data)
    
    return frames_data

def calculate_wall_distances(ball_bbox, video_width, video_height):
    """
    Calculate distance from ball to each wall.
    Returns dict with distances to left, right, front (top), and back (bottom) walls.
    """
    if not ball_bbox:
        return None
    
    ball_center = ((ball_bbox['x1'] + ball_bbox['x2']) / 2, 
                   (ball_bbox['y1'] + ball_bbox['y2']) / 2)
    
    # Distance to each wall (in pixels)
    return {
        'left_wall': ball_center[0],  # Distance from left edge
        'right_wall': video_width - ball_center[0],  # Distance from right edge
        'front_wall': ball_center[1],  # Distance from top (front wall in typical view)
        'back_wall': video_height - ball_center[1]  # Distance from bottom
    }

def _finalize_rail_shot(shot_data, p1_rail_shots, p2_rail_shots, min_vertical_ratio):
    """
    Finalize a shot sequence and determine if it was a valid rail shot.
    A valid rail shot:
    1. Has the ball staying on one side of the court (wall_side is set)
    2. Has more vertical movement than horizontal (traveling down the wall)
    3. Has a measurable minimum distance from the wall
    """
    if not shot_data['wall_side']:
        return  # Ball crossed court, not a rail
    
    positions = shot_data['positions']
    if len(positions) < 2:
        return
    
    # Calculate total movement
    start_pos = positions[0]
    end_pos = positions[-1]
    total_dx = abs(end_pos[0] - start_pos[0])
    total_dy = abs(end_pos[1] - start_pos[1])
    
    # Check if movement is more vertical than horizontal (rail characteristic)
    if total_dx > 0 and total_dy / max(total_dx, 1) < min_vertical_ratio:
        return  # Too much horizontal movement, likely a cross-court or boast
    
    # Find minimum wall distance during the shot (the "tightness")
    wall_distances = [p[2] for p in positions]
    min_wall_dist = min(wall_distances)
    shot_length = math.sqrt(total_dx**2 + total_dy**2)
    
    # Attribute to the correct player
    if shot_data['player'] == 1:
        p1_rail_shots.append((min_wall_dist, shot_length))
    else:
        p2_rail_shots.append((min_wall_dist, shot_length))

def analyze_tight_rails(frames_data, video_width, video_height):
    """
    Analyze how tight each player's rail shots are.
    
    A RAIL SHOT is defined as:
    1. Ball traveling roughly parallel to the side wall (more vertical than horizontal movement)
    2. Ball staying within 25% of court width from the wall during the shot
    3. Ball traveling a significant distance (not just bouncing in place)
    
    For each detected rail shot, we measure the MINIMUM distance from the wall
    during that shot's trajectory - this is the "tightness" of the rail.
    
    Professional tight rails are typically 6-18 inches from the wall.
    """
    # Track detected rail shots for each player
    # Each entry is the minimum wall distance during a rail shot sequence
    p1_rail_shots = []  # List of (min_distance, shot_length) tuples
    p2_rail_shots = []
    
    # State for tracking current shot sequence
    current_shot = {
        'active': False,
        'player': None,  # 1 or 2
        'positions': [],  # List of (x, y, wall_dist) tuples
        'start_frame': 0,
        'wall_side': None  # 'left' or 'right'
    }
    
    prev_ball_center = None
    frames_since_hit = 0
    
    # Thresholds (as percentage of court width)
    RAIL_ZONE_PCT = 0.25  # Ball must be within 25% of wall to be considered a rail
    MIN_SHOT_FRAMES = 5   # Minimum frames for a valid shot sequence
    MIN_VERTICAL_RATIO = 0.5  # Vertical movement must be at least 50% of horizontal
    
    rail_zone_pixels = video_width * RAIL_ZONE_PCT
    
    for frame_idx, frame_data in enumerate(frames_data):
        ball_bbox = frame_data.get('ball')
        players = frame_data.get('players', [])
        
        if not ball_bbox or len(players) < 2:
            # End current shot if ball lost
            if current_shot['active'] and len(current_shot['positions']) >= MIN_SHOT_FRAMES:
                _finalize_rail_shot(current_shot, p1_rail_shots, p2_rail_shots, MIN_VERTICAL_RATIO)
            current_shot = {'active': False, 'player': None, 'positions': [], 'start_frame': 0, 'wall_side': None}
            prev_ball_center = None
            continue
        
        ball_center = frame_data.get('ball_center')
        if not ball_center:
            continue
        
        # Players are already consistently tracked - sort by id for consistent ordering
        players.sort(key=lambda p: p.get('id', 0))
        
        # Calculate wall distances
        left_wall_dist = ball_center[0]
        right_wall_dist = video_width - ball_center[0]
        min_wall_dist = min(left_wall_dist, right_wall_dist)
        wall_side = 'left' if left_wall_dist < right_wall_dist else 'right'
        
        # Detect hits (velocity direction change)
        hit_detected = False
        if prev_ball_center is not None:
            dx = ball_center[0] - prev_ball_center[0]
            dy = ball_center[1] - prev_ball_center[1]
            velocity_mag = math.sqrt(dx**2 + dy**2)
            
            # Significant movement indicates the ball is in play
            if velocity_mag > 5:
                frames_since_hit += 1
                
                # Check for direction reversal (hit detection)
                if frames_since_hit > 3:  # Allow some frames between hits
                    # A hit is detected when ball changes direction significantly
                    # This is a simplified check - look for x-direction reversal near a player
                    p1_dist = calculate_distance(players[0]['center'], ball_center)
                    p2_dist = calculate_distance(players[1]['center'], ball_center)
                    
                    near_player_threshold = video_width * 0.15  # Within 15% of court width
                    
                    if p1_dist < near_player_threshold or p2_dist < near_player_threshold:
                        # Ball is near a player - potential hit
                        hit_detected = True
                        hitting_player = 1 if p1_dist < p2_dist else 2
                        frames_since_hit = 0
        
        # If hit detected, finalize previous shot and start new one
        if hit_detected:
            # Finalize previous shot if it was a valid rail
            if current_shot['active'] and len(current_shot['positions']) >= MIN_SHOT_FRAMES:
                _finalize_rail_shot(current_shot, p1_rail_shots, p2_rail_shots, MIN_VERTICAL_RATIO)
            
            # Start tracking new shot
            current_shot = {
                'active': True,
                'player': hitting_player,
                'positions': [(ball_center[0], ball_center[1], min_wall_dist)],
                'start_frame': frame_idx,
                'wall_side': wall_side if min_wall_dist < rail_zone_pixels else None
            }
        elif current_shot['active']:
            # Continue tracking current shot
            current_shot['positions'].append((ball_center[0], ball_center[1], min_wall_dist))
            
            # Update wall_side if ball enters rail zone
            if min_wall_dist < rail_zone_pixels and current_shot['wall_side'] is None:
                current_shot['wall_side'] = wall_side
            
            # End shot if ball leaves rail zone significantly (went cross-court)
            if current_shot['wall_side'] is not None:
                # Check if ball crossed to opposite side
                if wall_side != current_shot['wall_side'] and min_wall_dist < rail_zone_pixels:
                    # Ball went cross-court, not a rail shot
                    current_shot['wall_side'] = None  # Mark as invalid rail
        
        prev_ball_center = ball_center
    
    # Finalize any remaining shot
    if current_shot['active'] and len(current_shot['positions']) >= MIN_SHOT_FRAMES:
        _finalize_rail_shot(current_shot, p1_rail_shots, p2_rail_shots, MIN_VERTICAL_RATIO)
    
    # Calculate statistics from detected rail shots
    def calculate_rail_stats(rail_shots):
        if not rail_shots:
            return None, 0, 0
        
        # rail_shots is list of (min_distance, shot_length) tuples
        min_distances = [shot[0] for shot in rail_shots]
        
        # Average of the minimum distances (tightest point of each rail)
        avg_tightness = sum(min_distances) / len(min_distances)
        
        # Count of "tight" rails (minimum distance < 10% of court width, ~25 inches)
        tight_threshold = video_width * 0.10
        tight_count = sum(1 for d in min_distances if d < tight_threshold)
        
        return avg_tightness, len(rail_shots), tight_count
    
    p1_avg_rail, p1_total_rails, p1_tight_count = calculate_rail_stats(p1_rail_shots)
    p2_avg_rail, p2_total_rails, p2_tight_count = calculate_rail_stats(p2_rail_shots)
    
    # Handle case where one or both players have no detected rail shots
    if p1_avg_rail is None and p2_avg_rail is None:
        return {
            'player1': {
                'tight_rail_count': 0,
                'total_rail_shots': 0,
                'tight_rail_pct': 0,
                'avg_wall_distance': -1,
                'avg_wall_distance_pct': -1,
                'avg_wall_distance_ft': -1,
                'avg_wall_distance_inches': -1
            },
            'player2': {
                'tight_rail_count': 0,
                'total_rail_shots': 0,
                'tight_rail_pct': 0,
                'avg_wall_distance': -1,
                'avg_wall_distance_pct': -1,
                'avg_wall_distance_ft': -1,
                'avg_wall_distance_inches': -1
            },
            'comparison': {
                'tighter_rails_pct': 0,
                'tight_rail_advantage': 'N/A',
                'winner': 'N/A',
                'total_tight_rails': 0,
                'total_rail_shots': 0
            },
            'analysis': {
                'winner': 'N/A',
                'summary': 'Unable to analyze rail tightness - insufficient ball tracking data to detect rail shots.'
            }
        }
    
    # Set defaults for missing data
    if p1_avg_rail is None:
        p1_avg_rail = 0
        p1_tight_count = 0
        p1_total_rails = 0
    if p2_avg_rail is None:
        p2_avg_rail = 0
        p2_tight_count = 0
        p2_total_rails = 0
    
    # Calculate COMPARATIVE metrics instead of absolute measurements
    # Compare how much tighter one player's rails are vs the other (as a percentage)
    total_tight_rails = p1_tight_count + p2_tight_count
    total_rail_shots = p1_total_rails + p2_total_rails
    
    # Calculate tight rail percentage advantage
    # If P1 has 60% of tight rails and P2 has 40%, P1 has a 20% advantage
    if total_tight_rails > 0:
        p1_tight_pct = (p1_tight_count / total_tight_rails) * 100
        p2_tight_pct = (p2_tight_count / total_tight_rails) * 100
        tight_rail_advantage_pct = abs(p1_tight_pct - p2_tight_pct)
    else:
        p1_tight_pct = 0
        p2_tight_pct = 0
        tight_rail_advantage_pct = 0
    
    # Determine winner based on tight rail count (more tight rails = better)
    if p1_tight_count > p2_tight_count:
        rail_winner = 'Player 1'
        tight_rail_advantage = 'Player 1'
    elif p2_tight_count > p1_tight_count:
        rail_winner = 'Player 2'
        tight_rail_advantage = 'Player 2'
    elif p1_avg_rail > 0 and p2_avg_rail > 0:
        # Tie on tight count - use average tightness (lower is better)
        if p1_avg_rail < p2_avg_rail:
            rail_winner = 'Player 1'
            tight_rail_advantage = 'Player 1'
        elif p2_avg_rail < p1_avg_rail:
            rail_winner = 'Player 2'
            tight_rail_advantage = 'Player 2'
        else:
            rail_winner = 'Even'
            tight_rail_advantage = 'Even'
    else:
        rail_winner = 'Even'
        tight_rail_advantage = 'Even'
    
    # Generate comparative analysis text
    if total_rail_shots == 0:
        rail_analysis = "Unable to analyze rail tightness - insufficient ball tracking data."
    elif p1_tight_count == 0 and p2_tight_count == 0:
        rail_analysis = f"Both players hit {total_rail_shots} rail shots but neither hit particularly tight rails."
    elif tight_rail_advantage == 'Even':
        rail_analysis = f"Both players hit similar tight rails ({p1_tight_count} vs {p2_tight_count} tight shots)."
    else:
        winner_tight = p1_tight_count if tight_rail_advantage == 'Player 1' else p2_tight_count
        loser_tight = p2_tight_count if tight_rail_advantage == 'Player 1' else p1_tight_count
        rail_analysis = f"{tight_rail_advantage} hit tighter rails with {tight_rail_advantage_pct:.0f}% more tight shots ({winner_tight} vs {loser_tight})."
    
    # Calculate average distances in different units for display
    # Assuming squash court width of 21 feet (252 inches / 6.4 meters)
    COURT_WIDTH_PIXELS = video_width
    COURT_WIDTH_FEET = 21.0
    COURT_WIDTH_INCHES = 252.0
    
    def calc_distance_units(avg_rail_px):
        """Convert pixel distance to percentage, feet, and inches."""
        if avg_rail_px is None or avg_rail_px <= 0:
            return -1, -1, -1, -1
        pct = round((avg_rail_px / COURT_WIDTH_PIXELS) * 100, 1)
        ft = round((avg_rail_px / COURT_WIDTH_PIXELS) * COURT_WIDTH_FEET, 1)
        inches = round((avg_rail_px / COURT_WIDTH_PIXELS) * COURT_WIDTH_INCHES, 1)
        return round(avg_rail_px, 1), pct, ft, inches
    
    p1_wall_dist, p1_wall_pct, p1_wall_ft, p1_wall_inches = calc_distance_units(p1_avg_rail)
    p2_wall_dist, p2_wall_pct, p2_wall_ft, p2_wall_inches = calc_distance_units(p2_avg_rail)
    
    return {
        'player1': {
            'tight_rail_count': p1_tight_count,
            'total_rail_shots': p1_total_rails,
            'tight_rail_pct': round(p1_tight_pct, 1),
            'avg_wall_distance': p1_wall_dist,
            'avg_wall_distance_pct': p1_wall_pct,
            'avg_wall_distance_ft': p1_wall_ft,
            'avg_wall_distance_inches': p1_wall_inches
        },
        'player2': {
            'tight_rail_count': p2_tight_count,
            'total_rail_shots': p2_total_rails,
            'tight_rail_pct': round(p2_tight_pct, 1),
            'avg_wall_distance': p2_wall_dist,
            'avg_wall_distance_pct': p2_wall_pct,
            'avg_wall_distance_ft': p2_wall_ft,
            'avg_wall_distance_inches': p2_wall_inches
        },
        'comparison': {
            'tighter_rails_pct': round(tight_rail_advantage_pct, 1),
            'tight_rail_advantage': tight_rail_advantage,
            'winner': rail_winner,
            'total_tight_rails': total_tight_rails,
            'total_rail_shots': total_rail_shots
        },
        'analysis': {
            'winner': rail_winner,
            'summary': rail_analysis
        }
    }

# =============================================================================
# ZONE DWELL TIME & HEATMAP ANALYSIS
# =============================================================================

def analyze_zone_dwell_time(frames_data, video_width, video_height):
    """
    Analyze how much time each player spends in different court zones.
    
    Court is divided into 6 zones:
    - Front Left, Front Right (attacking zones - front 30%)
    - Mid Left, Mid Right (T area - middle 40%)  
    - Back Left, Back Right (defensive zones - back 30%)
    
    Returns zone percentages and heatmap data for each player.
    """
    # Define zone boundaries
    zones = {
        'front_left': {'x': (0, 0.5), 'y': (0, 0.30)},
        'front_right': {'x': (0.5, 1.0), 'y': (0, 0.30)},
        'mid_left': {'x': (0, 0.5), 'y': (0.30, 0.70)},
        'mid_right': {'x': (0.5, 1.0), 'y': (0.30, 0.70)},
        'back_left': {'x': (0, 0.5), 'y': (0.70, 1.0)},
        'back_right': {'x': (0.5, 1.0), 'y': (0.70, 1.0)},
    }
    
    # Initialize zone counts for each player
    p1_zones = {zone: 0 for zone in zones}
    p2_zones = {zone: 0 for zone in zones}
    
    # Heatmap grid (10x10 for visualization)
    grid_size = 10
    p1_heatmap = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    p2_heatmap = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    
    p1_total_frames = 0
    p2_total_frames = 0
    
    for frame_data in frames_data:
        players = frame_data.get('players', [])
        if len(players) < 2:
            continue
        
        # Sort by x position (left = p1, right = p2)
        players.sort(key=lambda p: p['center'][0])
        
        for i, player in enumerate(players[:2]):
            center = player['center']
            # Normalize position to 0-1 range
            norm_x = center[0] / video_width
            norm_y = center[1] / video_height
            
            zone_counts = p1_zones if i == 0 else p2_zones
            heatmap = p1_heatmap if i == 0 else p2_heatmap
            
            if i == 0:
                p1_total_frames += 1
            else:
                p2_total_frames += 1
            
            # Determine which zone
            for zone_name, bounds in zones.items():
                if (bounds['x'][0] <= norm_x < bounds['x'][1] and 
                    bounds['y'][0] <= norm_y < bounds['y'][1]):
                    zone_counts[zone_name] += 1
                    break
            
            # Update heatmap
            grid_x = min(int(norm_x * grid_size), grid_size - 1)
            grid_y = min(int(norm_y * grid_size), grid_size - 1)
            heatmap[grid_y][grid_x] += 1
    
    # Convert to percentages
    def to_percentages(zone_counts, total):
        if total == 0:
            return {zone: 0 for zone in zone_counts}
        return {zone: round((count / total) * 100, 1) for zone, count in zone_counts.items()}
    
    p1_zone_pct = to_percentages(p1_zones, p1_total_frames)
    p2_zone_pct = to_percentages(p2_zones, p2_total_frames)
    
    # Normalize heatmaps to 0-100 scale
    def normalize_heatmap(heatmap):
        max_val = max(max(row) for row in heatmap) or 1
        return [[round((cell / max_val) * 100) for cell in row] for row in heatmap]
    
    p1_heatmap_norm = normalize_heatmap(p1_heatmap)
    p2_heatmap_norm = normalize_heatmap(p2_heatmap)
    
    # Calculate aggregate zones
    p1_front = p1_zone_pct['front_left'] + p1_zone_pct['front_right']
    p1_mid = p1_zone_pct['mid_left'] + p1_zone_pct['mid_right']
    p1_back = p1_zone_pct['back_left'] + p1_zone_pct['back_right']
    
    p2_front = p2_zone_pct['front_left'] + p2_zone_pct['front_right']
    p2_mid = p2_zone_pct['mid_left'] + p2_zone_pct['mid_right']
    p2_back = p2_zone_pct['back_left'] + p2_zone_pct['back_right']
    
    # Generate analysis
    analysis = []
    
    # Front court analysis
    if p1_front > p2_front + 10:
        analysis.append("Player 1 is spending significantly more time in the front court, indicating an attacking style or successful short game.")
    elif p2_front > p1_front + 10:
        analysis.append("Player 2 is spending significantly more time in the front court, indicating an attacking style or successful short game.")
    
    # Back court analysis
    if p1_back > p2_back + 15:
        analysis.append("Player 1 is being pushed to the back corners more often - they may be under defensive pressure.")
    elif p2_back > p1_back + 15:
        analysis.append("Player 2 is being pushed to the back corners more often - they may be under defensive pressure.")
    
    # T area dominance
    if p1_mid > p2_mid + 10:
        analysis.append("Player 1 is dominating the T area, controlling the center of the court.")
    elif p2_mid > p1_mid + 10:
        analysis.append("Player 2 is dominating the T area, controlling the center of the court.")
    
    if not analysis:
        analysis.append("Both players are using the court similarly, with no clear positional advantage.")
    
    return {
        'player1': {
            'zones': p1_zone_pct,
            'aggregate': {
                'front_court': round(p1_front, 1),
                'mid_court': round(p1_mid, 1),
                'back_court': round(p1_back, 1)
            },
            'heatmap': p1_heatmap_norm
        },
        'player2': {
            'zones': p2_zone_pct,
            'aggregate': {
                'front_court': round(p2_front, 1),
                'mid_court': round(p2_mid, 1),
                'back_court': round(p2_back, 1)
            },
            'heatmap': p2_heatmap_norm
        },
        'analysis': ' '.join(analysis)
    }


# =============================================================================
# FRONT COURT ANALYSIS (Dropshots & Retrieves)
# =============================================================================

def analyze_court_pressure(frames_data, video_width, video_height, fps):
    """
    Analyze court pressure - who's forcing their opponent to the back of the court more.
    
    This measures relative positioning between players:
    - When Player A is in front of Player B, Player A is applying pressure
    - The player who keeps their opponent pinned back is dominating
    - Being forced to the back court = defending, being pushed around
    
    This is more meaningful than "front court play" because it shows who's
    dictating the rally and who's being pushed around.
    
    Also tracks pressure by quarter for visualization.
    """
    # Track pressure metrics - overall
    p1_ahead_frames = 0  # Frames where P1 is closer to front wall than P2
    p2_ahead_frames = 0  # Frames where P2 is closer to front wall than P1
    p1_back_court_frames = 0  # Frames where P1 is in back 40% of court
    p2_back_court_frames = 0  # Frames where P2 is in back 40% of court
    total_frames = 0
    
    # Track pressure by quarter for graph
    total_frame_count = len(frames_data)
    quarter_size = max(1, total_frame_count // 4)
    
    # Per-quarter tracking: {quarter: {'p1_ahead': 0, 'p2_ahead': 0, 'total': 0}}
    quarters = {1: {'p1_ahead': 0, 'p2_ahead': 0, 'total': 0},
                2: {'p1_ahead': 0, 'p2_ahead': 0, 'total': 0},
                3: {'p1_ahead': 0, 'p2_ahead': 0, 'total': 0},
                4: {'p1_ahead': 0, 'p2_ahead': 0, 'total': 0}}
    
    back_court_threshold = 0.60  # Back 40% of court (y > 0.60 in normalized coords)
    position_margin = 0.05  # 5% margin to count as "meaningfully ahead"
    
    for frame_idx, frame_data in enumerate(frames_data):
        players = frame_data.get('players', [])
        
        if len(players) < 2:
            continue
        
        # Get player positions using consistent IDs
        p1_player = next((p for p in players if p['id'] == 0), None)
        p2_player = next((p for p in players if p['id'] == 1), None)
        
        if not p1_player or not p2_player:
            continue
        
        p1_center = p1_player['center']
        p2_center = p2_player['center']
        
        # Normalize Y positions (lower Y = closer to front wall)
        p1_norm_y = p1_center[1] / video_height
        p2_norm_y = p2_center[1] / video_height
        
        total_frames += 1
        
        # Determine which quarter this frame is in
        quarter = min(4, (frame_idx // quarter_size) + 1)
        quarters[quarter]['total'] += 1
        
        # Who's in front of whom? (lower Y = more forward)
        if p1_norm_y < p2_norm_y - position_margin:  # P1 is ahead by meaningful margin
            p1_ahead_frames += 1
            quarters[quarter]['p1_ahead'] += 1
        elif p2_norm_y < p1_norm_y - position_margin:  # P2 is ahead by meaningful margin
            p2_ahead_frames += 1
            quarters[quarter]['p2_ahead'] += 1
        
        # Track back court time (being pushed back)
        if p1_norm_y > back_court_threshold:
            p1_back_court_frames += 1
        if p2_norm_y > back_court_threshold:
            p2_back_court_frames += 1
    
    if total_frames == 0:
        return {
            'player1': {'pressure_pct': 0, 'back_court_pct': 0, 'pressure_by_quarter': {'q1': 0, 'q2': 0, 'q3': 0, 'q4': 0}},
            'player2': {'pressure_pct': 0, 'back_court_pct': 0, 'pressure_by_quarter': {'q1': 0, 'q2': 0, 'q3': 0, 'q4': 0}},
            'comparison': {'pressure_winner': 'N/A', 'pressure_advantage_pct': 0, 'winner': 'N/A'},
            'analysis': {'winner': 'N/A', 'summary': 'Insufficient data for court pressure analysis.'}
        }
    
    # Calculate overall percentages
    p1_pressure_pct = round((p1_ahead_frames / total_frames) * 100, 1)
    p2_pressure_pct = round((p2_ahead_frames / total_frames) * 100, 1)
    p1_back_court_pct = round((p1_back_court_frames / total_frames) * 100, 1)
    p2_back_court_pct = round((p2_back_court_frames / total_frames) * 100, 1)
    
    # Calculate pressure percentage by quarter
    def calc_quarter_pct(quarter_data, player_key):
        if quarter_data['total'] == 0:
            return 0
        return round((quarter_data[player_key] / quarter_data['total']) * 100, 1)
    
    p1_pressure_by_quarter = {
        'q1': calc_quarter_pct(quarters[1], 'p1_ahead'),
        'q2': calc_quarter_pct(quarters[2], 'p1_ahead'),
        'q3': calc_quarter_pct(quarters[3], 'p1_ahead'),
        'q4': calc_quarter_pct(quarters[4], 'p1_ahead')
    }
    p2_pressure_by_quarter = {
        'q1': calc_quarter_pct(quarters[1], 'p2_ahead'),
        'q2': calc_quarter_pct(quarters[2], 'p2_ahead'),
        'q3': calc_quarter_pct(quarters[3], 'p2_ahead'),
        'q4': calc_quarter_pct(quarters[4], 'p2_ahead')
    }
    
    # Determine who's applying more pressure
    pressure_diff = p1_pressure_pct - p2_pressure_pct
    pressure_advantage_pct = abs(pressure_diff)
    
    if pressure_diff > 5:  # P1 is ahead more often (5% threshold)
        pressure_winner = 'Player 1'
        pushed_back = 'Player 2'
    elif pressure_diff < -5:  # P2 is ahead more often
        pressure_winner = 'Player 2'
        pushed_back = 'Player 1'
    else:
        pressure_winner = 'Even'
        pushed_back = None
    
    # Generate analysis
    if pressure_winner == 'Even':
        analysis = f"Both players traded court position evenly ({p1_pressure_pct}% vs {p2_pressure_pct}% time in front). Neither consistently forced the other into the back court."
    else:
        winner_pct = p1_pressure_pct if pressure_winner == 'Player 1' else p2_pressure_pct
        loser_back_pct = p2_back_court_pct if pressure_winner == 'Player 1' else p1_back_court_pct
        analysis = f"{pressure_winner} was in front {winner_pct}% of the time, forcing {pushed_back} to defend from the back court ({loser_back_pct}% of time in back court)."
    
    return {
        'player1': {
            'pressure_pct': p1_pressure_pct,
            'back_court_pct': p1_back_court_pct,
            'pressure_by_quarter': p1_pressure_by_quarter
        },
        'player2': {
            'pressure_pct': p2_pressure_pct,
            'back_court_pct': p2_back_court_pct,
            'pressure_by_quarter': p2_pressure_by_quarter
        },
        'comparison': {
            'pressure_winner': pressure_winner,
            'pressure_advantage_pct': round(pressure_advantage_pct, 1),
            'winner': pressure_winner
        },
        'analysis': {
            'winner': pressure_winner,
            'summary': analysis
        }
    }


# =============================================================================
# FATIGUE ANALYSIS
# =============================================================================

def analyze_fatigue(frames_data, video_width, video_height, fps):
    """
    Analyze player fatigue by tracking speed decline over time.
    
    Divides the match into quarters and compares:
    - Average movement speed (early vs late)
    - Burst speed (max speed in each quarter)
    - Speed decline percentage
    - Fatigue score (0-100, higher = more fatigued)
    
    This metric relies on player detection (91%+ reliable) rather than ball detection.
    """
    if len(frames_data) < 100:
        return {
            'player1': {'fatigue_score': 0, 'speed_decline_pct': 0, 'early_speed': 0, 'late_speed': 0,
                       'early_burst': 0, 'late_burst': 0, 'burst_decline_pct': 0},
            'player2': {'fatigue_score': 0, 'speed_decline_pct': 0, 'early_speed': 0, 'late_speed': 0,
                       'early_burst': 0, 'late_burst': 0, 'burst_decline_pct': 0},
            'analysis': {'winner': 'N/A', 'summary': 'Insufficient data for fatigue analysis.'}
        }
    
    # Divide match into quarters
    total_frames = len(frames_data)
    quarter_size = total_frames // 4
    
    # Track speed per quarter for each player
    # Speed = distance moved per frame, converted to pixels/second
    p1_speeds = {1: [], 2: [], 3: [], 4: []}
    p2_speeds = {1: [], 2: [], 3: [], 4: []}
    
    prev_p1_pos = None
    prev_p2_pos = None
    
    # Minimum movement threshold to filter out stationary frames (pixels)
    min_movement = 5
    # Maximum realistic movement per frame (filters out tracking errors)
    max_movement_per_frame = video_width * 0.15  # 15% of video width
    
    for frame_idx, frame_data in enumerate(frames_data):
        quarter = min(4, (frame_idx // quarter_size) + 1)
        players = frame_data.get('players', [])
        
        if len(players) < 2:
            prev_p1_pos = None
            prev_p2_pos = None
            continue
        
        # Sort by player ID for consistency (Player 1 = id 0, Player 2 = id 1)
        players.sort(key=lambda p: p.get('id', 0))
        
        p1_center = players[0].get('center')
        p2_center = players[1].get('center')
        
        if prev_p1_pos and p1_center:
            # Calculate distance moved (pixels per frame)
            dist = math.sqrt((p1_center[0] - prev_p1_pos[0])**2 + 
                            (p1_center[1] - prev_p1_pos[1])**2)
            # Convert to speed (pixels/second)
            speed = dist * fps
            # Filter: only count actual movement, exclude tracking errors
            if dist > min_movement and dist < max_movement_per_frame:
                p1_speeds[quarter].append(speed)
        
        if prev_p2_pos and p2_center:
            dist = math.sqrt((p2_center[0] - prev_p2_pos[0])**2 + 
                            (p2_center[1] - prev_p2_pos[1])**2)
            speed = dist * fps
            if dist > min_movement and dist < max_movement_per_frame:
                p2_speeds[quarter].append(speed)
        
        prev_p1_pos = p1_center
        prev_p2_pos = p2_center
    
    # Calculate average and burst (max) speed per quarter
    def avg(lst): 
        return sum(lst) / len(lst) if lst else 0
    
    def burst(lst, percentile=95):
        """Get burst speed (95th percentile to avoid outliers)"""
        if not lst:
            return 0
        sorted_speeds = sorted(lst)
        idx = int(len(sorted_speeds) * percentile / 100)
        return sorted_speeds[min(idx, len(sorted_speeds) - 1)]
    
    # Early = Q1, Late = Q4
    p1_q1_speed = avg(p1_speeds[1])
    p1_q4_speed = avg(p1_speeds[4])
    p2_q1_speed = avg(p2_speeds[1])
    p2_q4_speed = avg(p2_speeds[4])
    
    p1_q1_burst = burst(p1_speeds[1])
    p1_q4_burst = burst(p1_speeds[4])
    p2_q1_burst = burst(p2_speeds[1])
    p2_q4_burst = burst(p2_speeds[4])
    
    # Calculate speed decline percentage (positive = slowing down)
    def calc_decline(early, late):
        if early <= 0:
            return 0
        return ((early - late) / early) * 100
    
    p1_speed_decline = calc_decline(p1_q1_speed, p1_q4_speed)
    p2_speed_decline = calc_decline(p2_q1_speed, p2_q4_speed)
    
    p1_burst_decline = calc_decline(p1_q1_burst, p1_q4_burst)
    p2_burst_decline = calc_decline(p2_q1_burst, p2_q4_burst)
    
    # Fatigue score: weighted combination of speed and burst decline
    # Capped at 0-100, higher = more fatigued
    # Weight burst decline more heavily as it shows explosive capacity loss
    p1_fatigue = max(0, min(100, (p1_speed_decline * 0.4 + p1_burst_decline * 0.6)))
    p2_fatigue = max(0, min(100, (p2_speed_decline * 0.4 + p2_burst_decline * 0.6)))
    
    # Convert pixel speeds to approximate court coverage (feet/second)
    # Squash court is ~21 feet wide, video_width pixels
    COURT_WIDTH_FT = 21.0
    pixels_per_foot = video_width / COURT_WIDTH_FT
    
    def to_ft_per_sec(pixel_speed):
        return pixel_speed / pixels_per_foot if pixels_per_foot > 0 else 0
    
    # Determine winner (lower fatigue = better conditioning)
    if p1_fatigue < p2_fatigue - 5:  # 5% threshold for meaningful difference
        winner = 'Player 1'
    elif p2_fatigue < p1_fatigue - 5:
        winner = 'Player 2'
    else:
        winner = 'Even'
    
    # Generate analysis text
    if p1_q1_speed == 0 and p2_q1_speed == 0:
        analysis_text = "Unable to analyze fatigue - insufficient movement data."
    elif winner == 'Even':
        analysis_text = f"Both players maintained similar conditioning throughout the match. Player 1 slowed {p1_speed_decline:.0f}%, Player 2 slowed {p2_speed_decline:.0f}%."
    else:
        better_player = winner
        worse_player = 'Player 2' if winner == 'Player 1' else 'Player 1'
        better_decline = p1_speed_decline if winner == 'Player 1' else p2_speed_decline
        worse_decline = p2_speed_decline if winner == 'Player 1' else p1_speed_decline
        better_burst_decline = p1_burst_decline if winner == 'Player 1' else p2_burst_decline
        worse_burst_decline = p2_burst_decline if winner == 'Player 1' else p1_burst_decline
        
        analysis_text = f"{better_player} showed better conditioning, slowing only {better_decline:.0f}% (burst: {better_burst_decline:.0f}%) compared to {worse_player}'s {worse_decline:.0f}% decline (burst: {worse_burst_decline:.0f}%). "
        
        if worse_decline > 20:
            analysis_text += f"{worse_player} may be experiencing significant fatigue in the later stages."
        elif worse_decline > 10:
            analysis_text += f"{worse_player} is showing signs of fatigue as the match progresses."
    
    return {
        'player1': {
            'fatigue_score': round(p1_fatigue, 0),
            'speed_decline_pct': round(p1_speed_decline, 1),
            'early_speed': round(to_ft_per_sec(p1_q1_speed), 1),
            'late_speed': round(to_ft_per_sec(p1_q4_speed), 1),
            'early_burst': round(to_ft_per_sec(p1_q1_burst), 1),
            'late_burst': round(to_ft_per_sec(p1_q4_burst), 1),
            'burst_decline_pct': round(p1_burst_decline, 1),
            'speeds_by_quarter': {
                'q1': round(to_ft_per_sec(p1_q1_speed), 1),
                'q2': round(to_ft_per_sec(avg(p1_speeds[2])), 1),
                'q3': round(to_ft_per_sec(avg(p1_speeds[3])), 1),
                'q4': round(to_ft_per_sec(p1_q4_speed), 1)
            }
        },
        'player2': {
            'fatigue_score': round(p2_fatigue, 0),
            'speed_decline_pct': round(p2_speed_decline, 1),
            'early_speed': round(to_ft_per_sec(p2_q1_speed), 1),
            'late_speed': round(to_ft_per_sec(p2_q4_speed), 1),
            'early_burst': round(to_ft_per_sec(p2_q1_burst), 1),
            'late_burst': round(to_ft_per_sec(p2_q4_burst), 1),
            'burst_decline_pct': round(p2_burst_decline, 1),
            'speeds_by_quarter': {
                'q1': round(to_ft_per_sec(p2_q1_speed), 1),
                'q2': round(to_ft_per_sec(avg(p2_speeds[2])), 1),
                'q3': round(to_ft_per_sec(avg(p2_speeds[3])), 1),
                'q4': round(to_ft_per_sec(p2_q4_speed), 1)
            }
        },
        'analysis': {
            'winner': winner,
            'summary': analysis_text
        }
    }


# =============================================================================
# PERFORMANCE DECAY ANALYSIS
# =============================================================================

def analyze_performance_decay(frames_data, video_width, video_height, fps):
    """
    Compare player performance between first half and second half of the match.
    
    Tracks changes in:
    - T-Dominance
    - Scramble Score (distance from T)
    - Attack Rate
    - Movement Efficiency
    
    Returns decay metrics showing if a player's performance drops over time.
    """
    if len(frames_data) < 20:
        return {'error': 'Not enough frames to analyze performance decay'}
    
    # Split frames into first half and second half
    mid_point = len(frames_data) // 2
    first_half = frames_data[:mid_point]
    second_half = frames_data[mid_point:]
    
    t_position = calculate_t_position(video_width, video_height)
    
    def calculate_half_metrics(frames, t_pos):
        """Calculate metrics for a half of the match"""
        p1_distances_from_t = []
        p2_distances_from_t = []
        p1_closer_to_t = 0
        p2_closer_to_t = 0
        p1_high_velocity = 0
        p2_high_velocity = 0
        p1_total_movement = 0
        p2_total_movement = 0
        
        prev_positions = [None, None]
        velocity_threshold = 50
        frames_with_both = 0
        
        for frame_data in frames:
            players = frame_data.get('players', [])
            if len(players) < 2:
                continue
            
            frames_with_both += 1
            players.sort(key=lambda p: p['center'][0])
            
            for i, player in enumerate(players[:2]):
                center = player['center']
                dist_from_t = calculate_distance(center, t_pos)
                
                if i == 0:
                    p1_distances_from_t.append(dist_from_t)
                else:
                    p2_distances_from_t.append(dist_from_t)
                
                if prev_positions[i] is not None:
                    movement = calculate_distance(center, prev_positions[i])
                    if i == 0:
                        p1_total_movement += movement
                        if movement > velocity_threshold:
                            p1_high_velocity += 1
                    else:
                        p2_total_movement += movement
                        if movement > velocity_threshold:
                            p2_high_velocity += 1
                
                prev_positions[i] = center
            
            # T-dominance
            p1_dist = calculate_distance(players[0]['center'], t_pos)
            p2_dist = calculate_distance(players[1]['center'], t_pos)
            if p1_dist < p2_dist:
                p1_closer_to_t += 1
            else:
                p2_closer_to_t += 1
        
        if frames_with_both == 0:
            return None
        
        return {
            'p1': {
                't_dominance': (p1_closer_to_t / frames_with_both) * 100,
                'scramble': sum(p1_distances_from_t) / len(p1_distances_from_t) if p1_distances_from_t else 0,
                'attacks': p1_high_velocity,
                'movement': p1_total_movement,
                'frames': frames_with_both
            },
            'p2': {
                't_dominance': (p2_closer_to_t / frames_with_both) * 100,
                'scramble': sum(p2_distances_from_t) / len(p2_distances_from_t) if p2_distances_from_t else 0,
                'attacks': p2_high_velocity,
                'movement': p2_total_movement,
                'frames': frames_with_both
            }
        }
    
    first_metrics = calculate_half_metrics(first_half, t_position)
    second_metrics = calculate_half_metrics(second_half, t_position)
    
    if not first_metrics or not second_metrics:
        return {'error': 'Could not calculate metrics for both halves'}
    
    def calculate_decay(first_val, second_val, lower_is_better=False):
        """Calculate % change. Negative = decline, Positive = improvement"""
        if first_val == 0:
            return 0
        change = ((second_val - first_val) / first_val) * 100
        if lower_is_better:
            change = -change  # Invert for metrics where lower is better
        return round(change, 1)
    
    # Calculate decay for each metric
    p1_decay = {
        't_dominance': calculate_decay(first_metrics['p1']['t_dominance'], second_metrics['p1']['t_dominance']),
        'scramble': calculate_decay(first_metrics['p1']['scramble'], second_metrics['p1']['scramble'], lower_is_better=True),
        'attack_rate': calculate_decay(
            first_metrics['p1']['attacks'] / max(first_metrics['p1']['frames'], 1),
            second_metrics['p1']['attacks'] / max(second_metrics['p1']['frames'], 1)
        ),
        'movement_rate': calculate_decay(
            first_metrics['p1']['movement'] / max(first_metrics['p1']['frames'], 1),
            second_metrics['p1']['movement'] / max(second_metrics['p1']['frames'], 1)
        )
    }
    
    p2_decay = {
        't_dominance': calculate_decay(first_metrics['p2']['t_dominance'], second_metrics['p2']['t_dominance']),
        'scramble': calculate_decay(first_metrics['p2']['scramble'], second_metrics['p2']['scramble'], lower_is_better=True),
        'attack_rate': calculate_decay(
            first_metrics['p2']['attacks'] / max(first_metrics['p2']['frames'], 1),
            second_metrics['p2']['attacks'] / max(second_metrics['p2']['frames'], 1)
        ),
        'movement_rate': calculate_decay(
            first_metrics['p2']['movement'] / max(first_metrics['p2']['frames'], 1),
            second_metrics['p2']['movement'] / max(second_metrics['p2']['frames'], 1)
        )
    }
    
    # Overall decay score (average of all metrics)
    p1_overall = sum(p1_decay.values()) / len(p1_decay)
    p2_overall = sum(p2_decay.values()) / len(p2_decay)
    
    # Generate analysis
    analysis = []
    
    # Check for significant decay (> 15% decline)
    if p1_overall < -15:
        analysis.append("⚠️ Player 1 shows significant performance decline in the second half - potential fatigue or mental pressure.")
    elif p1_overall > 15:
        analysis.append("📈 Player 1 is getting stronger as the match progresses - good fitness and mental resilience.")
    
    if p2_overall < -15:
        analysis.append("⚠️ Player 2 shows significant performance decline in the second half - potential fatigue or mental pressure.")
    elif p2_overall > 15:
        analysis.append("📈 Player 2 is getting stronger as the match progresses - good fitness and mental resilience.")
    
    # Specific insights
    if p1_decay['t_dominance'] < -20:
        analysis.append("Player 1 is losing T control in the second half - they're being pushed out of position more.")
    if p2_decay['t_dominance'] < -20:
        analysis.append("Player 2 is losing T control in the second half - they're being pushed out of position more.")
    
    if p1_decay['attack_rate'] < -25:
        analysis.append("Player 1's attack rate has dropped significantly - they may be playing more conservatively or lacking energy.")
    if p2_decay['attack_rate'] < -25:
        analysis.append("Player 2's attack rate has dropped significantly - they may be playing more conservatively or lacking energy.")
    
    if not analysis:
        analysis.append("Both players are maintaining consistent performance throughout the match.")
    
    return {
        'player1': {
            'first_half': {
                't_dominance': round(first_metrics['p1']['t_dominance'], 1),
                'scramble': round(first_metrics['p1']['scramble'], 1),
                'attacks': first_metrics['p1']['attacks'],
                'movement': round(first_metrics['p1']['movement'], 1)
            },
            'second_half': {
                't_dominance': round(second_metrics['p1']['t_dominance'], 1),
                'scramble': round(second_metrics['p1']['scramble'], 1),
                'attacks': second_metrics['p1']['attacks'],
                'movement': round(second_metrics['p1']['movement'], 1)
            },
            'decay': p1_decay,
            'overall_trend': round(p1_overall, 1)
        },
        'player2': {
            'first_half': {
                't_dominance': round(first_metrics['p2']['t_dominance'], 1),
                'scramble': round(first_metrics['p2']['scramble'], 1),
                'attacks': first_metrics['p2']['attacks'],
                'movement': round(first_metrics['p2']['movement'], 1)
            },
            'second_half': {
                't_dominance': round(second_metrics['p2']['t_dominance'], 1),
                'scramble': round(second_metrics['p2']['scramble'], 1),
                'attacks': second_metrics['p2']['attacks'],
                'movement': round(second_metrics['p2']['movement'], 1)
            },
            'decay': p2_decay,
            'overall_trend': round(p2_overall, 1)
        },
        'analysis': ' '.join(analysis)
    }


def calculate_t_position(video_width, video_height, sport='squash', camera_angle='back'):
    """
    Calculate the T position (center of court) based on sport and camera angle.
    
    Args:
        video_width: Width of video frame in pixels
        video_height: Height of video frame in pixels
        sport: Sport type ('squash', 'padel', 'tennis', 'table_tennis')
        camera_angle: Camera position ('back', 'side', 'front', 'overhead')
    
    Returns:
        Tuple (x, y) representing T position in pixels
    """
    # Get sport-specific T position ratio
    try:
        t_ratio = get_t_position_ratio(sport, camera_angle)
    except:
        # Fallback to default ratios
        sport_ratios = {
            'squash': 0.55,
            'padel': 0.50,
            'tennis': 0.50,
            'table_tennis': 0.50
        }
        t_ratio = sport_ratios.get(sport, 0.55)
        
        # Adjust for camera angle
        if camera_angle == 'front':
            t_ratio -= 0.1
        elif camera_angle == 'side':
            t_ratio = 0.5
    
    # Clamp ratio to valid range
    t_ratio = max(0.3, min(0.7, t_ratio))
    
    return (video_width / 2, video_height * t_ratio)

def analyze_squash_match(detection_data, sport='squash', camera_angle='back'):
    """
    Analyze racket sport match data and return comprehensive analytics.
    
    Args:
        detection_data: Dict with video_info and detections
        sport: Sport type ('squash', 'padel', 'tennis', 'table_tennis')
        camera_angle: Camera position ('back', 'side', 'front', 'overhead')
    
    Returns:
        Dict with all sport-specific analytics
    """
    video_info = detection_data.get('video_info', {})
    detections = detection_data.get('detections', [])
    
    if not detections:
        return {'error': 'No detection data available'}
    
    video_width = video_info.get('width', 1920)
    video_height = video_info.get('height', 1080)
    fps = video_info.get('fps', 30)
    total_frames = video_info.get('total_frames', len(detections))
    
    # Get sport-specific configuration
    try:
        sport_config = get_sport_config(sport)
        player_conf_threshold = sport_config.get('player_conf_threshold', 0.5)
    except:
        sport_config = {'player_conf_threshold': 0.5}
        player_conf_threshold = 0.5
    
    # Get T position based on sport and camera angle
    t_position = calculate_t_position(video_width, video_height, sport, camera_angle)
    
    # Identify players across frames with sport-specific confidence threshold
    frames_data = identify_players(detections, video_width, video_height, player_conf_threshold)
    
    # Initialize player stats
    player1_stats = {
        'distances_from_t': [],
        'positions': [],
        'confidences': [],  # Track detection confidence for weighted calculations
        'total_distance_moved': 0,
        'frames_closer_to_t': 0,
        'weighted_t_dominance': 0,  # Confidence-weighted T-dominance
        'high_velocity_shots': 0,
        'attack_with_ball_nearby': 0  # Improved attack detection
    }
    
    player2_stats = {
        'distances_from_t': [],
        'positions': [],
        'confidences': [],
        'total_distance_moved': 0,
        'frames_closer_to_t': 0,
        'weighted_t_dominance': 0,
        'high_velocity_shots': 0,
        'attack_with_ball_nearby': 0
    }
    
    prev_positions = [None, None]
    prev_ball_pos = None
    
    # Sport-aware velocity threshold (smaller courts need lower thresholds)
    base_velocity_threshold = 50
    try:
        court_width = sport_config.get('court_width_meters', 6.4)
        # Scale threshold based on court size relative to squash
        velocity_threshold = base_velocity_threshold * (court_width / 6.4)
    except:
        velocity_threshold = base_velocity_threshold
    
    frames_with_both_players = 0
    total_confidence_weight = 0
    
    for frame_data in frames_data:
        players = frame_data.get('players', [])
        ball_center = frame_data.get('ball_center')
        
        if len(players) < 2:
            continue
            
        frames_with_both_players += 1
        
        # Players are already consistently tracked by identify_players() using proximity tracking
        # Player with id=0 is always Player 1 (initially on left), id=1 is Player 2 (initially on right)
        # Sort by id to ensure consistent ordering
        players.sort(key=lambda p: p.get('id', 0))
        
        # Calculate average confidence for this frame (for weighted metrics)
        frame_confidence = sum(p.get('confidence', 0.5) for p in players[:2]) / 2
        total_confidence_weight += frame_confidence
        
        for i, player in enumerate(players[:2]):
            stats = player1_stats if i == 0 else player2_stats
            center = player['center']
            confidence = player.get('confidence', 0.5)
            
            # Track confidence for weighted calculations
            stats['confidences'].append(confidence)
            
            # Distance from T
            dist_from_t = calculate_distance(center, t_position)
            stats['distances_from_t'].append(dist_from_t)
            stats['positions'].append(center)
            
            # Calculate movement (velocity)
            if prev_positions[i] is not None:
                movement = calculate_distance(center, prev_positions[i])
                stats['total_distance_moved'] += movement
                
                # High velocity detection (potential attacking shot)
                if movement > velocity_threshold:
                    stats['high_velocity_shots'] += 1
                    
                    # Improved attack detection: check if ball is nearby
                    if ball_center is not None:
                        ball_dist = calculate_distance(center, ball_center)
                        # If ball is within reasonable striking distance
                        if ball_dist < video_width * 0.2:  # Within 20% of court width
                            stats['attack_with_ball_nearby'] += 1
            
            prev_positions[i] = center
        
        # T-Dominance: who is closer to T this frame (confidence-weighted)
        if len(players) >= 2:
            p1_dist = calculate_distance(players[0]['center'], t_position)
            p2_dist = calculate_distance(players[1]['center'], t_position)
            
            if p1_dist < p2_dist:
                player1_stats['frames_closer_to_t'] += 1
                player1_stats['weighted_t_dominance'] += frame_confidence
            else:
                player2_stats['frames_closer_to_t'] += 1
                player2_stats['weighted_t_dominance'] += frame_confidence
        
        prev_ball_pos = ball_center
    
    # Calculate final metrics
    if frames_with_both_players == 0:
        return {'error': 'Could not identify two players in video'}
    
    # Scramble Score (average distance from T, normalized by court size)
    p1_avg_dist = sum(player1_stats['distances_from_t']) / len(player1_stats['distances_from_t']) if player1_stats['distances_from_t'] else 0
    p2_avg_dist = sum(player2_stats['distances_from_t']) / len(player2_stats['distances_from_t']) if player2_stats['distances_from_t'] else 0
    
    # Running Score (total distance covered)
    p1_total_dist = player1_stats['total_distance_moved']
    p2_total_dist = player2_stats['total_distance_moved']
    
    # T-Dominance (confidence-weighted percentage of time closer to T)
    if total_confidence_weight > 0:
        p1_t_dominance = (player1_stats['weighted_t_dominance'] / total_confidence_weight) * 100
        p2_t_dominance = (player2_stats['weighted_t_dominance'] / total_confidence_weight) * 100
    else:
        p1_t_dominance = (player1_stats['frames_closer_to_t'] / frames_with_both_players) * 100
        p2_t_dominance = (player2_stats['frames_closer_to_t'] / frames_with_both_players) * 100
    
    # Attack Score - prefer ball-nearby attacks if available, else use velocity-based
    if player1_stats['attack_with_ball_nearby'] > 0 or player2_stats['attack_with_ball_nearby'] > 0:
        # Use improved attack detection
        p1_attacks = player1_stats['attack_with_ball_nearby']
        p2_attacks = player2_stats['attack_with_ball_nearby']
    else:
        # Fallback to velocity-based detection
        p1_attacks = player1_stats['high_velocity_shots']
        p2_attacks = player2_stats['high_velocity_shots']
    
    # Determine who is scrambling more
    scramble_analysis = ""
    if p1_avg_dist > p2_avg_dist * 1.3:
        scramble_analysis = "Player 1 is scrambling. They are forced significantly further from the T on average, indicating they are under pressure."
    elif p2_avg_dist > p1_avg_dist * 1.3:
        scramble_analysis = "Player 2 is scrambling. They are forced significantly further from the T on average, indicating they are under pressure."
    else:
        scramble_analysis = "Both players are maintaining similar court positions. The rally is evenly contested."
    
    # Running analysis
    running_analysis = ""
    if p1_total_dist > p2_total_dist * 1.2:
        running_analysis = "Player 1 is working harder, covering significantly more ground to stay in rallies."
    elif p2_total_dist > p1_total_dist * 1.2:
        running_analysis = "Player 2 is working harder, covering significantly more ground to stay in rallies."
    else:
        running_analysis = "Both players are covering similar distances. Physical effort is evenly matched."
    
    # T-Dominance analysis
    t_analysis = ""
    if p1_t_dominance > 55:
        t_analysis = "Player 1 is controlling the center of the court, forcing Player 2 to move around them."
    elif p2_t_dominance > 55:
        t_analysis = "Player 2 is controlling the center of the court, forcing Player 1 to move around them."
    else:
        t_analysis = "Neither player has established clear T-dominance. Court control is being contested."
    
    # Attack analysis
    attack_analysis = ""
    if p1_attacks > p2_attacks * 1.15:
        attack_analysis = "Player 1 is more aggressive, generating more pace and pressure with attacking shots."
    elif p2_attacks > p1_attacks * 1.15:
        attack_analysis = "Player 2 is more aggressive, generating more pace and pressure with attacking shots."
    else:
        attack_analysis = "Both players are showing similar levels of aggression in their shot-making."
    
    # Match duration
    duration_seconds = total_frames / fps if fps > 0 else 0
    
    # Calculate data quality metrics
    ball_frames = sum(1 for f in frames_data if f.get('ball') is not None)
    ball_detection_rate = ball_frames / len(frames_data) if frames_data else 0
    player_detection_rate = frames_with_both_players / len(frames_data) if frames_data else 0
    
    # Calculate average detection confidence
    all_confidences = player1_stats['confidences'] + player2_stats['confidences']
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    
    # Overall quality score
    quality_score = (ball_detection_rate * 0.4 + player_detection_rate * 0.6) * 100
    is_reliable = ball_detection_rate >= 0.10 and player_detection_rate >= 0.70
    
    # Analyze tight rails (only for sports with walls)
    try:
        has_walls = sport_config.get('has_walls', True)
    except:
        has_walls = True
    
    if has_walls:
        tight_rails = analyze_tight_rails(frames_data, video_width, video_height)
    else:
        tight_rails = {
            'player1': {
                'avg_wall_distance': -1,
                'avg_wall_distance_pct': -1,
                'avg_wall_distance_ft': -1,
                'avg_wall_distance_inches': -1,
                'tight_rail_count': 0,
                'tight_rail_pct': 0,
                'total_rail_shots': 0
            },
            'player2': {
                'avg_wall_distance': -1,
                'avg_wall_distance_pct': -1,
                'avg_wall_distance_ft': -1,
                'avg_wall_distance_inches': -1,
                'tight_rail_count': 0,
                'tight_rail_pct': 0,
                'total_rail_shots': 0
            },
            'comparison': {
                'tighter_rails_pct': 0,
                'tight_rail_advantage': 'N/A',
                'winner': 'N/A',
                'total_tight_rails': 0,
                'total_rail_shots': 0
            },
            'analysis': {'winner': 'N/A', 'summary': 'Wall shots not applicable for this sport.'}
        }
    
    # Analyze shot sequences for the Shot Sequencer feature
    shot_sequences = analyze_shot_sequences(frames_data, fps, video_width, video_height)
    
    # Analyze zone dwell time (heatmap data)
    zone_analysis = analyze_zone_dwell_time(frames_data, video_width, video_height)
    
    # Analyze court pressure (who's forcing opponent to back court)
    court_pressure = analyze_court_pressure(frames_data, video_width, video_height, fps)
    
    # Analyze performance decay (first half vs second half)
    performance_decay = analyze_performance_decay(frames_data, video_width, video_height, fps)
    
    # Analyze fatigue (speed decline over time)
    fatigue = analyze_fatigue(frames_data, video_width, video_height, fps)
    
    return {
        'sport': sport,
        'camera_angle': camera_angle,
        'match_info': {
            'duration_seconds': duration_seconds,
            'total_frames': total_frames,
            'frames_analyzed': frames_with_both_players,
            'fps': fps,
            'court_dimensions': {'width': video_width, 'height': video_height},
            't_position': t_position
        },
        'data_quality': {
            'ball_detection_rate': round(ball_detection_rate * 100, 1),
            'player_detection_rate': round(player_detection_rate * 100, 1),
            'avg_detection_confidence': round(avg_confidence * 100, 1),
            'quality_score': round(quality_score, 1),
            'is_reliable': is_reliable
        },
        'player1': {
            'name': 'Player 1 (Left)',
            'scramble_score': round(p1_avg_dist, 1),
            'running_score': round(p1_total_dist, 1),
            't_dominance': round(p1_t_dominance, 1),
            'attack_score': p1_attacks,
            'avg_rail_distance': tight_rails['player1']['avg_wall_distance'],
            'avg_rail_distance_pct': tight_rails['player1'].get('avg_wall_distance_pct', -1),
            'avg_rail_distance_ft': tight_rails['player1'].get('avg_wall_distance_ft', -1),
            'avg_rail_distance_inches': tight_rails['player1'].get('avg_wall_distance_inches', -1),
            'tight_rail_count': tight_rails['player1']['tight_rail_count'],
            'pressure_pct': court_pressure['player1']['pressure_pct'],
            'back_court_pct': court_pressure['player1']['back_court_pct'],
            'pressure_by_quarter': court_pressure['player1'].get('pressure_by_quarter', {'q1': 0, 'q2': 0, 'q3': 0, 'q4': 0}),
            'fatigue_score': fatigue['player1']['fatigue_score'],
            'speed_decline_pct': fatigue['player1']['speed_decline_pct'],
            'early_speed': fatigue['player1']['early_speed'],
            'late_speed': fatigue['player1']['late_speed'],
            'burst_decline_pct': fatigue['player1']['burst_decline_pct']
        },
        'player2': {
            'name': 'Player 2 (Right)',
            'scramble_score': round(p2_avg_dist, 1),
            'running_score': round(p2_total_dist, 1),
            't_dominance': round(p2_t_dominance, 1),
            'attack_score': p2_attacks,
            'avg_rail_distance': tight_rails['player2']['avg_wall_distance'],
            'avg_rail_distance_pct': tight_rails['player2'].get('avg_wall_distance_pct', -1),
            'avg_rail_distance_ft': tight_rails['player2'].get('avg_wall_distance_ft', -1),
            'avg_rail_distance_inches': tight_rails['player2'].get('avg_wall_distance_inches', -1),
            'tight_rail_count': tight_rails['player2']['tight_rail_count'],
            'pressure_pct': court_pressure['player2']['pressure_pct'],
            'back_court_pct': court_pressure['player2']['back_court_pct'],
            'pressure_by_quarter': court_pressure['player2'].get('pressure_by_quarter', {'q1': 0, 'q2': 0, 'q3': 0, 'q4': 0}),
            'fatigue_score': fatigue['player2']['fatigue_score'],
            'speed_decline_pct': fatigue['player2']['speed_decline_pct'],
            'early_speed': fatigue['player2']['early_speed'],
            'late_speed': fatigue['player2']['late_speed'],
            'burst_decline_pct': fatigue['player2']['burst_decline_pct']
        },
        'analysis': {
            'scramble': {
                'summary': scramble_analysis,
                'winner': 'Player 2' if p1_avg_dist > p2_avg_dist else 'Player 1',
                'ratio': round(max(p1_avg_dist, p2_avg_dist) / max(min(p1_avg_dist, p2_avg_dist), 1), 2)
            },
            'running': {
                'summary': running_analysis,
                'harder_worker': 'Player 1' if p1_total_dist > p2_total_dist else 'Player 2',
                'ratio': round(max(p1_total_dist, p2_total_dist) / max(min(p1_total_dist, p2_total_dist), 1), 2)
            },
            't_dominance': {
                'summary': t_analysis,
                'controller': 'Player 1' if p1_t_dominance > p2_t_dominance else 'Player 2'
            },
            'attack': {
                'summary': attack_analysis,
                'more_aggressive': 'Player 1' if p1_attacks > p2_attacks else 'Player 2'
            },
            'tight_rails': tight_rails['analysis'],
            'court_pressure': court_pressure['analysis'],
            'fatigue': fatigue['analysis']
        },
        'insights': generate_match_insights(
            p1_avg_dist, p2_avg_dist,
            p1_total_dist, p2_total_dist,
            p1_t_dominance, p2_t_dominance,
            p1_attacks, p2_attacks
        ),
        'shot_sequences': shot_sequences,
        'zone_analysis': zone_analysis,
        'court_pressure': court_pressure,
        'performance_decay': performance_decay,
        'fatigue': fatigue
    }

def generate_match_insights(p1_scramble, p2_scramble, p1_running, p2_running, 
                           p1_t_dom, p2_t_dom, p1_attacks, p2_attacks):
    """Generate overall match insights"""
    insights = []
    
    # Determine likely winner based on metrics
    p1_score = 0
    p2_score = 0
    
    # Lower scramble is better
    if p1_scramble < p2_scramble:
        p1_score += 1
    else:
        p2_score += 1
    
    # Higher T-dominance is better
    if p1_t_dom > p2_t_dom:
        p1_score += 2  # T-dominance is very important
    else:
        p2_score += 2
    
    # More attacks is generally better
    if p1_attacks > p2_attacks:
        p1_score += 1
    else:
        p2_score += 1
    
    if p1_score > p2_score:
        insights.append({
            'type': 'advantage',
            'title': 'Player 1 Advantage',
            'description': 'Based on court position and aggression metrics, Player 1 appears to have the upper hand in this match.'
        })
    elif p2_score > p1_score:
        insights.append({
            'type': 'advantage',
            'title': 'Player 2 Advantage',
            'description': 'Based on court position and aggression metrics, Player 2 appears to have the upper hand in this match.'
        })
    else:
        insights.append({
            'type': 'even',
            'title': 'Closely Contested',
            'description': 'This match is very evenly contested with neither player establishing clear dominance.'
        })
    
    # Add specific insights
    if abs(p1_t_dom - p2_t_dom) > 20:
        dominant = 'Player 1' if p1_t_dom > p2_t_dom else 'Player 2'
        insights.append({
            'type': 'tactical',
            'title': 'Court Control',
            'description': f'{dominant} is dictating play from the T position, forcing their opponent to retrieve from difficult positions.'
        })
    
    if max(p1_running, p2_running) / max(min(p1_running, p2_running), 1) > 1.5:
        runner = 'Player 1' if p1_running > p2_running else 'Player 2'
        insights.append({
            'type': 'physical',
            'title': 'Physical Demand',
            'description': f'{runner} is covering significantly more distance. This could lead to fatigue in longer matches.'
        })
    
    return insights

def format_distance(pixels, video_width=1920, sport='squash'):
    """
    Convert pixel distance to approximate meters based on sport court dimensions.
    
    Args:
        pixels: Distance in pixels
        video_width: Video frame width in pixels
        sport: Sport type for court dimension lookup
    
    Returns:
        Distance in meters (rounded to 2 decimal places)
    """
    # Get sport-specific court width
    try:
        config = get_sport_config(sport)
        court_width = config.get('court_width_meters', 6.4)
    except:
        # Fallback court widths
        court_widths = {
            'squash': 6.4,
            'padel': 10.0,
            'tennis': 10.97,
            'table_tennis': 1.525
        }
        court_width = court_widths.get(sport, 6.4)
    
    meters_per_pixel = court_width / video_width
    meters = pixels * meters_per_pixel
    return round(meters, 2)


def format_distance_feet(pixels, video_width=1920, sport='squash'):
    """
    Convert pixel distance to feet based on sport court dimensions.
    
    Args:
        pixels: Distance in pixels
        video_width: Video frame width in pixels
        sport: Sport type for court dimension lookup
    
    Returns:
        Distance in feet (rounded to 1 decimal place)
    """
    meters = format_distance(pixels, video_width, sport)
    feet = meters * 3.28084
    return round(feet, 1)


# =============================================================================
# SHOT SEQUENCER - Rally Detection and Shot Classification
# =============================================================================

def detect_rallies(frames_data, fps, video_width, video_height):
    """
    Detect individual rallies/points by identifying pauses in play.
    
    A rally ends when:
    - Both players are stationary for 1+ seconds (FPS frames)
    - Players reset to service box positions
    - Significant gap in movement activity
    
    Returns list of rally objects with start/end frames.
    """
    if not frames_data or fps <= 0:
        return []
    
    rallies = []
    current_rally_start = None
    stationary_frames = 0
    stationary_threshold = fps * 0.8  # ~0.8 seconds of no movement = rally end
    movement_threshold = 15  # pixels - below this is considered stationary
    
    prev_positions = {}
    rally_id = 0
    
    for i, frame_data in enumerate(frames_data):
        players = frame_data.get('players', [])
        frame_num = frame_data.get('frame', i)
        
        if len(players) < 2:
            continue
        
        # Calculate total movement
        total_movement = 0
        for j, player in enumerate(players[:2]):
            center = player['center']
            if j in prev_positions:
                movement = calculate_distance(center, prev_positions[j])
                total_movement += movement
            prev_positions[j] = center
        
        # Check if players are stationary
        if total_movement < movement_threshold:
            stationary_frames += 1
        else:
            # Players are moving
            if current_rally_start is None:
                # Start a new rally
                current_rally_start = frame_num
                rally_id += 1
            stationary_frames = 0
        
        # Check if rally has ended (enough stationary frames)
        if stationary_frames >= stationary_threshold and current_rally_start is not None:
            # End the current rally
            rally_end = frame_num - int(stationary_threshold)
            if rally_end > current_rally_start + fps:  # Rally must be at least 1 second
                rallies.append({
                    'rally_id': rally_id,
                    'start_frame': current_rally_start,
                    'end_frame': rally_end,
                    'duration_seconds': round((rally_end - current_rally_start) / fps, 1)
                })
            current_rally_start = None
            stationary_frames = 0
    
    # Handle last rally if video ends during play
    if current_rally_start is not None:
        last_frame = frames_data[-1].get('frame', len(frames_data))
        if last_frame > current_rally_start + fps:
            rallies.append({
                'rally_id': rally_id,
                'start_frame': current_rally_start,
                'end_frame': last_frame,
                'duration_seconds': round((last_frame - current_rally_start) / fps, 1)
            })
    
    return rallies


def get_court_zone(position, video_width, video_height):
    """
    Determine which zone of the court a position is in.
    
    Zones:
    - front_left, front_right (top 30% of frame)
    - mid_left, mid_right (middle 40% of frame)
    - back_left, back_right (bottom 30% of frame)
    """
    x, y = position
    
    # Horizontal: left or right of center
    h_zone = 'left' if x < video_width / 2 else 'right'
    
    # Vertical: front (top), mid, back (bottom)
    if y < video_height * 0.30:
        v_zone = 'front'
    elif y < video_height * 0.70:
        v_zone = 'mid'
    else:
        v_zone = 'back'
    
    return f"{v_zone}_{h_zone}"


def classify_shot_type(ball_trajectory, player_pos, prev_ball_pos, video_width, video_height):
    """
    Classify shot type based on ball movement and player position.
    
    Shot types:
    - serve: First shot of rally
    - rail: Ball stays on same side, parallel to wall
    - cross_court: Ball crosses center line diagonally
    - drop: Ball moves to front court
    - drive: Fast horizontal movement, mid-court
    - boast: Ball comes off side wall at angle
    - lob: Ball goes high to back corners
    """
    if not ball_trajectory or not prev_ball_pos:
        return 'unknown'
    
    curr_x, curr_y = ball_trajectory
    prev_x, prev_y = prev_ball_pos
    
    # Calculate movement
    dx = curr_x - prev_x
    dy = curr_y - prev_y
    distance = math.sqrt(dx**2 + dy**2)
    
    if distance < 10:  # Ball hasn't moved enough
        return 'unknown'
    
    center_x = video_width / 2
    front_line = video_height * 0.30
    
    # Determine shot type based on ball trajectory
    
    # Drop shot: Ball moves toward front of court
    if curr_y < front_line and dy < -30:
        return 'drop'
    
    # Cross court: Ball crosses center line
    crossed_center = (prev_x < center_x and curr_x > center_x) or (prev_x > center_x and curr_x < center_x)
    if crossed_center and abs(dx) > 50:
        return 'cross_court'
    
    # Rail: Ball stays on same side, moves parallel to wall
    same_side = (prev_x < center_x and curr_x < center_x) or (prev_x > center_x and curr_x > center_x)
    if same_side and abs(dy) > abs(dx) * 0.5:
        return 'rail'
    
    # Boast: Sharp angle change (ball near side wall)
    near_left_wall = curr_x < video_width * 0.15
    near_right_wall = curr_x > video_width * 0.85
    if (near_left_wall or near_right_wall) and abs(dx) > 30:
        return 'boast'
    
    # Lob: Ball moving to back corners
    if curr_y > video_height * 0.75 and dy > 30:
        return 'lob'
    
    # Drive: Fast movement in mid-court
    if distance > 50 and video_height * 0.30 < curr_y < video_height * 0.70:
        return 'drive'
    
    return 'drive'  # Default to drive


def analyze_rally_shots(frames_data, rally, fps, video_width, video_height):
    """
    Analyze shots within a single rally.
    
    Returns list of shots with type, player, zone, and frame.
    """
    shots = []
    start_frame = rally['start_frame']
    end_frame = rally['end_frame']
    
    # Filter frames for this rally
    rally_frames = [f for f in frames_data if start_frame <= f.get('frame', 0) <= end_frame]
    
    if not rally_frames:
        return shots
    
    prev_ball_pos = None
    prev_closest_player = None
    shot_cooldown = 0  # Prevent detecting same shot multiple times
    cooldown_frames = int(fps * 0.3)  # 0.3 second cooldown between shots
    
    # First shot is serve
    first_shot_added = False
    
    for frame_data in rally_frames:
        frame_num = frame_data.get('frame', 0)
        players = frame_data.get('players', [])
        ball_center = frame_data.get('ball_center')
        
        if shot_cooldown > 0:
            shot_cooldown -= 1
        
        if len(players) < 2:
            continue
        
        # Players are already consistently tracked - sort by id for consistent ordering
        players_sorted = sorted(players[:2], key=lambda p: p.get('id', 0))
        
        if ball_center:
            # Determine which player is closest to ball (likely hitting)
            p1_dist = calculate_distance(players_sorted[0]['center'], ball_center)
            p2_dist = calculate_distance(players_sorted[1]['center'], ball_center)
            
            closest_player = 1 if p1_dist < p2_dist else 2
            proximity = min(p1_dist, p2_dist)
            
            # Detect a shot when ball is close to a player and changes direction
            if proximity < 150 and shot_cooldown <= 0:
                if prev_closest_player is not None and closest_player != prev_closest_player:
                    # Ball has changed sides - a shot was made
                    shot_type = classify_shot_type(ball_center, players_sorted[prev_closest_player - 1]['center'], 
                                                   prev_ball_pos, video_width, video_height)
                    
                    player_pos = players_sorted[prev_closest_player - 1]['center']
                    zone = get_court_zone(player_pos, video_width, video_height)
                    
                    # First shot of rally is serve
                    if not first_shot_added:
                        shot_type = 'serve'
                        first_shot_added = True
                    
                    shots.append({
                        'frame': frame_num,
                        'timestamp': round(frame_num / fps, 2),
                        'player': prev_closest_player,
                        'type': shot_type,
                        'zone': zone
                    })
                    shot_cooldown = cooldown_frames
                
                prev_closest_player = closest_player
            
            prev_ball_pos = ball_center
    
    # If no shots detected, create estimated shots based on rally duration
    if not shots and rally['duration_seconds'] > 1:
        # Estimate ~1 shot per 1.5 seconds, alternating players
        estimated_shots = max(2, int(rally['duration_seconds'] / 1.5))
        shot_interval = (end_frame - start_frame) / estimated_shots
        
        for i in range(estimated_shots):
            frame = int(start_frame + i * shot_interval)
            player = (i % 2) + 1
            shot_type = 'serve' if i == 0 else ['rail', 'cross_court', 'drive'][i % 3]
            
            shots.append({
                'frame': frame,
                'timestamp': round(frame / fps, 2),
                'player': player,
                'type': shot_type,
                'zone': 'mid_left' if player == 1 else 'mid_right'
            })
    
    return shots


def count_shot_types(shots):
    """Count shots by type for each player."""
    p1_counts = defaultdict(int)
    p2_counts = defaultdict(int)
    
    for shot in shots:
        player = shot['player']
        shot_type = shot['type']
        
        if player == 1:
            p1_counts[shot_type] += 1
        else:
            p2_counts[shot_type] += 1
    
    return dict(p1_counts), dict(p2_counts)


def analyze_shot_sequences(frames_data, fps, video_width, video_height):
    """
    Main function to analyze shot sequences in a match.
    
    Returns complete rally and shot data for the Shot Sequencer feature.
    """
    # Detect rallies
    rallies = detect_rallies(frames_data, fps, video_width, video_height)
    
    if not rallies:
        # If no rallies detected, treat entire match as one rally
        if frames_data:
            rallies = [{
                'rally_id': 1,
                'start_frame': frames_data[0].get('frame', 0),
                'end_frame': frames_data[-1].get('frame', len(frames_data)),
                'duration_seconds': round(len(frames_data) / fps, 1)
            }]
    
    # Analyze shots in each rally
    for rally in rallies:
        shots = analyze_rally_shots(frames_data, rally, fps, video_width, video_height)
        rally['shots'] = shots
        rally['shot_count'] = len(shots)
        
        # Count shot types per player
        p1_counts, p2_counts = count_shot_types(shots)
        rally['player1_shots'] = p1_counts
        rally['player2_shots'] = p2_counts
        
        # Determine rally winner (player who hit last shot loses, or estimate from T-dominance)
        if shots:
            # Last shot indicates who made an error (simplified assumption)
            last_shot_player = shots[-1]['player']
            rally['winner'] = 2 if last_shot_player == 1 else 1
        else:
            rally['winner'] = None
    
    # Aggregate statistics
    total_shots = sum(r['shot_count'] for r in rallies)
    shot_type_totals = defaultdict(lambda: {'p1': 0, 'p2': 0})
    
    for rally in rallies:
        for shot in rally.get('shots', []):
            shot_type = shot['type']
            player = shot['player']
            if player == 1:
                shot_type_totals[shot_type]['p1'] += 1
            else:
                shot_type_totals[shot_type]['p2'] += 1
    
    return {
        'rallies': rallies,
        'rally_count': len(rallies),
        'total_shots': total_shots,
        'avg_rally_length': round(total_shots / len(rallies), 1) if rallies else 0,
        'avg_rally_duration': round(sum(r['duration_seconds'] for r in rallies) / len(rallies), 1) if rallies else 0,
        'shot_type_summary': dict(shot_type_totals)
    }


# =============================================================================
# MULTI-GAME MATCH COMBINATION
# =============================================================================

def combine_match_analytics(game_analytics_list, game_names=None):
    """
    Combine multiple game analyses into a single match analysis.
    
    Args:
        game_analytics_list: List of squash_analytics dicts (one per game)
        game_names: Optional list of names for each game (e.g., ["Game 1", "Game 2", "Game 3"])
    
    Returns:
        Combined match analytics dict with per-game breakdowns
    """
    if not game_analytics_list:
        return {'error': 'No game analytics provided'}
    
    num_games = len(game_analytics_list)
    if game_names is None:
        game_names = [f"Game {i+1}" for i in range(num_games)]
    
    # Initialize aggregated stats
    total_duration = 0
    total_frames = 0
    total_frames_analyzed = 0
    
    # Player aggregates
    p1_total_scramble = 0
    p2_total_scramble = 0
    p1_total_running = 0
    p2_total_running = 0
    p1_total_t_dominance = 0
    p2_total_t_dominance = 0
    p1_total_attacks = 0
    p2_total_attacks = 0
    p1_total_rail_dist = 0
    p2_total_rail_dist = 0
    p1_rail_games = 0  # Count games with valid rail data
    p2_rail_games = 0
    p1_total_tight_rails = 0
    p2_total_tight_rails = 0
    
    # Per-game stats for breakdown
    games = []
    
    # Zone analysis aggregates
    p1_zone_totals = {'front_court': 0, 'mid_court': 0, 'back_court': 0}
    p2_zone_totals = {'front_court': 0, 'mid_court': 0, 'back_court': 0}
    
    # Court pressure aggregates
    p1_total_pressure_pct = 0
    p2_total_pressure_pct = 0
    p1_total_back_court_pct = 0
    p2_total_back_court_pct = 0
    
    # Fatigue aggregates (weighted by duration)
    p1_total_fatigue = 0
    p2_total_fatigue = 0
    p1_total_speed_decline = 0
    p2_total_speed_decline = 0
    p1_total_early_speed = 0
    p2_total_early_speed = 0
    p1_total_late_speed = 0
    p2_total_late_speed = 0
    p1_total_burst_decline = 0
    p2_total_burst_decline = 0
    p1_fatigue_games = 0
    p2_fatigue_games = 0
    
    # Performance decay - compare first half of games to second half
    first_half_games = []
    second_half_games = []
    
    # Performance trends across games
    p1_t_dom_trend = []
    p2_t_dom_trend = []
    p1_attack_trend = []
    p2_attack_trend = []
    p1_scramble_trend = []
    p2_scramble_trend = []
    
    # Store first game's job_id for player identification frame
    first_game_job_id = None
    first_game_video_name = None
    
    for i, game in enumerate(game_analytics_list):
        match_info = game.get('match_info', {})
        p1 = game.get('player1', {})
        p2 = game.get('player2', {})
        analysis = game.get('analysis', {})
        zone = game.get('zone_analysis', {})
        
        # Aggregate match info
        game_duration = match_info.get('duration_seconds', 0)
        total_duration += game_duration
        total_frames += match_info.get('total_frames', 0)
        total_frames_analyzed += match_info.get('frames_analyzed', 0)
        
        # Aggregate player stats (weighted by duration for averages)
        p1_scramble = p1.get('scramble_score', 0)
        p2_scramble = p2.get('scramble_score', 0)
        p1_total_scramble += p1_scramble * game_duration
        p2_total_scramble += p2_scramble * game_duration
        
        p1_running = p1.get('running_score', 0)
        p2_running = p2.get('running_score', 0)
        p1_total_running += p1_running
        p2_total_running += p2_running
        
        p1_t_dom = p1.get('t_dominance', 0)
        p2_t_dom = p2.get('t_dominance', 0)
        p1_total_t_dominance += p1_t_dom * game_duration
        p2_total_t_dominance += p2_t_dom * game_duration
        
        p1_attacks = p1.get('attack_score', 0)
        p2_attacks = p2.get('attack_score', 0)
        p1_total_attacks += p1_attacks
        p2_total_attacks += p2_attacks
        
        # Rail distances (only count if valid)
        p1_rail = p1.get('avg_rail_distance', -1)
        p2_rail = p2.get('avg_rail_distance', -1)
        if p1_rail > 0:
            p1_total_rail_dist += p1_rail
            p1_rail_games += 1
        if p2_rail > 0:
            p2_total_rail_dist += p2_rail
            p2_rail_games += 1
        
        p1_total_tight_rails += p1.get('tight_rail_count', 0)
        p2_total_tight_rails += p2.get('tight_rail_count', 0)
        
        # Court pressure
        p1_total_pressure_pct += p1.get('pressure_pct', 0) * game_duration
        p2_total_pressure_pct += p2.get('pressure_pct', 0) * game_duration
        p1_total_back_court_pct += p1.get('back_court_pct', 0) * game_duration
        p2_total_back_court_pct += p2.get('back_court_pct', 0) * game_duration
        
        # Fatigue (weighted by duration, only count valid data)
        p1_fatigue = p1.get('fatigue_score', 0)
        p2_fatigue = p2.get('fatigue_score', 0)
        p1_early = p1.get('early_speed', 0)
        p2_early = p2.get('early_speed', 0)
        p1_late = p1.get('late_speed', 0)
        p2_late = p2.get('late_speed', 0)
        p1_burst = p1.get('burst_decline_pct', 0)
        p2_burst = p2.get('burst_decline_pct', 0)
        
        if p1_fatigue > 0 or p1.get('speed_decline_pct', 0) != 0 or p1_early > 0:
            p1_total_fatigue += p1_fatigue * game_duration
            p1_total_speed_decline += p1.get('speed_decline_pct', 0) * game_duration
            p1_total_early_speed += p1_early * game_duration
            p1_total_late_speed += p1_late * game_duration
            p1_total_burst_decline += p1_burst * game_duration
            p1_fatigue_games += game_duration
        if p2_fatigue > 0 or p2.get('speed_decline_pct', 0) != 0 or p2_early > 0:
            p2_total_fatigue += p2_fatigue * game_duration
            p2_total_speed_decline += p2.get('speed_decline_pct', 0) * game_duration
            p2_total_early_speed += p2_early * game_duration
            p2_total_late_speed += p2_late * game_duration
            p2_total_burst_decline += p2_burst * game_duration
            p2_fatigue_games += game_duration
        
        # Zone analysis
        p1_zone = zone.get('player1', {}).get('aggregate', {})
        p2_zone = zone.get('player2', {}).get('aggregate', {})
        for key in p1_zone_totals:
            p1_zone_totals[key] += p1_zone.get(key, 0) * game_duration
            p2_zone_totals[key] += p2_zone.get(key, 0) * game_duration
        
        # Track trends
        p1_t_dom_trend.append(p1_t_dom)
        p2_t_dom_trend.append(p2_t_dom)
        p1_attack_trend.append(p1_attacks)
        p2_attack_trend.append(p2_attacks)
        p1_scramble_trend.append(p1_scramble)
        p2_scramble_trend.append(p2_scramble)
        
        # Split games for performance decay
        if i < num_games // 2:
            first_half_games.append(game)
        else:
            second_half_games.append(game)
        
        # Determine game winner (based on T-dominance primarily)
        if p1_t_dom > p2_t_dom:
            game_winner = "Player 1"
        elif p2_t_dom > p1_t_dom:
            game_winner = "Player 2"
        else:
            game_winner = "Tie"
        
        # Store per-game breakdown
        games.append({
            'name': game_names[i],
            'duration_seconds': game_duration,
            'duration_formatted': f"{int(game_duration // 60)}m {int(game_duration % 60)}s",
            'player1': {
                't_dominance': p1_t_dom,
                'scramble_score': round(p1_scramble, 1),
                'running_score': round(p1_running, 1),
                'attack_score': p1_attacks,
                'avg_rail_distance': p1_rail if p1_rail > 0 else None
            },
            'player2': {
                't_dominance': p2_t_dom,
                'scramble_score': round(p2_scramble, 1),
                'running_score': round(p2_running, 1),
                'attack_score': p2_attacks,
                'avg_rail_distance': p2_rail if p2_rail > 0 else None
            },
            'winner_by_t_control': game_winner,
            'analysis_summary': analysis.get('t_dominance', {}).get('summary', '')
        })
    
    # Calculate weighted averages
    if total_duration > 0:
        p1_avg_scramble = p1_total_scramble / total_duration
        p2_avg_scramble = p2_total_scramble / total_duration
        p1_avg_t_dom = p1_total_t_dominance / total_duration
        p2_avg_t_dom = p2_total_t_dominance / total_duration
        p1_avg_pressure_pct = p1_total_pressure_pct / total_duration
        p2_avg_pressure_pct = p2_total_pressure_pct / total_duration
        p1_avg_back_court_pct = p1_total_back_court_pct / total_duration
        p2_avg_back_court_pct = p2_total_back_court_pct / total_duration
        
        for key in p1_zone_totals:
            p1_zone_totals[key] = round(p1_zone_totals[key] / total_duration, 1)
            p2_zone_totals[key] = round(p2_zone_totals[key] / total_duration, 1)
    else:
        p1_avg_scramble = p2_avg_scramble = 0
        p1_avg_t_dom = p2_avg_t_dom = 50
        p1_avg_pressure_pct = p2_avg_pressure_pct = 0
        p1_avg_back_court_pct = p2_avg_back_court_pct = 0
    
    # Calculate rail averages - use pre-calculated percentages from games when available
    # (more accurate since they were calculated with correct video width)
    p1_avg_rail = round(p1_total_rail_dist / p1_rail_games, 1) if p1_rail_games > 0 else -1
    p2_avg_rail = round(p2_total_rail_dist / p2_rail_games, 1) if p2_rail_games > 0 else -1
    
    # Use pre-calculated percentage/feet/inches from games if available
    p1_total_rail_pct = 0
    p2_total_rail_pct = 0
    p1_rail_pct_games = 0
    p2_rail_pct_games = 0
    
    for game_data in game_analytics_list:
        p1 = game_data.get('player1', {})
        p2 = game_data.get('player2', {})
        
        p1_pct = p1.get('avg_rail_distance_pct', -1)
        p2_pct = p2.get('avg_rail_distance_pct', -1)
        
        if p1_pct > 0:
            p1_total_rail_pct += p1_pct
            p1_rail_pct_games += 1
        if p2_pct > 0:
            p2_total_rail_pct += p2_pct
            p2_rail_pct_games += 1
    
    # Calculate average percentages from games
    SQUASH_COURT_WIDTH_FT = 21.0
    SQUASH_COURT_WIDTH_INCHES = 252.0
    
    if p1_rail_pct_games > 0:
        p1_avg_rail_pct = round(p1_total_rail_pct / p1_rail_pct_games, 1)
        p1_avg_rail_ft = round(p1_avg_rail_pct / 100 * SQUASH_COURT_WIDTH_FT, 1)
        p1_avg_rail_inches = round(p1_avg_rail_pct / 100 * SQUASH_COURT_WIDTH_INCHES, 1)
    else:
        p1_avg_rail_pct = -1
        p1_avg_rail_ft = -1
        p1_avg_rail_inches = -1
    
    if p2_rail_pct_games > 0:
        p2_avg_rail_pct = round(p2_total_rail_pct / p2_rail_pct_games, 1)
        p2_avg_rail_ft = round(p2_avg_rail_pct / 100 * SQUASH_COURT_WIDTH_FT, 1)
        p2_avg_rail_inches = round(p2_avg_rail_pct / 100 * SQUASH_COURT_WIDTH_INCHES, 1)
    else:
        p2_avg_rail_pct = -1
        p2_avg_rail_ft = -1
        p2_avg_rail_inches = -1
    
    # Determine match trends
    def calculate_trend(values):
        if len(values) < 2:
            return 0
        # Compare last game to first game
        return ((values[-1] - values[0]) / max(values[0], 1)) * 100
    
    p1_t_dom_change = calculate_trend(p1_t_dom_trend)
    p2_t_dom_change = calculate_trend(p2_t_dom_trend)
    
    # Count game wins by T-dominance
    p1_games_won = sum(1 for g in games if g['winner_by_t_control'] == "Player 1")
    p2_games_won = sum(1 for g in games if g['winner_by_t_control'] == "Player 2")
    
    # Determine match winner
    if p1_games_won > p2_games_won:
        match_winner = "Player 1"
        match_score = f"{p1_games_won}-{p2_games_won}"
    elif p2_games_won > p1_games_won:
        match_winner = "Player 2"
        match_score = f"{p2_games_won}-{p1_games_won}"
    else:
        match_winner = "Tie"
        match_score = f"{p1_games_won}-{p2_games_won}"
    
    # Calculate performance decay (first half vs second half of match)
    def calculate_half_averages(game_list):
        if not game_list:
            return {'p1_t_dom': 0, 'p2_t_dom': 0, 'p1_attacks': 0, 'p2_attacks': 0, 'p1_scramble': 0, 'p2_scramble': 0}
        
        total_dur = sum(g.get('match_info', {}).get('duration_seconds', 1) for g in game_list)
        p1_t = sum(g.get('player1', {}).get('t_dominance', 0) * g.get('match_info', {}).get('duration_seconds', 1) for g in game_list) / max(total_dur, 1)
        p2_t = sum(g.get('player2', {}).get('t_dominance', 0) * g.get('match_info', {}).get('duration_seconds', 1) for g in game_list) / max(total_dur, 1)
        p1_a = sum(g.get('player1', {}).get('attack_score', 0) for g in game_list)
        p2_a = sum(g.get('player2', {}).get('attack_score', 0) for g in game_list)
        p1_s = sum(g.get('player1', {}).get('scramble_score', 0) * g.get('match_info', {}).get('duration_seconds', 1) for g in game_list) / max(total_dur, 1)
        p2_s = sum(g.get('player2', {}).get('scramble_score', 0) * g.get('match_info', {}).get('duration_seconds', 1) for g in game_list) / max(total_dur, 1)
        
        return {'p1_t_dom': p1_t, 'p2_t_dom': p2_t, 'p1_attacks': p1_a, 'p2_attacks': p2_a, 'p1_scramble': p1_s, 'p2_scramble': p2_s}
    
    first_half_stats = calculate_half_averages(first_half_games)
    second_half_stats = calculate_half_averages(second_half_games)
    
    def calc_decay(first, second, lower_is_better=False):
        if first == 0:
            return 0
        change = ((second - first) / first) * 100
        if lower_is_better:
            change = -change
        return round(change, 1)
    
    performance_decay = {
        'player1': {
            'first_half': {
                't_dominance': round(first_half_stats['p1_t_dom'], 1),
                'attacks': first_half_stats['p1_attacks'],
                'scramble': round(first_half_stats['p1_scramble'], 1)
            },
            'second_half': {
                't_dominance': round(second_half_stats['p1_t_dom'], 1),
                'attacks': second_half_stats['p1_attacks'],
                'scramble': round(second_half_stats['p1_scramble'], 1)
            },
            'decay': {
                't_dominance': calc_decay(first_half_stats['p1_t_dom'], second_half_stats['p1_t_dom']),
                'attack_rate': calc_decay(first_half_stats['p1_attacks'], second_half_stats['p1_attacks']),
                'scramble': calc_decay(first_half_stats['p1_scramble'], second_half_stats['p1_scramble'], lower_is_better=True)
            },
            'overall_trend': round((calc_decay(first_half_stats['p1_t_dom'], second_half_stats['p1_t_dom']) + 
                                   calc_decay(first_half_stats['p1_attacks'], second_half_stats['p1_attacks']) +
                                   calc_decay(first_half_stats['p1_scramble'], second_half_stats['p1_scramble'], lower_is_better=True)) / 3, 1)
        },
        'player2': {
            'first_half': {
                't_dominance': round(first_half_stats['p2_t_dom'], 1),
                'attacks': first_half_stats['p2_attacks'],
                'scramble': round(first_half_stats['p2_scramble'], 1)
            },
            'second_half': {
                't_dominance': round(second_half_stats['p2_t_dom'], 1),
                'attacks': second_half_stats['p2_attacks'],
                'scramble': round(second_half_stats['p2_scramble'], 1)
            },
            'decay': {
                't_dominance': calc_decay(first_half_stats['p2_t_dom'], second_half_stats['p2_t_dom']),
                'attack_rate': calc_decay(first_half_stats['p2_attacks'], second_half_stats['p2_attacks']),
                'scramble': calc_decay(first_half_stats['p2_scramble'], second_half_stats['p2_scramble'], lower_is_better=True)
            },
            'overall_trend': round((calc_decay(first_half_stats['p2_t_dom'], second_half_stats['p2_t_dom']) + 
                                   calc_decay(first_half_stats['p2_attacks'], second_half_stats['p2_attacks']) +
                                   calc_decay(first_half_stats['p2_scramble'], second_half_stats['p2_scramble'], lower_is_better=True)) / 3, 1)
        },
        'analysis': ''
    }
    
    # Generate performance decay analysis
    decay_analysis = []
    p1_overall = performance_decay['player1']['overall_trend']
    p2_overall = performance_decay['player2']['overall_trend']
    
    if p1_overall < -15:
        decay_analysis.append("Player 1 shows performance decline in later games - potential fatigue.")
    elif p1_overall > 15:
        decay_analysis.append("Player 1 is getting stronger as the match progresses.")
    
    if p2_overall < -15:
        decay_analysis.append("Player 2 shows performance decline in later games - potential fatigue.")
    elif p2_overall > 15:
        decay_analysis.append("Player 2 is getting stronger as the match progresses.")
    
    if not decay_analysis:
        decay_analysis.append("Both players maintaining consistent performance throughout the match.")
    
    performance_decay['analysis'] = ' '.join(decay_analysis)
    
    # Generate match analysis text
    total_minutes = int(total_duration // 60)
    
    if match_winner != "Tie":
        analysis_text = f"Over {num_games} games totaling {total_minutes} minutes, {match_winner} won the match {match_score} based on T-control. "
    else:
        analysis_text = f"This was an evenly contested {num_games}-game match lasting {total_minutes} minutes, tied at {match_score}. "
    
    if abs(p1_t_dom_change) > 10 or abs(p2_t_dom_change) > 10:
        if p1_t_dom_change > p2_t_dom_change:
            analysis_text += f"Player 1 showed significant improvement across the match, increasing T-control by {p1_t_dom_change:.0f}%. "
        else:
            analysis_text += f"Player 2 showed significant improvement across the match, increasing T-control by {p2_t_dom_change:.0f}%. "
    
    # Rail analysis (use inches for better precision)
    rail_analysis = {}
    if p1_avg_rail_inches > 0 and p2_avg_rail_inches > 0:
        if p1_avg_rail_inches < p2_avg_rail_inches:
            rail_analysis = {
                'winner': 'Player 1',
                'summary': f"Player 1 hit tighter rails (avg {p1_avg_rail_inches:.0f}\" from wall) compared to Player 2 ({p2_avg_rail_inches:.0f}\")."
            }
        else:
            rail_analysis = {
                'winner': 'Player 2',
                'summary': f"Player 2 hit tighter rails (avg {p2_avg_rail_inches:.0f}\" from wall) compared to Player 1 ({p1_avg_rail_inches:.0f}\")."
            }
    elif p1_avg_rail_inches > 0:
        rail_analysis = {'winner': 'Player 1', 'summary': f'Only Player 1 has sufficient rail tracking data (avg {p1_avg_rail_inches:.0f}\" from wall).'}
    elif p2_avg_rail_inches > 0:
        rail_analysis = {'winner': 'Player 2', 'summary': f'Only Player 2 has sufficient rail tracking data (avg {p2_avg_rail_inches:.0f}\" from wall).'}
    else:
        rail_analysis = {'winner': 'N/A', 'summary': 'Insufficient ball tracking data for rail analysis.'}
    
    # Zone analysis text
    zone_analysis_text = []
    p1_front = p1_zone_totals.get('front_court', 0)
    p2_front = p2_zone_totals.get('front_court', 0)
    p1_back = p1_zone_totals.get('back_court', 0)
    p2_back = p2_zone_totals.get('back_court', 0)
    
    if p1_front > p2_front + 10:
        zone_analysis_text.append("Player 1 spent more time in the front court (attacking).")
    elif p2_front > p1_front + 10:
        zone_analysis_text.append("Player 2 spent more time in the front court (attacking).")
    
    if p1_back > p2_back + 15:
        zone_analysis_text.append("Player 1 was pushed to the back more often (defensive).")
    elif p2_back > p1_back + 15:
        zone_analysis_text.append("Player 2 was pushed to the back more often (defensive).")
    
    if not zone_analysis_text:
        zone_analysis_text.append("Both players used the court similarly across the match.")
    
    # Calculate fatigue averages
    p1_avg_fatigue = round(p1_total_fatigue / p1_fatigue_games, 0) if p1_fatigue_games > 0 else 0
    p2_avg_fatigue = round(p2_total_fatigue / p2_fatigue_games, 0) if p2_fatigue_games > 0 else 0
    p1_avg_speed_decline = round(p1_total_speed_decline / p1_fatigue_games, 1) if p1_fatigue_games > 0 else 0
    p2_avg_speed_decline = round(p2_total_speed_decline / p2_fatigue_games, 1) if p2_fatigue_games > 0 else 0
    p1_avg_early_speed = round(p1_total_early_speed / p1_fatigue_games, 1) if p1_fatigue_games > 0 else 0
    p2_avg_early_speed = round(p2_total_early_speed / p2_fatigue_games, 1) if p2_fatigue_games > 0 else 0
    p1_avg_late_speed = round(p1_total_late_speed / p1_fatigue_games, 1) if p1_fatigue_games > 0 else 0
    p2_avg_late_speed = round(p2_total_late_speed / p2_fatigue_games, 1) if p2_fatigue_games > 0 else 0
    p1_avg_burst_decline = round(p1_total_burst_decline / p1_fatigue_games, 1) if p1_fatigue_games > 0 else 0
    p2_avg_burst_decline = round(p2_total_burst_decline / p2_fatigue_games, 1) if p2_fatigue_games > 0 else 0
    
    # Fatigue analysis
    if p1_avg_fatigue == 0 and p2_avg_fatigue == 0:
        fatigue_winner = 'N/A'
        fatigue_summary = 'Insufficient data for fatigue analysis.'
    elif p1_avg_fatigue < p2_avg_fatigue - 5:
        fatigue_winner = 'Player 1'
        fatigue_summary = f"Player 1 showed better conditioning (fatigue score: {p1_avg_fatigue:.0f}) compared to Player 2 ({p2_avg_fatigue:.0f}). Player 1 slowed {p1_avg_speed_decline:.0f}% vs Player 2's {p2_avg_speed_decline:.0f}%."
    elif p2_avg_fatigue < p1_avg_fatigue - 5:
        fatigue_winner = 'Player 2'
        fatigue_summary = f"Player 2 showed better conditioning (fatigue score: {p2_avg_fatigue:.0f}) compared to Player 1 ({p1_avg_fatigue:.0f}). Player 2 slowed {p2_avg_speed_decline:.0f}% vs Player 1's {p1_avg_speed_decline:.0f}%."
    else:
        fatigue_winner = 'Even'
        fatigue_summary = f"Both players maintained similar conditioning. Player 1 fatigue: {p1_avg_fatigue:.0f} ({p1_avg_speed_decline:.0f}% decline), Player 2: {p2_avg_fatigue:.0f} ({p2_avg_speed_decline:.0f}% decline)."
    
    return {
        'match_type': 'combined',
        'num_games': num_games,
        'match_info': {
            'total_duration_seconds': total_duration,
            'total_duration_formatted': f"{total_minutes}m {int(total_duration % 60)}s",
            'total_frames': total_frames,
            'total_frames_analyzed': total_frames_analyzed,
            'games_played': num_games
        },
        'match_result': {
            'winner': match_winner,
            'score': match_score,
            'player1_games': p1_games_won,
            'player2_games': p2_games_won
        },
        'player1': {
            'name': 'Player 1 (Left)',
            'games_won': p1_games_won,
            'avg_t_dominance': round(p1_avg_t_dom, 1),
            'avg_scramble_score': round(p1_avg_scramble, 1),
            'total_running_score': round(p1_total_running, 1),
            'total_attack_score': p1_total_attacks,
            'avg_rail_distance': p1_avg_rail,
            'avg_rail_distance_pct': p1_avg_rail_pct,
            'avg_rail_distance_ft': p1_avg_rail_ft,
            'avg_rail_distance_inches': p1_avg_rail_inches,
            'total_tight_rails': p1_total_tight_rails,
            'tight_rail_count': p1_total_tight_rails,  # Alias for compatibility
            't_dominance_trend': round(p1_t_dom_change, 1),
            'pressure_pct': round(p1_avg_pressure_pct, 1),
            'back_court_pct': round(p1_avg_back_court_pct, 1),
            'fatigue_score': p1_avg_fatigue,
            'speed_decline_pct': p1_avg_speed_decline,
            'early_speed': p1_avg_early_speed,
            'late_speed': p1_avg_late_speed,
            'burst_decline_pct': p1_avg_burst_decline
        },
        'player2': {
            'name': 'Player 2 (Right)',
            'games_won': p2_games_won,
            'avg_t_dominance': round(p2_avg_t_dom, 1),
            'avg_scramble_score': round(p2_avg_scramble, 1),
            'total_running_score': round(p2_total_running, 1),
            'total_attack_score': p2_total_attacks,
            'avg_rail_distance': p2_avg_rail,
            'avg_rail_distance_pct': p2_avg_rail_pct,
            'avg_rail_distance_ft': p2_avg_rail_ft,
            'avg_rail_distance_inches': p2_avg_rail_inches,
            'total_tight_rails': p2_total_tight_rails,
            'tight_rail_count': p2_total_tight_rails,  # Alias for compatibility
            't_dominance_trend': round(p2_t_dom_change, 1),
            'pressure_pct': round(p2_avg_pressure_pct, 1),
            'back_court_pct': round(p2_avg_back_court_pct, 1),
            'fatigue_score': p2_avg_fatigue,
            'speed_decline_pct': p2_avg_speed_decline,
            'early_speed': p2_avg_early_speed,
            'late_speed': p2_avg_late_speed,
            'burst_decline_pct': p2_avg_burst_decline
        },
        'games': games,
        'trends': {
            'p1_t_dominance': p1_t_dom_trend,
            'p2_t_dominance': p2_t_dom_trend,
            'p1_attacks': p1_attack_trend,
            'p2_attacks': p2_attack_trend,
            'p1_scramble': p1_scramble_trend,
            'p2_scramble': p2_scramble_trend
        },
        'zone_analysis': {
            'player1': {'aggregate': p1_zone_totals},
            'player2': {'aggregate': p2_zone_totals},
            'analysis': ' '.join(zone_analysis_text)
        },
        'performance_decay': performance_decay,
        'analysis': {
            'match_summary': analysis_text,
            't_dominance': {
                'controller': match_winner if match_winner != "Tie" else "Neither",
                'summary': f"{match_winner} controlled the T more consistently across the match." if match_winner != "Tie" else "Both players shared T-control evenly."
            },
            'scramble': {
                'winner': 'Player 1' if p1_avg_scramble < p2_avg_scramble else 'Player 2',
                'ratio': round(max(p1_avg_scramble, p2_avg_scramble) / max(min(p1_avg_scramble, p2_avg_scramble), 1), 2)
            },
            'running': {
                'harder_worker': 'Player 1' if p1_total_running > p2_total_running else 'Player 2',
                'ratio': round(max(p1_total_running, p2_total_running) / max(min(p1_total_running, p2_total_running), 1), 2)
            },
            'attack': {
                'more_aggressive': 'Player 1' if p1_total_attacks > p2_total_attacks else 'Player 2'
            },
            'tight_rails': rail_analysis,
            'court_pressure': {
                'winner': 'Player 1' if p1_avg_pressure_pct > p2_avg_pressure_pct + 5 else 
                          'Player 2' if p2_avg_pressure_pct > p1_avg_pressure_pct + 5 else 'Even',
                'summary': (f"Player 1 was in front {p1_avg_pressure_pct:.0f}% of the time (opponent in back court {p2_avg_back_court_pct:.0f}%). " +
                           f"Player 2 was in front {p2_avg_pressure_pct:.0f}% of the time (opponent in back court {p1_avg_back_court_pct:.0f}%). " +
                          (f"Player 1 applied more court pressure, forcing Player 2 to defend." if p1_avg_pressure_pct > p2_avg_pressure_pct + 5 else
                           f"Player 2 applied more court pressure, forcing Player 1 to defend." if p2_avg_pressure_pct > p1_avg_pressure_pct + 5 else
                           "Both players traded court position evenly."))
            },
            'fatigue': {
                'winner': fatigue_winner,
                'summary': fatigue_summary
            }
        }
    }

