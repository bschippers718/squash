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
    Identify the two main players from detections.
    Players are the two largest, highest-confidence person detections.
    Returns player positions per frame.
    """
    frames_data = []
    
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
        
        # Take top 2 persons as players
        players = persons[:2] if len(persons) >= 2 else persons
        
        frame_data = {
            'frame': frame_num,
            'timestamp': timestamp,
            'players': []
        }
        
        for i, player in enumerate(players):
            center = calculate_center(player['bbox'])
            frame_data['players'].append({
                'id': i,
                'center': center,
                'bbox': player['bbox'],
                'confidence': player['confidence']
            })
        
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

def analyze_tight_rails(frames_data, video_width, video_height):
    """
    Analyze how tight each player's rail shots are.
    A tight rail is when the ball is close to the side wall.
    
    Uses ball position changes to attribute shots to the player who was
    closer to the ball when its trajectory changed (indicating a hit).
    """
    # Track ball positions when near each player
    p1_rail_distances = []  # Distances from side walls when P1 likely hit
    p2_rail_distances = []  # Distances from side walls when P2 likely hit
    
    prev_ball_center = None
    prev_ball_velocity = None
    hit_cooldown = 0  # Frames to wait between detected hits
    
    for frame_data in frames_data:
        ball_bbox = frame_data.get('ball')
        players = frame_data.get('players', [])
        
        if hit_cooldown > 0:
            hit_cooldown -= 1
        
        if not ball_bbox or len(players) < 2:
            prev_ball_center = None
            continue
        
        ball_center = frame_data.get('ball_center')
        if not ball_center:
            continue
        
        # Sort players by x position (left = p1, right = p2)
        players.sort(key=lambda p: p['center'][0])
        
        # Calculate wall distances
        wall_dist = calculate_wall_distances(ball_bbox, video_width, video_height)
        if not wall_dist:
            prev_ball_center = ball_center
            continue
        
        # Minimum distance to either side wall (for rail shots)
        min_side_wall_dist = min(wall_dist['left_wall'], wall_dist['right_wall'])
        
        # Calculate ball velocity (direction change indicates a hit)
        hit_detected = False
        if prev_ball_center is not None:
            current_velocity = (
                ball_center[0] - prev_ball_center[0],
                ball_center[1] - prev_ball_center[1]
            )
            
            # Check for direction change (indicates a hit)
            if prev_ball_velocity is not None and hit_cooldown <= 0:
                # Check if velocity changed significantly (dot product negative = direction change)
                dot_product = (current_velocity[0] * prev_ball_velocity[0] + 
                             current_velocity[1] * prev_ball_velocity[1])
                velocity_magnitude = math.sqrt(current_velocity[0]**2 + current_velocity[1]**2)
                
                # Direction changed or ball moved significantly
                if dot_product < 0 or velocity_magnitude > 20:
                    hit_detected = True
            
            prev_ball_velocity = current_velocity
        else:
            prev_ball_velocity = None
        
        # Determine which player is closer to the ball
        p1_dist = calculate_distance(players[0]['center'], ball_center)
        p2_dist = calculate_distance(players[1]['center'], ball_center)
        
        # Attribute shot to closer player
        # Use a more generous threshold - players can be further from ball
        max_attribution_distance = video_width * 0.4  # 40% of court width
        
        if p1_dist < max_attribution_distance or p2_dist < max_attribution_distance:
            if p1_dist < p2_dist:
                # Player 1 is closer - attribute this ball position to them
                p1_rail_distances.append(min_side_wall_dist)
                if hit_detected:
                    hit_cooldown = 5  # Wait 5 frames before next hit detection
            else:
                # Player 2 is closer - attribute this ball position to them
                p2_rail_distances.append(min_side_wall_dist)
                if hit_detected:
                    hit_cooldown = 5
        
        prev_ball_center = ball_center
    
    # Calculate averages - use median for more robust measurement
    def safe_average(distances):
        if not distances:
            return None
        # Remove outliers (ball positions in center of court)
        filtered = [d for d in distances if d < video_width * 0.4]
        if not filtered:
            return sum(distances) / len(distances)
        return sum(filtered) / len(filtered)
    
    p1_avg_rail = safe_average(p1_rail_distances)
    p2_avg_rail = safe_average(p2_rail_distances)
    
    # Handle case where one or both players have no data
    if p1_avg_rail is None and p2_avg_rail is None:
        # No rail data at all
        return {
            'player1': {
                'avg_wall_distance': -1,  # -1 indicates no data
                'tight_rail_count': 0,
                'total_shots_analyzed': 0
            },
            'player2': {
                'avg_wall_distance': -1,  # -1 indicates no data
                'tight_rail_count': 0,
                'total_shots_analyzed': 0
            },
            'analysis': {
                'winner': 'N/A',
                'summary': 'Unable to analyze rail tightness - ball tracking data insufficient.'
            }
        }
    
    # If only one player has data, mark the other as no data
    p1_has_data = p1_avg_rail is not None and len(p1_rail_distances) >= 5
    p2_has_data = p2_avg_rail is not None and len(p2_rail_distances) >= 5
    
    if not p1_has_data:
        p1_avg_rail = -1  # -1 indicates no data
    if not p2_has_data:
        p2_avg_rail = -1  # -1 indicates no data
    
    # Count tight rails (within 15% of court width from wall - more generous threshold)
    tight_threshold = video_width * 0.15
    p1_tight_count = sum(1 for d in p1_rail_distances if d < tight_threshold)
    p2_tight_count = sum(1 for d in p2_rail_distances if d < tight_threshold)
    
    # Analysis - handle missing data cases
    if p1_avg_rail < 0 and p2_avg_rail < 0:
        rail_winner = 'N/A'
        rail_analysis = "Unable to analyze rail tightness - ball tracking data insufficient for both players."
    elif p1_avg_rail < 0:
        rail_winner = 'Player 2'
        rail_analysis = "Only Player 2 has sufficient ball tracking data for rail analysis. Player 1's rail tightness could not be measured."
    elif p2_avg_rail < 0:
        rail_winner = 'Player 1'
        rail_analysis = "Only Player 1 has sufficient ball tracking data for rail analysis. Player 2's rail tightness could not be measured."
    elif p1_avg_rail > 0 and p2_avg_rail > 0:
        ratio = max(p1_avg_rail, p2_avg_rail) / min(p1_avg_rail, p2_avg_rail)
        if p1_avg_rail < p2_avg_rail and ratio > 1.15:
            rail_winner = 'Player 1'
            rail_analysis = "Player 1 is hitting tighter rails, keeping the ball closer to the wall and making retrieval more difficult for their opponent."
        elif p2_avg_rail < p1_avg_rail and ratio > 1.15:
            rail_winner = 'Player 2'
            rail_analysis = "Player 2 is hitting tighter rails, keeping the ball closer to the wall and making retrieval more difficult for their opponent."
        else:
            rail_winner = 'Even'
            rail_analysis = "Both players are hitting rails with similar tightness. Neither has a clear advantage in wall proximity."
    else:
        rail_winner = 'Even'
        rail_analysis = "Both players are hitting rails with similar tightness. Neither has a clear advantage in wall proximity."
    
    # Calculate percentage of court width (resolution-independent)
    # A squash court is ~21 feet wide, so we can estimate feet from percentage
    SQUASH_COURT_WIDTH_FT = 21.0
    p1_pct = (p1_avg_rail / video_width * 100) if p1_avg_rail > 0 else -1
    p2_pct = (p2_avg_rail / video_width * 100) if p2_avg_rail > 0 else -1
    p1_feet = (p1_pct / 100 * SQUASH_COURT_WIDTH_FT) if p1_pct > 0 else -1
    p2_feet = (p2_pct / 100 * SQUASH_COURT_WIDTH_FT) if p2_pct > 0 else -1
    
    return {
        'player1': {
            'avg_wall_distance': round(p1_avg_rail, 1),  # Keep raw pixels for backwards compat
            'avg_wall_distance_pct': round(p1_pct, 1) if p1_pct > 0 else -1,
            'avg_wall_distance_ft': round(p1_feet, 1) if p1_feet > 0 else -1,
            'tight_rail_count': p1_tight_count,
            'total_shots_analyzed': len(p1_rail_distances)
        },
        'player2': {
            'avg_wall_distance': round(p2_avg_rail, 1),
            'avg_wall_distance_pct': round(p2_pct, 1) if p2_pct > 0 else -1,
            'avg_wall_distance_ft': round(p2_feet, 1) if p2_feet > 0 else -1,
            'tight_rail_count': p2_tight_count,
            'total_shots_analyzed': len(p2_rail_distances)
        },
        'analysis': {
            'winner': rail_winner,
            'summary': rail_analysis
        },
        'video_width': video_width  # Store for reference
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
        
        # Sort players by x position to maintain consistency (left = player1, right = player2)
        players.sort(key=lambda p: p['center'][0])
        
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
            'player1': {'avg_wall_distance': -1, 'tight_rail_count': 0, 'total_shots_analyzed': 0},
            'player2': {'avg_wall_distance': -1, 'tight_rail_count': 0, 'total_shots_analyzed': 0},
            'analysis': {'winner': 'N/A', 'summary': 'Wall shots not applicable for this sport.'}
        }
    
    # Analyze shot sequences for the Shot Sequencer feature
    shot_sequences = analyze_shot_sequences(frames_data, fps, video_width, video_height)
    
    # Analyze zone dwell time (heatmap data)
    zone_analysis = analyze_zone_dwell_time(frames_data, video_width, video_height)
    
    # Analyze performance decay (first half vs second half)
    performance_decay = analyze_performance_decay(frames_data, video_width, video_height, fps)
    
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
            'tight_rail_count': tight_rails['player1']['tight_rail_count']
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
            'tight_rail_count': tight_rails['player2']['tight_rail_count']
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
            'tight_rails': tight_rails['analysis']
        },
        'insights': generate_match_insights(
            p1_avg_dist, p2_avg_dist,
            p1_total_dist, p2_total_dist,
            p1_t_dominance, p2_t_dominance,
            p1_attacks, p2_attacks
        ),
        'shot_sequences': shot_sequences,
        'zone_analysis': zone_analysis,
        'performance_decay': performance_decay
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
        
        # Sort players by x position (left = p1, right = p2)
        players_sorted = sorted(players[:2], key=lambda p: p['center'][0])
        
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
        
        for key in p1_zone_totals:
            p1_zone_totals[key] = round(p1_zone_totals[key] / total_duration, 1)
            p2_zone_totals[key] = round(p2_zone_totals[key] / total_duration, 1)
    else:
        p1_avg_scramble = p2_avg_scramble = 0
        p1_avg_t_dom = p2_avg_t_dom = 50
    
    # Calculate rail averages
    p1_avg_rail = round(p1_total_rail_dist / p1_rail_games, 1) if p1_rail_games > 0 else -1
    p2_avg_rail = round(p2_total_rail_dist / p2_rail_games, 1) if p2_rail_games > 0 else -1
    
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
    
    # Rail analysis
    rail_analysis = {}
    if p1_avg_rail > 0 and p2_avg_rail > 0:
        if p1_avg_rail < p2_avg_rail:
            rail_analysis = {
                'winner': 'Player 1',
                'summary': f"Player 1 hit tighter rails (avg {p1_avg_rail:.1f}px from wall) compared to Player 2 ({p2_avg_rail:.1f}px)."
            }
        else:
            rail_analysis = {
                'winner': 'Player 2',
                'summary': f"Player 2 hit tighter rails (avg {p2_avg_rail:.1f}px from wall) compared to Player 1 ({p1_avg_rail:.1f}px)."
            }
    elif p1_avg_rail > 0:
        rail_analysis = {'winner': 'Player 1', 'summary': 'Only Player 1 has sufficient rail tracking data.'}
    elif p2_avg_rail > 0:
        rail_analysis = {'winner': 'Player 2', 'summary': 'Only Player 2 has sufficient rail tracking data.'}
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
            'total_tight_rails': p1_total_tight_rails,
            't_dominance_trend': round(p1_t_dom_change, 1)
        },
        'player2': {
            'name': 'Player 2 (Right)',
            'games_won': p2_games_won,
            'avg_t_dominance': round(p2_avg_t_dom, 1),
            'avg_scramble_score': round(p2_avg_scramble, 1),
            'total_running_score': round(p2_total_running, 1),
            'total_attack_score': p2_total_attacks,
            'avg_rail_distance': p2_avg_rail,
            'total_tight_rails': p2_total_tight_rails,
            't_dominance_trend': round(p2_t_dom_change, 1)
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
            'tight_rails': rail_analysis
        }
    }

