#!/usr/bin/env python3
"""
Sport-Specific Configuration for Racket Sports Analytics
Defines court dimensions, detection thresholds, and metric parameters per sport.
"""

# Sport-specific configurations
SPORT_CONFIGS = {
    'squash': {
        'name': 'Squash',
        'model': 'yolov8n.pt',  # Will use fine-tuned model when available
        'court_width_meters': 6.4,
        'court_length_meters': 9.75,
        't_position_ratio': 0.55,  # T is 55% from front wall (closer to front)
        'ball_conf_threshold': 0.3,
        'player_conf_threshold': 0.5,
        'has_walls': True,
        'has_tight_rails': True,
        'metrics': {
            't_dominance': {'enabled': True, 'label': 'T-Dominance'},
            'scramble': {'enabled': True, 'label': 'Scramble Score'},
            'running': {'enabled': True, 'label': 'Running Score'},
            'attack': {'enabled': True, 'label': 'Attack Score'},
            'tight_rails': {'enabled': True, 'label': 'Tight Rails'},
            'zone_analysis': {'enabled': True, 'label': 'Court Zone Analysis'},
            'performance_decay': {'enabled': True, 'label': 'Performance Decay'}
        },
        'zones': {
            'front_ratio': 0.30,  # Front 30% of court
            'mid_ratio': 0.40,    # Middle 40% (T area)
            'back_ratio': 0.30    # Back 30%
        }
    },
    'padel': {
        'name': 'Padel',
        'model': 'yolov8n.pt',
        'court_width_meters': 10.0,
        'court_length_meters': 20.0,
        't_position_ratio': 0.50,  # Center of court
        'ball_conf_threshold': 0.25,
        'player_conf_threshold': 0.5,
        'has_walls': True,
        'has_tight_rails': True,  # Padel has walls too
        'metrics': {
            't_dominance': {'enabled': True, 'label': 'Net Control'},
            'scramble': {'enabled': True, 'label': 'Court Coverage'},
            'running': {'enabled': True, 'label': 'Distance Covered'},
            'attack': {'enabled': True, 'label': 'Attack Score'},
            'tight_rails': {'enabled': True, 'label': 'Wall Shots'},
            'zone_analysis': {'enabled': True, 'label': 'Court Position'},
            'performance_decay': {'enabled': True, 'label': 'Stamina Trend'}
        },
        'zones': {
            'front_ratio': 0.35,  # Net area
            'mid_ratio': 0.30,
            'back_ratio': 0.35   # Back wall area
        }
    },
    'tennis': {
        'name': 'Tennis',
        'model': 'yolov8n.pt',
        'court_width_meters': 10.97,  # Singles court width
        'court_length_meters': 23.77,
        't_position_ratio': 0.50,  # Baseline is key position
        'ball_conf_threshold': 0.2,  # Tennis ball is fast, lower threshold
        'player_conf_threshold': 0.5,
        'has_walls': False,
        'has_tight_rails': False,  # No walls in tennis
        'metrics': {
            't_dominance': {'enabled': True, 'label': 'Baseline Control'},
            'scramble': {'enabled': True, 'label': 'Court Coverage'},
            'running': {'enabled': True, 'label': 'Distance Covered'},
            'attack': {'enabled': True, 'label': 'Winners'},
            'tight_rails': {'enabled': False, 'label': 'N/A'},
            'zone_analysis': {'enabled': True, 'label': 'Court Position'},
            'performance_decay': {'enabled': True, 'label': 'Endurance'}
        },
        'zones': {
            'front_ratio': 0.40,  # Net approach zone
            'mid_ratio': 0.30,    # Mid-court
            'back_ratio': 0.30    # Baseline
        }
    },
    'table_tennis': {
        'name': 'Table Tennis',
        'model': 'yolov8n.pt',
        'court_width_meters': 1.525,
        'court_length_meters': 2.74,
        't_position_ratio': 0.50,
        'ball_conf_threshold': 0.15,  # Very small, fast ball
        'player_conf_threshold': 0.5,
        'has_walls': False,
        'has_tight_rails': False,
        'metrics': {
            't_dominance': {'enabled': True, 'label': 'Table Control'},
            'scramble': {'enabled': True, 'label': 'Reaction Speed'},
            'running': {'enabled': True, 'label': 'Movement'},
            'attack': {'enabled': True, 'label': 'Smash Count'},
            'tight_rails': {'enabled': False, 'label': 'N/A'},
            'zone_analysis': {'enabled': True, 'label': 'Position Analysis'},
            'performance_decay': {'enabled': True, 'label': 'Focus Trend'}
        },
        'zones': {
            'front_ratio': 0.50,  # Close to table
            'mid_ratio': 0.30,
            'back_ratio': 0.20    # Further back
        }
    }
}

# Camera angle configurations
CAMERA_ANGLES = {
    'back': {
        'description': 'Camera behind players, looking at front wall',
        't_adjustment': 0.0,  # No adjustment needed
        'perspective_correction': 1.0
    },
    'side': {
        'description': 'Camera from side of court',
        't_adjustment': 0.0,
        'perspective_correction': 0.85  # Adjust for side perspective
    },
    'front': {
        'description': 'Camera at front, looking at players',
        't_adjustment': -0.1,  # T appears higher in frame
        'perspective_correction': 1.0
    },
    'overhead': {
        'description': 'Bird\'s eye view from above',
        't_adjustment': 0.0,
        'perspective_correction': 1.0
    }
}

def get_sport_config(sport='squash'):
    """Get configuration for a specific sport."""
    return SPORT_CONFIGS.get(sport.lower(), SPORT_CONFIGS['squash'])

def get_camera_config(angle='back'):
    """Get configuration for a specific camera angle."""
    return CAMERA_ANGLES.get(angle.lower(), CAMERA_ANGLES['back'])

def get_t_position_ratio(sport='squash', camera_angle='back'):
    """Calculate T position ratio based on sport and camera angle."""
    sport_config = get_sport_config(sport)
    camera_config = get_camera_config(camera_angle)
    
    base_ratio = sport_config['t_position_ratio']
    adjustment = camera_config['t_adjustment']
    
    return max(0.3, min(0.7, base_ratio + adjustment))

def get_court_dimensions(sport='squash'):
    """Get court dimensions in meters for a sport."""
    config = get_sport_config(sport)
    return {
        'width': config['court_width_meters'],
        'length': config['court_length_meters']
    }

def pixels_to_meters(pixels, video_width, sport='squash'):
    """Convert pixel distance to meters based on sport court dimensions."""
    config = get_sport_config(sport)
    meters_per_pixel = config['court_width_meters'] / video_width
    return pixels * meters_per_pixel

def pixels_to_feet(pixels, video_width, sport='squash'):
    """Convert pixel distance to feet based on sport court dimensions."""
    meters = pixels_to_meters(pixels, video_width, sport)
    return meters * 3.28084  # 1 meter = 3.28084 feet

def get_enabled_metrics(sport='squash'):
    """Get list of enabled metrics for a sport."""
    config = get_sport_config(sport)
    return {
        key: value for key, value in config['metrics'].items() 
        if value['enabled']
    }

def get_zone_boundaries(sport='squash'):
    """Get zone boundary ratios for a sport."""
    config = get_sport_config(sport)
    zones = config['zones']
    
    front_end = zones['front_ratio']
    mid_end = front_end + zones['mid_ratio']
    
    return {
        'front': (0, front_end),
        'mid': (front_end, mid_end),
        'back': (mid_end, 1.0)
    }

def get_detection_thresholds(sport='squash'):
    """Get detection confidence thresholds for a sport."""
    config = get_sport_config(sport)
    return {
        'ball': config['ball_conf_threshold'],
        'player': config['player_conf_threshold']
    }

# Validation thresholds
QUALITY_THRESHOLDS = {
    'min_ball_detection_rate': 0.10,  # At least 10% of frames should have ball
    'min_player_detection_rate': 0.70,  # At least 70% of frames should have 2 players
    'min_confidence_for_analytics': 0.5,  # Minimum confidence for reliable analytics
    'ball_weight_in_quality': 0.4,
    'player_weight_in_quality': 0.6
}

def calculate_quality_score(ball_rate, player_rate):
    """Calculate overall quality score from detection rates."""
    weights = QUALITY_THRESHOLDS
    score = (ball_rate * weights['ball_weight_in_quality'] + 
             player_rate * weights['player_weight_in_quality']) * 100
    return round(score, 1)

def is_data_reliable(ball_rate, player_rate):
    """Check if detection data is reliable enough for analytics."""
    return (ball_rate >= QUALITY_THRESHOLDS['min_ball_detection_rate'] and 
            player_rate >= QUALITY_THRESHOLDS['min_player_detection_rate'])

