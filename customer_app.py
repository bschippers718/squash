#!/usr/bin/env python3
"""
Customer-Facing Video Upload & Processing Website
Allows customers to upload videos and track processing status.
"""

import os

# Load environment variables from .env file (optional - for local dev)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed (production uses env vars directly)

# Set YOLO config directory to writable location (for Railway/cloud deployments)
# This must be set BEFORE importing ultralytics
os.environ['YOLO_CONFIG_DIR'] = '/tmp/ultralytics'
os.environ['HOME'] = os.environ.get('HOME', '/tmp')  # Fallback for cloud environments

# Disable OpenCV GUI features for headless environments
# These must be set BEFORE importing cv2
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
# Prevent OpenCV from trying to load libGL.so.1
os.environ['OPENCV_DISABLE_OPENCL'] = '1'

from flask import Flask, render_template, request, jsonify, send_from_directory, Response, g
import re
import uuid
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates/customer', static_folder='static')

# Disable caching for all responses
@app.after_request
def add_no_cache_headers(response):
    """Add headers to prevent caching of API and HTML responses."""
    # Don't cache JSON API responses or HTML pages
    if response.content_type and ('application/json' in response.content_type or 'text/html' in response.content_type):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# Configuration
UPLOAD_FOLDER = Path('uploads')
RESULTS_FOLDER = Path('customer_results')
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'm4v', 'webm'}
# No file size limit - videos can be large
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = None  # Unlimited

# Create directories
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

# Job tracking (in production, use Redis or a database)
jobs = {}
jobs_lock = threading.Lock()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_job(job_id):
    with jobs_lock:
        return jobs.get(job_id, {}).copy()

def update_job(job_id, **kwargs):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)

def process_video_task(job_id, video_path, sport='squash'):
    """Background task to process video with sport-specific parameters"""
    import traceback
    import sys
    
    print(f"[{job_id}] Starting video processing task...", flush=True)
    
    try:
        # Import here to avoid loading model at startup
        print(f"[{job_id}] Importing video_processor...", flush=True)
        from video_processor import process_video
        print(f"[{job_id}] video_processor imported successfully", flush=True)
        
        update_job(job_id, status='processing', progress=0, message=f'Starting {sport} video processing...')
        
        output_dir = RESULTS_FOLDER / job_id
        output_dir.mkdir(exist_ok=True)
        
        def progress_callback(progress, message):
            print(f"[{job_id}] Progress: {progress}% - {message}", flush=True)
            update_job(job_id, progress=progress, message=message)
        
        # Pass sport parameter to video processor
        # Use frame_skip=1 to process every frame for accurate ball tracking
        # Use max_process_width=None for full resolution (better ball detection)
        print(f"[{job_id}] Calling process_video...", flush=True)
        result = process_video(video_path, str(output_dir), progress_callback, sport=sport,
                               frame_skip=1, max_process_width=None)
        
        if result and result.get('success'):
            # Store sport info in result
            result['sport'] = sport
            update_job(
                job_id,
                status='completed',
                progress=100,
                message='Processing complete!',
                result=result,
                sport=sport,
                completed_at=datetime.now().isoformat()
            )
            print(f"[{job_id}] Processing completed successfully!", flush=True)
            
            # Auto-save to Supabase if user was logged in
            job = get_job(job_id)
            user_id = job.get('user_id')
            if user_id:
                try:
                    from supabase_client import save_match, upload_player_image
                    
                    analytics = result.get('squash_analytics', {})
                    
                    # Upload player image
                    players_image_url = None
                    output_dir = RESULTS_FOLDER / job_id
                    players_images = list(output_dir.glob("*_players.jpg"))
                    if players_images:
                        players_image_url = upload_player_image(job_id, str(players_images[0]))
                    
                    # Get player names
                    player1_name = analytics.get('player1', {}).get('name')
                    player2_name = analytics.get('player2', {}).get('name')
                    
                    # Save to Supabase
                    save_match(
                        user_id=user_id,
                        job_id=job_id,
                        filename=job.get('filename', job_id),
                        analytics=analytics,
                        sport=sport,
                        players_image_url=players_image_url,
                        player1_name=player1_name,
                        player2_name=player2_name
                    )
                    print(f"[{job_id}] Auto-saved to Supabase for user {user_id}", flush=True)
                except Exception as e:
                    print(f"[{job_id}] Auto-save to Supabase failed: {e}", flush=True)
        else:
            error_msg = result.get('error', 'Unknown error') if result else 'Processing failed'
            print(f"[{job_id}] Processing failed: {error_msg}", flush=True)
            update_job(job_id, status='failed', message=error_msg)
            
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[{job_id}] EXCEPTION: {error_msg}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        update_job(job_id, status='failed', message=error_msg)

@app.route('/favicon.ico')
def favicon():
    """Return a simple tennis ball favicon"""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="45" fill="#c8e600"/>
        <path d="M50 5 Q25 50 50 95" stroke="white" stroke-width="4" fill="none"/>
        <path d="M50 5 Q75 50 50 95" stroke="white" stroke-width="4" fill="none"/>
    </svg>'''
    return svg, 200, {'Content-Type': 'image/svg+xml'}

@app.route('/debug-env')
def debug_env():
    """Debug endpoint to check environment variables"""
    return jsonify({
        'CLERK_PUBLISHABLE_KEY': os.environ.get('CLERK_PUBLISHABLE_KEY', 'NOT SET')[:20] + '...' if os.environ.get('CLERK_PUBLISHABLE_KEY') else 'NOT SET',
        'CLERK_FRONTEND_API': os.environ.get('CLERK_FRONTEND_API', 'NOT SET')[:30] + '...' if os.environ.get('CLERK_FRONTEND_API') else 'NOT SET',
        'SUPABASE_URL': os.environ.get('SUPABASE_URL', 'NOT SET')[:30] + '...' if os.environ.get('SUPABASE_URL') else 'NOT SET',
        'all_env_keys': [k for k in os.environ.keys() if 'CLERK' in k or 'SUPABASE' in k]
    })

@app.route('/debug-lookup-match/<job_id>')
def debug_lookup_match(job_id):
    """Debug endpoint to check if a match exists in Supabase"""
    try:
        from supabase_client import get_match_by_job_id
        match = get_match_by_job_id(job_id)
        
        if match:
            # Don't return full analytics (too big), just metadata
            return jsonify({
                'found': True,
                'job_id': match.get('job_id'),
                'user_id': match.get('user_id'),
                'filename': match.get('filename'),
                'sport': match.get('sport'),
                'created_at': match.get('created_at'),
                'has_analytics': bool(match.get('analytics')),
                'analytics_keys': list(match.get('analytics', {}).keys()) if match.get('analytics') else [],
                'players_image_url': match.get('players_image_url')
            })
        else:
            return jsonify({
                'found': False,
                'job_id': job_id,
                'message': 'Match not found in Supabase. Was "Save Match" clicked while logged in?'
            })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'job_id': job_id
        }), 500

@app.route('/debug-create-test-match')
def debug_create_test_match():
    """Create a test match for debugging save functionality"""
    import uuid
    test_job_id = f"test_{uuid.uuid4().hex[:8]}"
    
    # Create test analytics
    test_analytics = {
        "sport": "squash",
        "camera_angle": "back",
        "match_info": {
            "duration_seconds": 300,
            "total_frames": 9000,
            "frames_analyzed": 8000,
            "fps": 30
        },
        "data_quality": {
            "ball_detection_rate": 12.5,
            "player_detection_rate": 88.2,
            "avg_detection_confidence": 82.1,
            "quality_score": 62.3,
            "is_reliable": True
        },
        "player1": {
            "name": "Test Player 1",
            "scramble_score": 105.2,
            "running_score": 28500.5,
            "t_dominance": 54.2,
            "attack_score": 22
        },
        "player2": {
            "name": "Test Player 2", 
            "scramble_score": 118.7,
            "running_score": 31200.3,
            "t_dominance": 45.8,
            "attack_score": 18
        },
        "analysis": {
            "t_dominance": {"summary": "Test Player 1 controlled the T."},
            "scramble": {"summary": "Test Player 2 was pushed more."}
        }
    }
    
    # Create results directory and save files
    result_dir = RESULTS_FOLDER / test_job_id
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Save analytics JSON
    import json
    analytics_path = result_dir / f"{test_job_id}_squash_analytics.json"
    with open(analytics_path, 'w') as f:
        json.dump(test_analytics, f, indent=2)
    
    # Save compact JSON
    compact_path = result_dir / f"{test_job_id}_compact_gemini.json"
    with open(compact_path, 'w') as f:
        json.dump({"video_name": test_job_id, "fps": 30, "duration_seconds": 300}, f)
    
    # Store in memory jobs dict
    jobs[test_job_id] = {
        'job_id': test_job_id,
        'status': 'complete',
        'filename': 'test_match.mp4',
        'sport': 'squash',
        'result': {
            'squash_analytics': test_analytics
        }
    }
    
    return jsonify({
        'success': True,
        'job_id': test_job_id,
        'results_url': f'/results/{test_job_id}',
        'message': f'Test match created. Go to /results/{test_job_id} to test saving.'
    })

@app.route('/')
def index():
    """Main upload page"""
    clerk_key = os.environ.get('CLERK_PUBLISHABLE_KEY', '')
    print(f"[DEBUG] CLERK_PUBLISHABLE_KEY: {'SET' if clerk_key else 'NOT SET'}", flush=True)
    return render_template('upload.html', 
        clerk_publishable_key=clerk_key
    )

@app.route('/my-matches')
def my_matches_page():
    """User's saved matches page - requires login"""
    return render_template('my_matches.html',
        clerk_publishable_key=os.environ.get('CLERK_PUBLISHABLE_KEY', '')
    )

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video file upload"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Get sport type from form data (default to squash)
    sport = request.form.get('sport', 'squash')
    valid_sports = ['squash', 'padel', 'tennis', 'table_tennis']
    if sport not in valid_sports:
        sport = 'squash'
    
    # Check if user is logged in (for auto-save after processing)
    user_id = None
    try:
        from auth import get_current_user
        user = get_current_user()
        if user:
            user_id = user.get('id')
    except Exception as e:
        print(f"Could not get user for auto-save: {e}")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    upload_path = UPLOAD_FOLDER / f"{job_id}_{filename}"
    file.save(str(upload_path))
    
    # Get file size
    file_size = os.path.getsize(upload_path)
    
    # Create job entry
    with jobs_lock:
        jobs[job_id] = {
            'id': job_id,
            'filename': filename,
            'file_size': file_size,
            'sport': sport,
            'status': 'queued',
            'progress': 0,
            'message': 'Video uploaded, waiting to process...',
            'created_at': datetime.now().isoformat(),
            'video_path': str(upload_path),
            'user_id': user_id  # Store user_id for auto-save
        }
    
    # Start processing in background thread (pass sport parameter)
    thread = threading.Thread(target=process_video_task, args=(job_id, str(upload_path), sport))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'sport': sport,
        'message': 'Video uploaded successfully! Processing started.'
    })


# ============================================================================
# Direct-to-Cloud Upload API (Scalable)
# ============================================================================

@app.route('/api/upload/init', methods=['POST'])
def init_direct_upload():
    """
    Initialize a direct upload to Supabase Storage.
    Returns a signed URL for the client to upload directly.
    """
    try:
        data = request.get_json() or {}
        filename = data.get('filename', '')
        sport = data.get('sport', 'squash')
        file_size = data.get('file_size', 0)
        
        if not filename:
            return jsonify({'error': 'Filename is required'}), 400
        
        # Validate file extension
        if not allowed_file(filename):
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Validate sport
        valid_sports = ['squash', 'padel', 'tennis', 'table_tennis']
        if sport not in valid_sports:
            sport = 'squash'
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())[:8]
        safe_filename = secure_filename(filename)
        
        # Get user ID if logged in
        user_id = None
        try:
            from auth import get_current_user
            user = get_current_user()
            if user:
                user_id = user.get('id')
        except Exception as e:
            print(f"Could not get user: {e}")
        
        # Create job in Supabase queue
        from supabase_client import create_job, create_signed_upload_url
        
        # Generate signed upload URL
        upload_info = create_signed_upload_url(job_id, safe_filename)
        if not upload_info:
            return jsonify({'error': 'Failed to create upload URL'}), 500
        
        # Create job record
        job = create_job(
            job_id=job_id,
            filename=safe_filename,
            sport=sport,
            user_id=user_id,
            storage_path=upload_info['path']
        )
        
        if not job:
            return jsonify({'error': 'Failed to create job'}), 500
        
        # Also create local job entry for real-time status
        with jobs_lock:
            jobs[job_id] = {
                'id': job_id,
                'filename': safe_filename,
                'file_size': file_size,
                'sport': sport,
                'status': 'uploading',
                'progress': 0,
                'message': 'Uploading video...',
                'created_at': datetime.now().isoformat(),
                'user_id': user_id,
                'storage_path': upload_info['path'],
                'upload_type': 'direct'  # Mark as direct upload
            }
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'upload_url': upload_info['signed_url'],
            'upload_token': upload_info.get('token'),
            'storage_path': upload_info['path'],
            'sport': sport
        })
        
    except Exception as e:
        print(f"Error initializing upload: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload/complete/<job_id>', methods=['POST'])
def complete_direct_upload(job_id):
    """
    Mark a direct upload as complete and queue for processing.
    Called by the client after successful upload to Supabase Storage.
    """
    try:
        # Get job from local memory or Supabase
        job = get_job(job_id)
        
        if not job:
            # Try to get from Supabase
            from supabase_client import get_job_by_job_id
            job = get_job_by_job_id(job_id)
        
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        # Update job status to queued
        from supabase_client import update_job as update_job_db
        update_job_db(job_id, status='queued', message='Video uploaded, waiting to process...')
        
        # Update local job
        update_job(job_id, status='queued', message='Video uploaded, waiting to process...')
        
        # Trigger background processing
        storage_path = job.get('storage_path')
        sport = job.get('sport', 'squash')
        
        if storage_path:
            # Start processing from cloud storage
            thread = threading.Thread(
                target=process_video_from_cloud,
                args=(job_id, storage_path, sport)
            )
            thread.daemon = True
            thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'status': 'queued',
            'message': 'Upload complete, processing started'
        })
        
    except Exception as e:
        print(f"Error completing upload: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload/status/<job_id>')
def get_direct_upload_status(job_id):
    """Get status of a direct upload job (from Supabase queue)."""
    try:
        # First check local memory
        job = get_job(job_id)
        
        if job:
            return jsonify(job)
        
        # Fall back to Supabase
        from supabase_client import get_job_by_job_id
        job = get_job_by_job_id(job_id)
        
        if not job:
            # Try loading from disk
            job = load_job_from_disk(job_id)
        
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        return jsonify(job)
        
    except Exception as e:
        print(f"Error getting upload status: {e}")
        return jsonify({'error': str(e)}), 500


def process_video_from_cloud(job_id, storage_path, sport='squash'):
    """
    Background task to process video from Supabase Storage.
    Downloads the video, processes it, and updates the job.
    """
    import traceback
    import tempfile
    
    print(f"[{job_id}] Starting cloud video processing...", flush=True)
    
    try:
        from supabase_client import (
            download_video_for_processing,
            update_job as update_job_db,
            delete_video_from_storage,
            upload_player_image
        )
        
        # Update status
        update_job(job_id, status='processing', progress=0, message='Downloading video...')
        update_job_db(job_id, status='processing', progress=0, message='Downloading video...', started_at=True)
        
        # Download video to temp file
        filename = storage_path.split('/')[-1]
        local_path = UPLOAD_FOLDER / f"{job_id}_{filename}"
        
        print(f"[{job_id}] Downloading from {storage_path} to {local_path}...", flush=True)
        
        if not download_video_for_processing(storage_path, str(local_path)):
            raise Exception("Failed to download video from storage")
        
        print(f"[{job_id}] Download complete, starting processing...", flush=True)
        
        # Now process using existing function
        from video_processor import process_video
        
        update_job(job_id, progress=5, message=f'Starting {sport} video processing...')
        update_job_db(job_id, progress=5, message=f'Starting {sport} video processing...')
        
        output_dir = RESULTS_FOLDER / job_id
        output_dir.mkdir(exist_ok=True)
        
        def progress_callback(progress, message):
            # Scale progress from 5-95 for processing phase
            scaled_progress = 5 + (progress * 0.9)
            print(f"[{job_id}] Progress: {scaled_progress:.1f}% - {message}", flush=True)
            update_job(job_id, progress=scaled_progress, message=message)
            # Update DB less frequently (every 10%)
            if int(progress) % 10 == 0:
                update_job_db(job_id, progress=scaled_progress, message=message)
        
        # Get sport-specific parameters
        from sport_model_config import get_sport_config
        sport_config = get_sport_config(sport)
        
        result = process_video(
            video_path=str(local_path),
            output_dir=str(output_dir),
            progress_callback=progress_callback,
            sport=sport,
            confidence=sport_config['confidence'],
            ball_confidence=sport_config['ball_confidence'],
            min_person_height_ratio=sport_config['min_person_height_ratio'],
            court_margin=sport_config['court_margin']
        )
        
        if result.get('success'):
            update_job(job_id, 
                status='completed', 
                progress=100, 
                message='Processing complete!',
                result=result
            )
            update_job_db(job_id,
                status='completed',
                progress=100,
                message='Processing complete!',
                analytics=result.get('analytics'),
                completed_at=True
            )
            
            # Try to upload player image and auto-save
            try:
                job = get_job(job_id)
                user_id = job.get('user_id')
                
                # Find and upload player image
                players_images = list(output_dir.glob('*_players.jpg'))
                if players_images and user_id:
                    players_image_url = upload_player_image(job_id, str(players_images[0]))
                    if players_image_url:
                        update_job_db(job_id, players_image_url=players_image_url)
                        
                        # Auto-save to matches
                        from supabase_client import save_match, check_match_saved
                        if not check_match_saved(job_id, user_id):
                            analytics = result.get('analytics', {})
                            p1_name = analytics.get('player1', {}).get('name', 'Player 1')
                            p2_name = analytics.get('player2', {}).get('name', 'Player 2')
                            
                            save_match(
                                user_id=user_id,
                                job_id=job_id,
                                filename=filename,
                                analytics=analytics,
                                sport=sport,
                                players_image_url=players_image_url,
                                player1_name=p1_name,
                                player2_name=p2_name
                            )
            except Exception as e:
                print(f"[{job_id}] Error in post-processing: {e}")
            
            # Clean up: delete video from cloud storage (keep local for serving)
            try:
                delete_video_from_storage(storage_path)
                print(f"[{job_id}] Deleted video from cloud storage")
            except Exception as e:
                print(f"[{job_id}] Failed to delete from cloud storage: {e}")
        else:
            error_msg = result.get('error', 'Processing failed')
            update_job(job_id, status='failed', message=error_msg)
            update_job_db(job_id, status='failed', error=error_msg)
            
    except Exception as e:
        error_msg = str(e)
        print(f"[{job_id}] Error processing cloud video: {error_msg}", flush=True)
        traceback.print_exc()
        update_job(job_id, status='failed', message=f'Error: {error_msg}')
        try:
            from supabase_client import update_job as update_job_db
            update_job_db(job_id, status='failed', error=error_msg)
        except:
            pass


@app.route('/status/<job_id>')
def get_status(job_id):
    """Get job status"""
    job = get_job(job_id)
    
    # If job not in memory, try to load from disk
    if not job:
        job = load_job_from_disk(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(job)

@app.route('/api/recent-jobs')
def get_recent_jobs():
    """Get list of recent completed jobs for the upload page"""
    recent_jobs = []
    
    try:
        # Scan the results folder for completed jobs
        if RESULTS_FOLDER.exists():
            # Get all directories sorted by modification time (newest first)
            dirs = sorted(RESULTS_FOLDER.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            
            for job_dir in dirs[:15]:  # Limit to 15 most recent
                if not job_dir.is_dir():
                    continue
                
                # Skip match_ directories (legacy combined matches)
                if job_dir.name.startswith('match_'):
                    continue
                    
                job_id = job_dir.name
                job_info = {
                    'id': job_id,
                    'created_at': datetime.fromtimestamp(job_dir.stat().st_mtime).isoformat()
                }
                
                # Look for analytics file
                analytics_files = list(job_dir.glob('*_squash_analytics.json'))
                if analytics_files:
                    try:
                        with open(analytics_files[0], 'r') as f:
                            analytics = json.load(f)
                        # Extract video name from filename
                        video_name = analytics_files[0].stem.replace('_squash_analytics', '')
                        job_info['name'] = video_name
                        job_info['filename'] = video_name
                        job_info['duration'] = analytics.get('match_info', {}).get('duration_seconds', 0)
                        job_info['sport'] = analytics.get('sport', 'squash')
                        
                        # Add highlight data
                        p1 = analytics.get('player1', {})
                        p2 = analytics.get('player2', {})
                        
                        total_distance = (p1.get('running_score', 0) or 0) + (p2.get('running_score', 0) or 0)
                        total_attacks = (p1.get('attack_score', 0) or 0) + (p2.get('attack_score', 0) or 0)
                        total_duration = analytics.get('match_info', {}).get('duration_seconds', 0)
                        
                        job_info['highlight'] = {
                            'total_distance': round(total_distance * 0.0334, 1),  # Convert pixels to feet
                            'total_attacks': total_attacks,
                            'total_duration': total_duration
                        }
                    except:
                        job_info['name'] = job_id
                else:
                    # Check if there's any video file to get the name
                    video_files = list(job_dir.glob('*_annotated.mp4'))
                    if video_files:
                        video_name = video_files[0].stem.replace('_annotated', '')
                        job_info['name'] = video_name
                    else:
                        continue  # Skip if no recognizable files
                
                recent_jobs.append(job_info)
        
        return jsonify({'jobs': recent_jobs})
        
    except Exception as e:
        print(f"Error getting recent jobs: {e}")
        return jsonify({'jobs': [], 'error': str(e)})

@app.route('/results/<job_id>')
def results_page(job_id):
    """Results page for a specific job"""
    job = get_job(job_id)
    
    # If job not in memory, try to load from disk
    if not job:
        job = load_job_from_disk(job_id)
    
    # If still not found, try to load from Supabase (for saved matches)
    from_supabase = False
    if not job:
        try:
            from supabase_client import get_match_by_job_id
            match = get_match_by_job_id(job_id)
            if match:
                # Reconstruct job from Supabase data
                analytics = match.get('analytics', {})
                job = {
                    'id': job_id,
                    'filename': match.get('filename', job_id),
                    'status': 'completed',
                    'result': {'squash_analytics': analytics},
                    'from_supabase': True,
                    'players_image_url': match.get('players_image_url')
                }
                from_supabase = True
        except Exception as e:
            print(f"Error loading from Supabase: {e}")
    
    if not job:
        return "Job not found", 404
    
    # Get metadata for OG tags (social sharing previews)
    video_name = job.get('filename', job_id)
    # Clean up video name for display
    clean_name = video_name.replace('_', ' ').replace(f"{job_id} ", "")
    
    # Find the players image for OG preview
    og_image = None
    if from_supabase and job.get('players_image_url'):
        og_image = job.get('players_image_url')
    else:
        result_dir = RESULTS_FOLDER / job_id
        players_images = list(result_dir.glob("*_players.jpg"))
        og_image = f"/results/{job_id}/files/{players_images[0].name}" if players_images else None
    
    return render_template('results.html', 
        job_id=job_id,
        og_title=clean_name,
        og_image=og_image,
        og_description="Squash match analysis powered by AI",
        clerk_publishable_key=os.environ.get('CLERK_PUBLISHABLE_KEY', '')
    )

def load_job_from_disk(job_id):
    """Try to load a completed job from disk"""
    result_dir = RESULTS_FOLDER / job_id
    
    if not result_dir.exists():
        return None
    
    # Find the analytics file
    analytics_files = list(result_dir.glob("*_squash_analytics.json"))
    compact_files = list(result_dir.glob("*_compact_gemini.json"))
    
    if not compact_files:
        return None
    
    try:
        # Load compact data
        with open(compact_files[0], 'r') as f:
            compact = json.load(f)
        
        # Load squash analytics if available
        squash_analytics = None
        if analytics_files:
            with open(analytics_files[0], 'r') as f:
                squash_analytics = json.load(f)
        
        # Get video name from filename
        video_name = compact_files[0].stem.replace('_compact_gemini', '')
        
        # Reconstruct the job
        result = {
            "success": True,
            "video_name": video_name.replace(f"{job_id}_", ""),
            "total_frames": compact.get("total_frames", 0),
            "fps": compact.get("fps", 30),
            "duration_seconds": compact.get("duration_seconds", 0),
            "processing_time": compact.get("processing_time_seconds", 0),
            "fps_processed": compact.get("total_frames", 0) / max(compact.get("processing_time_seconds", 1), 1),
            "total_detections": sum(compact.get("detection_summary", {}).values()),
            "detection_summary": compact.get("detection_summary", {}),
            "squash_analytics": squash_analytics
        }
        
        job = {
            'id': job_id,
            'filename': video_name,
            'status': 'completed',
            'progress': 100,
            'message': 'Loaded from disk',
            'result': result
        }
        
        # Store in memory for future requests
        with jobs_lock:
            jobs[job_id] = job
        
        return job
    except Exception as e:
        print(f"Error loading job {job_id} from disk: {e}")
        return None

@app.route('/results/<job_id>/files/<path:filename>')
def serve_result_file(job_id, filename):
    """Serve result files with range request support for video streaming"""
    result_dir = RESULTS_FOLDER / job_id
    file_path = result_dir / filename
    
    # For video files, handle range requests for proper streaming
    if filename.endswith('.mp4'):
        if not file_path.exists():
            return "File not found", 404
        
        file_size = file_path.stat().st_size
        
        # Check for range request
        range_header = request.headers.get('Range')
        
        if range_header:
            # Parse range header
            byte_start = 0
            byte_end = file_size - 1
            
            match = re.match(r'bytes=(\d+)-(\d*)', range_header)
            if match:
                byte_start = int(match.group(1))
                if match.group(2):
                    byte_end = int(match.group(2))
            
            # Limit chunk size to 10MB for smoother streaming
            chunk_size = 10 * 1024 * 1024  # 10MB
            if byte_end - byte_start > chunk_size:
                byte_end = byte_start + chunk_size
            
            content_length = byte_end - byte_start + 1
            
            def generate():
                with open(file_path, 'rb') as f:
                    f.seek(byte_start)
                    remaining = content_length
                    while remaining > 0:
                        chunk = f.read(min(8192, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk
            
            response = Response(
                generate(),
                status=206,
                mimetype='video/mp4',
                direct_passthrough=True
            )
            response.headers['Content-Range'] = f'bytes {byte_start}-{byte_end}/{file_size}'
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Content-Length'] = content_length
            return response
        else:
            # No range request - return full file info but let browser request ranges
            response = send_from_directory(str(result_dir), filename, mimetype='video/mp4')
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Content-Type'] = 'video/mp4'
            response.headers['Cache-Control'] = 'public, max-age=3600'
            return response
    
    # For non-video files, use standard send_from_directory
    return send_from_directory(str(result_dir), filename)

@app.route('/api/jobs')
def list_jobs():
    """List all jobs (for admin)"""
    with jobs_lock:
        return jsonify(list(jobs.values()))

@app.route('/api/results/<job_id>')
def get_results_api(job_id):
    """API endpoint to get results data for a job"""
    # Handle old combined match IDs gracefully
    if job_id.startswith('match_'):
        return jsonify({
            'error': 'Combined matches are no longer supported. Please upload individual game videos.',
            'legacy_match': True
        }), 410  # 410 Gone
    
    job = get_job(job_id)
    if not job:
        job = load_job_from_disk(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify({
        'job': {
            'id': job_id,
            'filename': job.get('filename', ''),
            'status': job.get('status', 'unknown'),
            'sport': job.get('sport', 'squash')
        },
        'result': job.get('result', {})
    })

@app.route('/stream/<job_id>')
def stream_progress(job_id):
    """Server-sent events for real-time progress"""
    def generate():
        last_progress = -1
        while True:
            job = get_job(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break
            
            if job.get('progress', 0) != last_progress or job.get('status') in ['completed', 'failed']:
                last_progress = job.get('progress', 0)
                yield f"data: {json.dumps(job)}\n\n"
            
            if job.get('status') in ['completed', 'failed']:
                break
            
            time.sleep(1)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/chat/<job_id>', methods=['POST'])
def chat_about_match(job_id):
    """Chat endpoint for asking questions about the match"""
    job = get_job(job_id)
    
    # If not in memory, try to load from disk
    if not job:
        job = load_job_from_disk(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    if job.get('status') != 'completed':
        return jsonify({'error': 'Analysis not yet complete'}), 400
    
    data = request.get_json()
    user_message = data.get('message', '')
    player_names = data.get('player_names', {})
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Get analytics data
    result = job.get('result', {})
    analytics = result.get('squash_analytics', {})
    
    if not analytics or analytics.get('error'):
        return jsonify({'error': 'No analytics data available'}), 400
    
    # Try to use Gemini API if available
    try:
        import google.generativeai as genai
        
        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            p1_name = player_names.get('player1', 'Player 1')
            p2_name = player_names.get('player2', 'Player 2')
            p1 = analytics.get('player1', {})
            p2 = analytics.get('player2', {})
            analysis = analytics.get('analysis', {})
            
            # Build context for the match
            context = f"""You are a squash match analyst assistant. Here is the match data:

Match Duration: {analytics.get('match_info', {}).get('duration_seconds', 0):.1f} seconds
Frames Analyzed: {analytics.get('match_info', {}).get('frames_analyzed', 0)}

{p1_name} Stats:
- T-Dominance: {p1.get('t_dominance', 0)}%
- Scramble Score (avg distance from T): {p1.get('scramble_score', 0)} px
- Running Score (total distance): {p1.get('running_score', 0)} px
- Attack Score: {p1.get('attack_score', 0)} high-speed shots
- Avg Rail Distance from Wall: {p1.get('avg_rail_distance', 0)} px
- Tight Rail Count: {p1.get('tight_rail_count', 0)}

{p2_name} Stats:
- T-Dominance: {p2.get('t_dominance', 0)}%
- Scramble Score (avg distance from T): {p2.get('scramble_score', 0)} px
- Running Score (total distance): {p2.get('running_score', 0)} px
- Attack Score: {p2.get('attack_score', 0)} high-speed shots
- Avg Rail Distance from Wall: {p2.get('avg_rail_distance', 0)} px
- Tight Rail Count: {p2.get('tight_rail_count', 0)}

Analysis Summary:
- T-Dominance: {analysis.get('t_dominance', {}).get('summary', 'N/A')}
- Scramble: {analysis.get('scramble', {}).get('summary', 'N/A')}
- Running: {analysis.get('running', {}).get('summary', 'N/A')}
- Attack: {analysis.get('attack', {}).get('summary', 'N/A')}
- Tight Rails: {analysis.get('tight_rails', {}).get('summary', 'N/A')}

Answer the user's question about this match. Be specific and reference the data when relevant. Keep responses concise but informative.

User Question: {user_message}"""

            response = model.generate_content(context)
            return jsonify({
                'response': response.text,
                'source': 'gemini'
            })
    except Exception as e:
        print(f"Gemini API error: {e}")
    
    # Fallback: Generate response from analytics data
    response = generate_fallback_response(user_message, analytics, player_names)
    return jsonify({
        'response': response,
        'source': 'local'
    })

def generate_fallback_response(question, analytics, player_names):
    """Generate a response without LLM API"""
    question_lower = question.lower()
    p1_name = player_names.get('player1', 'Player 1')
    p2_name = player_names.get('player2', 'Player 2')
    p1 = analytics.get('player1', {})
    p2 = analytics.get('player2', {})
    analysis = analytics.get('analysis', {})
    
    # Get T-dominance (field name differs between single and combined)
    p1_t_dom = p1.get('avg_t_dominance') or p1.get('t_dominance', 0)
    p2_t_dom = p2.get('avg_t_dominance') or p2.get('t_dominance', 0)
    p1_scramble = p1.get('avg_scramble_score') or p1.get('scramble_score', 0)
    p2_scramble = p2.get('avg_scramble_score') or p2.get('scramble_score', 0)
    p1_running = p1.get('total_running_score') or p1.get('running_score', 0)
    p2_running = p2.get('total_running_score') or p2.get('running_score', 0)
    p1_attacks = p1.get('total_attack_score') or p1.get('attack_score', 0)
    p2_attacks = p2.get('total_attack_score') or p2.get('attack_score', 0)
    
    # Check for common questions
    if 'win' in question_lower or 'better' in question_lower or 'ahead' in question_lower:
        if p1_t_dom > p2_t_dom:
            return f"Based on the analytics, {p1_name} appears to have the advantage with {p1_t_dom}% T-dominance compared to {p2_name}'s {p2_t_dom}%. {p1_name} is controlling the center of the court more effectively."
        else:
            return f"Based on the analytics, {p2_name} appears to have the advantage with {p2_t_dom}% T-dominance compared to {p1_name}'s {p1_t_dom}%. {p2_name} is controlling the center of the court more effectively."
    
    if 't-dom' in question_lower or 't dom' in question_lower or 'center' in question_lower or 'control' in question_lower:
        return f"T-Dominance shows who controls the center court. {p1_name}: {p1_t_dom}%, {p2_name}: {p2_t_dom}%. {analysis.get('t_dominance', {}).get('summary', '')}"
    
    if 'scrambl' in question_lower or 'pressure' in question_lower:
        return f"Scramble Score measures average distance from the T (lower is better). {p1_name}: {p1_scramble:.0f} px, {p2_name}: {p2_scramble:.0f} px. {analysis.get('scramble', {}).get('summary', '')}"
    
    if 'run' in question_lower or 'distance' in question_lower or 'fitness' in question_lower:
        return f"Running Score shows total distance covered. {p1_name}: {p1_running:.0f} px, {p2_name}: {p2_running:.0f} px. {analysis.get('running', {}).get('summary', '')}"
    
    if 'attack' in question_lower or 'aggress' in question_lower:
        return f"Attack Score counts high-velocity movements. {p1_name}: {p1_attacks}, {p2_name}: {p2_attacks}. {analysis.get('attack', {}).get('summary', '')}"
    
    if 'rail' in question_lower or 'wall' in question_lower or 'tight' in question_lower:
        p1_rail = p1.get('avg_rail_distance', 0)
        p2_rail = p2.get('avg_rail_distance', 0)
        p1_tight = p1.get('total_tight_rails') or p1.get('tight_rail_count', 0)
        p2_tight = p2.get('total_tight_rails') or p2.get('tight_rail_count', 0)
        return f"Tight Rails analysis shows ball proximity to walls. {p1_name}: avg {p1_rail} px from wall ({p1_tight} tight shots), {p2_name}: avg {p2_rail} px from wall ({p2_tight} tight shots). {analysis.get('tight_rails', {}).get('summary', '')}"
    
    if 'zone' in question_lower or 'court' in question_lower or 'position' in question_lower or 'heat' in question_lower:
        zone = analytics.get('zone_analysis', {})
        p1_zone = zone.get('player1', {}).get('aggregate', {})
        p2_zone = zone.get('player2', {}).get('aggregate', {})
        return f"Zone Analysis: {p1_name} spent {p1_zone.get('front_court', 0)}% front / {p1_zone.get('mid_court', 0)}% mid / {p1_zone.get('back_court', 0)}% back. {p2_name}: {p2_zone.get('front_court', 0)}% front / {p2_zone.get('mid_court', 0)}% mid / {p2_zone.get('back_court', 0)}% back. {zone.get('analysis', '')}"
    
    if 'decay' in question_lower or 'fatigue' in question_lower or 'tired' in question_lower or 'fitness' in question_lower:
        decay = analytics.get('performance_decay', {})
        return f"Performance Analysis: {decay.get('analysis', 'No decay data available.')}"
    
    # Default summary
    return f"""Here's a summary of the match:

**{p1_name}**: T-Dom {p1_t_dom}%, Attack {p1_attacks}, Scramble {p1_scramble:.0f} px

**{p2_name}**: T-Dom {p2_t_dom}%, Attack {p2_attacks}, Scramble {p2_scramble:.0f} px

Try asking about specific metrics like "Who controlled the T better?" or "Who hit tighter rails?" """

@app.route('/api/update-players/<job_id>', methods=['POST'])
def update_player_names(job_id):
    """Update player names for a job"""
    job = get_job(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    data = request.get_json()
    player1_name = data.get('player1', 'Player 1')
    player2_name = data.get('player2', 'Player 2')
    
    with jobs_lock:
        if job_id in jobs:
            if 'player_names' not in jobs[job_id]:
                jobs[job_id]['player_names'] = {}
            jobs[job_id]['player_names']['player1'] = player1_name
            jobs[job_id]['player_names']['player2'] = player2_name
    
    return jsonify({'success': True})


@app.route('/match/<match_id>')
def match_results_redirect(match_id):
    """Handle old combined match URLs - redirect to first game or show message"""
    from flask import redirect
    
    # Try to find the first game from the match_id pattern (match_xxxx_yyyy -> xxxx)
    if match_id.startswith('match_'):
        # Extract first job prefix from match_id (e.g., "match_abd7_19bd" -> "abd7")
        prefixes = match_id.replace('match_', '').split('_')
        if prefixes:
            first_prefix = prefixes[0]
            # Find a job directory that starts with this prefix
            for result_dir in RESULTS_FOLDER.iterdir():
                if result_dir.is_dir() and result_dir.name.startswith(first_prefix) and not result_dir.name.startswith('match_'):
                    return redirect(f'/results/{result_dir.name}', code=301)
    
    # Fallback: redirect to home with a message
    return redirect('/?error=combined_matches_removed', code=302)


# ============================================================================
# Authentication & User Management API Routes
# ============================================================================

@app.route('/api/user')
def get_current_user_api():
    """Get current authenticated user info"""
    try:
        from auth import get_current_user
        user = get_current_user()
        
        if not user:
            return jsonify({'authenticated': False, 'user': None})
        
        return jsonify({
            'authenticated': True,
            'user': {
                'id': user.get('id'),
                'clerk_id': user.get('clerk_id'),
                'email': user.get('email'),
                'name': user.get('name'),
                'image_url': user.get('image_url')
            }
        })
    except Exception as e:
        print(f"Error getting current user: {e}")
        return jsonify({'authenticated': False, 'user': None, 'error': str(e)})


@app.route('/api/my-matches')
def get_my_matches():
    """Get current user's saved matches"""
    try:
        from auth import get_current_user
        from supabase_client import get_user_matches
        
        user = get_current_user()
        
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        matches = get_user_matches(user['id'])
        
        # Filter out old combined matches (job_id starting with 'match_')
        # These are no longer supported after architecture simplification
        matches = [m for m in matches if not m.get('job_id', '').startswith('match_')]
        
        return jsonify({
            'matches': matches
        })
    except Exception as e:
        print(f"Error getting user matches: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/save-match/<job_id>', methods=['POST'])
def save_match_api(job_id):
    """Save a match to the current user's account"""
    try:
        from auth import get_current_user
        from supabase_client import save_match, upload_player_image, check_match_saved
        
        user = get_current_user()
        
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        # Check if already saved
        if check_match_saved(job_id, user['id']):
            return jsonify({'success': True, 'message': 'Match already saved'})
        
        # Load analytics from disk
        job = get_job(job_id)
        if not job:
            job = load_job_from_disk(job_id)
        
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        result = job.get('result', {})
        analytics = result.get('squash_analytics', {})
        
        if not analytics:
            return jsonify({'error': 'No analytics data found'}), 400
        
        # Upload player image to Supabase Storage
        players_image_url = None
        result_dir = RESULTS_FOLDER / job_id
        players_images = list(result_dir.glob("*_players.jpg"))
        
        if players_images:
            players_image_url = upload_player_image(job_id, str(players_images[0]))
        
        # Get player names from analytics
        player1_name = analytics.get('player1', {}).get('name')
        player2_name = analytics.get('player2', {}).get('name')
        
        # Save to Supabase
        saved_match = save_match(
            user_id=user['id'],
            job_id=job_id,
            filename=job.get('filename', job_id),
            analytics=analytics,
            sport=job.get('sport', 'squash'),
            players_image_url=players_image_url,
            player1_name=player1_name,
            player2_name=player2_name
        )
        
        return jsonify({
            'success': True,
            'match_id': saved_match.get('id') if saved_match else None
        })
        
    except Exception as e:
        print(f"Error saving match: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Return detailed error for debugging
        return jsonify({'error': f'Save failed: {str(e)}', 'details': traceback.format_exc()}), 500
        return jsonify({'error': str(e)}), 500


@app.route('/api/matches/<match_id>', methods=['DELETE'])
def delete_match_api(match_id):
    """Delete a saved match"""
    try:
        from auth import get_current_user
        from supabase_client import delete_match
        
        user = get_current_user()
        
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        
        success = delete_match(match_id, user['id'])
        
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Match not found or access denied'}), 404
            
    except Exception as e:
        print(f"Error deleting match: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/check-saved/<job_id>')
def check_match_saved_api(job_id):
    """Check if a match is saved for the current user"""
    try:
        from auth import get_current_user
        from supabase_client import check_match_saved
        
        user = get_current_user()
        
        if not user:
            return jsonify({'saved': False, 'authenticated': False})
        
        is_saved = check_match_saved(job_id, user['id'])
        
        return jsonify({
            'saved': is_saved,
            'authenticated': True
        })
        
    except Exception as e:
        print(f"Error checking saved status: {e}")
        return jsonify({'saved': False, 'error': str(e)})


# ============================================================================
# Background Queue Worker (Optional - for processing stale jobs)
# ============================================================================

_queue_worker_running = False

def start_queue_worker():
    """
    Start a background worker that polls for queued jobs.
    This catches any jobs that failed to start processing.
    """
    global _queue_worker_running
    
    if _queue_worker_running:
        return
    
    _queue_worker_running = True
    
    def worker():
        print("[Queue Worker] Started")
        while _queue_worker_running:
            try:
                from supabase_client import get_next_queued_job, update_job as update_job_db
                
                job = get_next_queued_job()
                
                if job:
                    job_id = job.get('job_id')
                    storage_path = job.get('storage_path')
                    sport = job.get('sport', 'squash')
                    
                    print(f"[Queue Worker] Found queued job: {job_id}")
                    
                    # Mark as processing to prevent re-pickup
                    update_job_db(job_id, status='processing', message='Starting processing...')
                    
                    # Process the video
                    if storage_path:
                        process_video_from_cloud(job_id, storage_path, sport)
                    else:
                        print(f"[Queue Worker] Job {job_id} has no storage_path, skipping")
                        update_job_db(job_id, status='failed', error='No storage path')
                
            except Exception as e:
                print(f"[Queue Worker] Error: {e}")
            
            # Poll every 30 seconds
            time.sleep(30)
    
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


def stop_queue_worker():
    """Stop the background queue worker."""
    global _queue_worker_running
    _queue_worker_running = False


if __name__ == '__main__':
    # Create template directory
    os.makedirs('templates/customer', exist_ok=True)
    
    # Get port from environment (Railway/Render/Heroku set this) or default to 5001
    port = int(os.environ.get('PORT', 5001))
    
    # Detect if running in production (Railway sets PORT)
    is_production = 'PORT' in os.environ or 'RAILWAY_ENVIRONMENT' in os.environ
    debug = not is_production and os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    
    print("🎬 Video Analysis Customer Portal")
    print("=" * 50)
    print(f"Environment: {'PRODUCTION' if is_production else 'DEVELOPMENT'}")
    print(f"Debug mode: {debug}")
    
    # Start background queue worker in production
    if is_production:
        print("Starting background queue worker...")
        start_queue_worker()
    
    print("Starting server...")
    print(f"Open your browser to: http://localhost:{port}")
    print("=" * 50, flush=True)
    
    # In production, disable reloader to prevent threading issues
    app.run(debug=debug, host='0.0.0.0', port=port, threaded=True, use_reloader=not is_production)

