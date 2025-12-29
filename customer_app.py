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

@app.route('/')
def index():
    """Main upload page"""
    return render_template('upload.html', 
        clerk_publishable_key=os.environ.get('CLERK_PUBLISHABLE_KEY', '')
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
                    
                job_id = job_dir.name
                job_info = {
                    'id': job_id,
                    'created_at': datetime.fromtimestamp(job_dir.stat().st_mtime).isoformat()
                }
                
                # Check if it's a combined match
                combined_file = job_dir / 'combined_match_analytics.json'
                if combined_file.exists():
                    try:
                        with open(combined_file, 'r') as f:
                            combined_data = json.load(f)
                        job_info['type'] = 'combined_match'
                        job_info['name'] = f"Match ({combined_data.get('total_games', '?')} games)"
                        job_info['duration'] = combined_data.get('total_duration_seconds', 0)
                        job_info['sport'] = combined_data.get('sport', 'squash')
                        
                        # Add highlight data for combined match
                        p1 = combined_data.get('player1', {})
                        p2 = combined_data.get('player2', {})
                        
                        # Calculate match-level totals (not player-specific since players aren't named yet)
                        total_distance = (p1.get('total_running_score', 0) or 0) + (p2.get('total_running_score', 0) or 0)
                        total_attacks = (p1.get('total_attack_score', 0) or 0) + (p2.get('total_attack_score', 0) or 0)
                        total_duration = combined_data.get('total_duration_seconds', 0)
                        
                        job_info['highlight'] = {
                            'total_distance': round(total_distance * 0.0334, 1),  # Convert pixels to feet
                            'total_attacks': total_attacks,
                            'total_duration': total_duration,
                            'games': combined_data.get('total_games', 0)
                        }
                    except:
                        job_info['type'] = 'combined_match'
                        job_info['name'] = 'Combined Match'
                else:
                    # Single game - look for analytics file
                    analytics_files = list(job_dir.glob('*_squash_analytics.json'))
                    if analytics_files:
                        try:
                            with open(analytics_files[0], 'r') as f:
                                analytics = json.load(f)
                            job_info['type'] = 'single_game'
                            # Extract video name from filename
                            video_name = analytics_files[0].stem.replace('_squash_analytics', '')
                            job_info['name'] = video_name
                            job_info['filename'] = video_name
                            job_info['duration'] = analytics.get('match_info', {}).get('duration_seconds', 0)
                            job_info['sport'] = analytics.get('sport', 'squash')
                            
                            # Add highlight data - match-level totals (not player-specific since players aren't named yet)
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
                            job_info['type'] = 'single_game'
                            job_info['name'] = job_id
                    else:
                        # Check if there's any video file to get the name
                        video_files = list(job_dir.glob('*_annotated.mp4'))
                        if video_files:
                            video_name = video_files[0].stem.replace('_annotated', '')
                            job_info['type'] = 'single_game'
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
    # Check if it's a combined match
    if job_id.startswith('match_'):
        match_dir = RESULTS_FOLDER / job_id
        if match_dir.exists():
            combined_file = match_dir / 'combined_match_analytics.json'
            if combined_file.exists():
                with open(combined_file, 'r') as f:
                    combined = json.load(f)
                return jsonify({
                    'job': {
                        'id': job_id,
                        'type': 'combined_match',
                        'status': 'completed',
                        'sport': combined.get('sport', 'squash')
                    },
                    'result': {
                        'squash_analytics': combined
                    }
                })
        return jsonify({'error': 'Match not found'}), 404
    
    # Single game
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
        if job_id.startswith('match_'):
            # Load combined match from disk
            match_dir = RESULTS_FOLDER / job_id
            if match_dir.exists():
                combined_file = match_dir / 'combined_match_analytics.json'
                if combined_file.exists():
                    with open(combined_file, 'r') as f:
                        combined = json.load(f)
                    job = {
                        'id': job_id,
                        'type': 'combined_match',
                        'status': 'completed',
                        'result': {'combined_analytics': combined}
                    }
        else:
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
    
    # Get analytics data - handle both single game and combined match
    result = job.get('result', {})
    is_combined = job.get('type') == 'combined_match' or job_id.startswith('match_')
    
    if is_combined:
        analytics = result.get('combined_analytics', {})
    else:
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
            
            if is_combined:
                # Build context for combined match
                match_info = analytics.get('match_info', {})
                match_result = analytics.get('match_result', {})
                zone = analytics.get('zone_analysis', {})
                decay = analytics.get('performance_decay', {})
                
                context = f"""You are a squash match analyst assistant. Here is the COMBINED MATCH data across {analytics.get('num_games', 0)} games:

Match Result: {match_result.get('winner', 'Unknown')} wins {match_result.get('score', 'N/A')}
Total Duration: {match_info.get('total_duration_formatted', 'N/A')}
Games Played: {match_info.get('games_played', 0)}

{p1_name} Stats:
- Games Won: {p1.get('games_won', 0)}
- Avg T-Dominance: {p1.get('avg_t_dominance', 0)}%
- Avg Scramble Score: {p1.get('avg_scramble_score', 0)} px
- Total Running Score: {p1.get('total_running_score', 0)} px
- Total Attacks: {p1.get('total_attack_score', 0)}
- Avg Rail Distance: {p1.get('avg_rail_distance', 0)} px
- Performance Trend: {p1.get('t_dominance_trend', 0):.1f}% change

{p2_name} Stats:
- Games Won: {p2.get('games_won', 0)}
- Avg T-Dominance: {p2.get('avg_t_dominance', 0)}%
- Avg Scramble Score: {p2.get('avg_scramble_score', 0)} px
- Total Running Score: {p2.get('total_running_score', 0)} px
- Total Attacks: {p2.get('total_attack_score', 0)}
- Avg Rail Distance: {p2.get('avg_rail_distance', 0)} px
- Performance Trend: {p2.get('t_dominance_trend', 0):.1f}% change

Zone Analysis:
- {p1_name} Front/Mid/Back: {zone.get('player1', {}).get('aggregate', {}).get('front_court', 0)}% / {zone.get('player1', {}).get('aggregate', {}).get('mid_court', 0)}% / {zone.get('player1', {}).get('aggregate', {}).get('back_court', 0)}%
- {p2_name} Front/Mid/Back: {zone.get('player2', {}).get('aggregate', {}).get('front_court', 0)}% / {zone.get('player2', {}).get('aggregate', {}).get('mid_court', 0)}% / {zone.get('player2', {}).get('aggregate', {}).get('back_court', 0)}%

Performance Decay Analysis: {decay.get('analysis', 'N/A')}

Match Summary: {analysis.get('match_summary', 'N/A')}

Answer the user's question about this match. Be specific and reference the data when relevant. Keep responses concise but informative.

User Question: {user_message}"""
            else:
                # Build context for single game
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
    response = generate_fallback_response(user_message, analytics, player_names, is_combined)
    return jsonify({
        'response': response,
        'source': 'local'
    })

def generate_fallback_response(question, analytics, player_names, is_combined=False):
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
        if is_combined:
            match_result = analytics.get('match_result', {})
            winner = match_result.get('winner', 'Unknown')
            score = match_result.get('score', '')
            return f"**{winner}** won the match **{score}** based on T-control across all games. {p1_name} had avg T-dominance of {p1_t_dom}% vs {p2_name}'s {p2_t_dom}%."
        else:
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
    if is_combined:
        match_result = analytics.get('match_result', {})
        return f"""**Match Summary ({analytics.get('num_games', 0)} games)**:

Winner: **{match_result.get('winner', 'Unknown')}** ({match_result.get('score', '')})

**{p1_name}**: Avg T-Dom {p1_t_dom}%, Attacks {p1_attacks}, Games Won {p1.get('games_won', 0)}

**{p2_name}**: Avg T-Dom {p2_t_dom}%, Attacks {p2_attacks}, Games Won {p2.get('games_won', 0)}

Try asking about specific metrics like "Who controlled the T better?", "Zone analysis?", or "Performance decay?" """
    else:
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


@app.route('/api/combine-match', methods=['POST'])
def combine_match():
    """Combine multiple game analyses into a single match"""
    from squash_analytics import combine_match_analytics
    
    data = request.get_json()
    job_ids = data.get('job_ids', [])
    game_names = data.get('game_names', None)
    
    if not job_ids or len(job_ids) < 1:
        return jsonify({'error': 'At least 1 game job ID required'}), 400
    
    # Load analytics for each game
    game_analytics = []
    game_info = []
    
    for job_id in job_ids:
        job = get_job(job_id)
        if not job:
            job = load_job_from_disk(job_id)
        
        if not job:
            return jsonify({'error': f'Job {job_id} not found'}), 404
        
        if job.get('status') != 'completed':
            return jsonify({'error': f'Job {job_id} not yet complete'}), 400
        
        analytics = job.get('result', {}).get('squash_analytics')
        if not analytics:
            return jsonify({'error': f'No analytics for job {job_id}'}), 400
        
        game_analytics.append(analytics)
        
        # Get video name for file references
        result = job.get('result', {})
        video_name = result.get('video_name', job.get('filename', job_id))
        
        game_info.append({
            'job_id': job_id,
            'filename': job.get('filename', job_id),
            'video_name': video_name
        })
    
    # Generate default game names if not provided
    if not game_names:
        game_names = [f"Game {i+1}" for i in range(len(job_ids))]
    
    # Combine analytics
    combined = combine_match_analytics(game_analytics, game_names)
    
    # Add game job references to combined analytics
    combined['game_jobs'] = game_info
    
    # Create a new "match" job
    match_id = 'match_' + '_'.join([jid[:4] for jid in job_ids])
    
    # Save combined analytics
    match_dir = RESULTS_FOLDER / match_id
    match_dir.mkdir(exist_ok=True)
    
    combined_file = match_dir / 'combined_match_analytics.json'
    with open(combined_file, 'w') as f:
        json.dump(combined, f, indent=2)
    
    # Store the combined match info
    match_job = {
        'id': match_id,
        'type': 'combined_match',
        'game_jobs': game_info,
        'job_ids': job_ids,
        'status': 'completed',
        'progress': 100,
        'message': 'Match combined successfully',
        'result': {
            'success': True,
            'combined_analytics': combined
        },
        'created_at': datetime.now().isoformat()
    }
    
    with jobs_lock:
        jobs[match_id] = match_job
    
    return jsonify({
        'success': True,
        'match_id': match_id,
        'combined_analytics': combined
    })


@app.route('/match/<match_id>')
def match_results_page(match_id):
    """Combined match results page - uses unified results template"""
    job = get_job(match_id)
    
    if not job:
        # Try to load from disk
        match_dir = RESULTS_FOLDER / match_id
        if match_dir.exists():
            combined_file = match_dir / 'combined_match_analytics.json'
            if combined_file.exists():
                with open(combined_file, 'r') as f:
                    combined = json.load(f)
                
                # Reconstruct game_jobs if missing (for older matches)
                if 'game_jobs' not in combined or not combined['game_jobs']:
                    if match_id.startswith('match_'):
                        prefixes = match_id.replace('match_', '').split('_')
                        game_jobs = []
                        for prefix in prefixes:
                            # Find directories that start with this prefix
                            for result_dir in RESULTS_FOLDER.iterdir():
                                if result_dir.is_dir() and result_dir.name.startswith(prefix) and not result_dir.name.startswith('match_'):
                                    # Try to find a players.jpg file
                                    for file in result_dir.glob('*_players.jpg'):
                                        video_name = file.stem.replace('_players', '').replace(f'{result_dir.name}_', '')
                                        game_jobs.append({
                                            'job_id': result_dir.name,
                                            'filename': video_name,
                                            'video_name': video_name
                                        })
                                        break
                                    if game_jobs:  # Found at least one, break to next prefix
                                        break
                        if game_jobs:
                            combined['game_jobs'] = game_jobs
                
                job = {
                    'id': match_id,
                    'type': 'combined_match',
                    'status': 'completed',
                    'result': {'combined_analytics': combined}
                }
                # Store in memory for future requests
                with jobs_lock:
                    jobs[match_id] = job
    
    if not job:
        return "Match not found", 404
    
    # Get metadata for OG tags (social sharing previews)
    og_title = "Match Analysis"
    og_image = None
    
    # Try to get match name and player image from combined analytics or first game
    if job.get('type') == 'combined_match':
        combined = job.get('result', {}).get('combined_analytics', {})
        game_jobs = combined.get('game_jobs', [])
        if game_jobs:
            # Use first game's info
            first_game = game_jobs[0]
            first_job_id = first_game.get('job_id', '')
            video_name = first_game.get('video_name', first_game.get('filename', ''))
            og_title = video_name.replace('_', ' ')
            
            # Find players image from first game
            first_game_dir = RESULTS_FOLDER / first_job_id
            if first_game_dir.exists():
                players_images = list(first_game_dir.glob("*_players.jpg"))
                if players_images:
                    og_image = f"/results/{first_job_id}/files/{players_images[0].name}"
    
    # Use unified results template - it auto-detects single vs multi-game
    return render_template('results.html', 
        job_id=match_id,
        og_title=og_title,
        og_image=og_image,
        og_description="Squash match analysis powered by AI",
        clerk_publishable_key=os.environ.get('CLERK_PUBLISHABLE_KEY', '')
    )


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
        
        analytics = job.get('result', {}).get('squash_analytics', {})
        
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
        print(f"Error saving match: {e}")
        import traceback
        traceback.print_exc()
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
    print("Starting server...")
    print(f"Open your browser to: http://localhost:{port}")
    print("=" * 50, flush=True)
    
    # In production, disable reloader to prevent threading issues
    app.run(debug=debug, host='0.0.0.0', port=port, threaded=True, use_reloader=not is_production)

