"""
Supabase Client Module
Handles database operations and storage for user matches.
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

# Initialize on first use to avoid import errors if supabase not installed yet
_supabase_client = None

def get_supabase_client():
    """Get or create Supabase client singleton."""
    global _supabase_client
    
    if _supabase_client is None:
        from supabase import create_client, Client
        
        url = os.environ.get('SUPABASE_URL')
        key = os.environ.get('SUPABASE_SERVICE_KEY') or os.environ.get('SUPABASE_KEY')
        
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY/SUPABASE_SERVICE_KEY must be set")
        
        _supabase_client = create_client(url, key)
    
    return _supabase_client


# ============================================================================
# User Management
# ============================================================================

def get_or_create_user(clerk_id: str, email: str = None, name: str = None) -> Dict[str, Any]:
    """
    Get existing user by Clerk ID or create a new one.
    Returns the user record.
    """
    supabase = get_supabase_client()
    
    # Try to find existing user
    result = supabase.table('users').select('*').eq('clerk_id', clerk_id).execute()
    
    if result.data:
        return result.data[0]
    
    # Create new user
    new_user = {
        'clerk_id': clerk_id,
        'email': email,
        'name': name
    }
    
    result = supabase.table('users').insert(new_user).execute()
    return result.data[0] if result.data else None


def get_user_by_clerk_id(clerk_id: str) -> Optional[Dict[str, Any]]:
    """Get user by Clerk ID."""
    supabase = get_supabase_client()
    result = supabase.table('users').select('*').eq('clerk_id', clerk_id).execute()
    return result.data[0] if result.data else None


# ============================================================================
# Match Management
# ============================================================================

def save_match(
    user_id: str,
    job_id: str,
    filename: str,
    analytics: Dict[str, Any],
    sport: str = 'squash',
    players_image_url: str = None,
    player1_name: str = None,
    player2_name: str = None
) -> Dict[str, Any]:
    """
    Save a match to the database.
    Returns the created match record.
    """
    supabase = get_supabase_client()
    
    # Extract duration from analytics (convert to int for database)
    duration_seconds = None
    if analytics:
        match_info = analytics.get('match_info', {})
        duration_val = match_info.get('duration_seconds')
        if duration_val is not None:
            duration_seconds = int(duration_val)
    
    match_data = {
        'user_id': user_id,
        'job_id': job_id,
        'filename': filename,
        'sport': sport,
        'analytics': analytics,
        'players_image_url': players_image_url,
        'player1_name': player1_name,
        'player2_name': player2_name,
        'duration_seconds': duration_seconds
    }
    
    # Insert (not upsert) - check_match_saved prevents duplicates
    # Using insert instead of upsert to avoid composite key issues
    try:
        result = supabase.table('matches').insert(match_data).execute()
    except Exception as e:
        # If duplicate, try to fetch the existing one
        if 'duplicate' in str(e).lower() or '23505' in str(e):
            existing = supabase.table('matches').select('*').eq('user_id', user_id).eq('job_id', job_id).execute()
            return existing.data[0] if existing.data else None
        raise
    
    return result.data[0] if result.data else None


def get_user_matches(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get all matches for a user, ordered by creation date (newest first).
    Returns list of match records (without full analytics for performance).
    """
    supabase = get_supabase_client()
    
    result = supabase.table('matches').select(
        'id, job_id, filename, sport, players_image_url, player1_name, player2_name, duration_seconds, created_at'
    ).eq('user_id', user_id).order('created_at', desc=True).limit(limit).execute()
    
    return result.data or []


def get_match_by_id(match_id: str, user_id: str = None) -> Optional[Dict[str, Any]]:
    """
    Get a specific match by ID.
    If user_id is provided, ensures the match belongs to that user.
    """
    supabase = get_supabase_client()
    
    query = supabase.table('matches').select('*').eq('id', match_id)
    
    if user_id:
        query = query.eq('user_id', user_id)
    
    result = query.execute()
    return result.data[0] if result.data else None


def get_match_by_job_id(job_id: str, user_id: str = None) -> Optional[Dict[str, Any]]:
    """
    Get a match by job_id.
    If user_id is provided, ensures the match belongs to that user.
    """
    supabase = get_supabase_client()
    
    query = supabase.table('matches').select('*').eq('job_id', job_id)
    
    if user_id:
        query = query.eq('user_id', user_id)
    
    result = query.execute()
    return result.data[0] if result.data else None


def delete_match(match_id: str, user_id: str) -> bool:
    """
    Delete a match. Requires user_id to ensure ownership.
    Returns True if deleted, False otherwise.
    """
    supabase = get_supabase_client()
    
    result = supabase.table('matches').delete().eq('id', match_id).eq('user_id', user_id).execute()
    
    return len(result.data) > 0 if result.data else False


def check_match_saved(job_id: str, user_id: str) -> bool:
    """Check if a match is already saved for a user."""
    supabase = get_supabase_client()
    
    result = supabase.table('matches').select('id').eq('job_id', job_id).eq('user_id', user_id).execute()
    
    return len(result.data) > 0 if result.data else False


# ============================================================================
# Storage (Player Images)
# ============================================================================

def upload_player_image(job_id: str, image_path: str) -> Optional[str]:
    """
    Upload player identification image to Supabase Storage.
    Returns the public URL of the uploaded image.
    """
    supabase = get_supabase_client()
    
    if not os.path.exists(image_path):
        return None
    
    # Read the image file
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    # Generate storage path
    filename = os.path.basename(image_path)
    storage_path = f"{job_id}/{filename}"
    
    try:
        # Upload to storage bucket
        result = supabase.storage.from_('player-images').upload(
            storage_path,
            image_data,
            file_options={"content-type": "image/jpeg", "upsert": "true"}
        )
        
        # Get public URL
        public_url = supabase.storage.from_('player-images').get_public_url(storage_path)
        
        return public_url
    except Exception as e:
        print(f"Error uploading player image: {e}")
        return None


def delete_player_image(job_id: str, filename: str) -> bool:
    """Delete player image from storage."""
    supabase = get_supabase_client()
    
    storage_path = f"{job_id}/{filename}"
    
    try:
        supabase.storage.from_('player-images').remove([storage_path])
        return True
    except Exception as e:
        print(f"Error deleting player image: {e}")
        return False

