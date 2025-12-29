"""
Authentication Module
Handles Clerk JWT verification and user session management.
"""

import os
import jwt
import requests
from functools import wraps
from typing import Optional, Dict, Any
from flask import request, jsonify, g

# Clerk JWKS endpoint cache
_jwks_cache = None
_jwks_cache_time = 0


def get_clerk_jwks():
    """
    Get Clerk's JSON Web Key Set for token verification.
    Caches the JWKS for performance.
    """
    global _jwks_cache, _jwks_cache_time
    import time
    
    # Cache for 1 hour
    if _jwks_cache and (time.time() - _jwks_cache_time) < 3600:
        return _jwks_cache
    
    # Get Clerk frontend API URL from publishable key
    publishable_key = os.environ.get('CLERK_PUBLISHABLE_KEY', '')
    
    # Extract the instance ID from the publishable key
    # Format: pk_test_xxx or pk_live_xxx
    if publishable_key.startswith('pk_'):
        # The JWKS URL is based on your Clerk frontend API
        # You can find this in Clerk Dashboard > API Keys
        clerk_frontend_api = os.environ.get('CLERK_FRONTEND_API')
        
        if not clerk_frontend_api:
            # Try to construct from publishable key
            # This is a simplified approach - in production, set CLERK_FRONTEND_API explicitly
            parts = publishable_key.split('_')
            if len(parts) >= 3:
                instance_id = parts[2][:24]  # Get the instance identifier
                clerk_frontend_api = f"https://{instance_id}.clerk.accounts.dev"
    
    if not clerk_frontend_api:
        raise ValueError("CLERK_FRONTEND_API environment variable must be set")
    
    jwks_url = f"{clerk_frontend_api}/.well-known/jwks.json"
    
    try:
        response = requests.get(jwks_url, timeout=10)
        response.raise_for_status()
        _jwks_cache = response.json()
        _jwks_cache_time = time.time()
        return _jwks_cache
    except Exception as e:
        print(f"Error fetching Clerk JWKS: {e}")
        return None


def verify_clerk_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify a Clerk JWT token and return the decoded payload.
    Returns None if verification fails.
    """
    if not token:
        return None
    
    # Remove 'Bearer ' prefix if present
    if token.startswith('Bearer '):
        token = token[7:]
    
    try:
        # Get the unverified header to find the key ID
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get('kid')
        
        if not kid:
            print("No key ID in token header")
            return None
        
        # Get JWKS and find the matching key
        jwks = get_clerk_jwks()
        if not jwks:
            print("Could not fetch JWKS")
            return None
        
        # Find the key with matching kid
        signing_key = None
        for key in jwks.get('keys', []):
            if key.get('kid') == kid:
                signing_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
                break
        
        if not signing_key:
            print(f"No matching key found for kid: {kid}")
            return None
        
        # Verify and decode the token
        payload = jwt.decode(
            token,
            signing_key,
            algorithms=['RS256'],
            options={
                'verify_exp': True,
                'verify_aud': False,  # Clerk doesn't always set aud
                'verify_iss': False   # We'll verify manually if needed
            }
        )
        
        return payload
        
    except jwt.ExpiredSignatureError:
        print("Token has expired")
        return None
    except jwt.InvalidTokenError as e:
        print(f"Invalid token: {e}")
        return None
    except Exception as e:
        print(f"Token verification error: {e}")
        return None


def get_current_user_from_request() -> Optional[Dict[str, Any]]:
    """
    Extract and verify the user from the current request.
    Looks for the token in the Authorization header or __session cookie.
    Returns the user info dict or None if not authenticated.
    """
    # Try Authorization header first
    auth_header = request.headers.get('Authorization')
    token = None
    
    if auth_header:
        token = auth_header
    else:
        # Try the __session cookie (Clerk's default cookie name)
        token = request.cookies.get('__session')
    
    if not token:
        return None
    
    # Verify the token
    payload = verify_clerk_token(token)
    
    if not payload:
        return None
    
    # Extract user info from Clerk token
    # Clerk tokens include 'sub' (user ID) and may include email, name, etc.
    user_info = {
        'clerk_id': payload.get('sub'),
        'email': payload.get('email'),
        'name': payload.get('name') or payload.get('first_name', ''),
        'image_url': payload.get('image_url'),
        'session_id': payload.get('sid')
    }
    
    # Add full name if we have first/last
    if payload.get('first_name') or payload.get('last_name'):
        first = payload.get('first_name', '')
        last = payload.get('last_name', '')
        user_info['name'] = f"{first} {last}".strip()
    
    return user_info


def get_current_user() -> Optional[Dict[str, Any]]:
    """
    Get the current authenticated user, creating them in Supabase if needed.
    Returns the full user record from Supabase, or None if not authenticated.
    """
    # Check if we already resolved the user in this request
    if hasattr(g, 'current_user'):
        return g.current_user
    
    user_info = get_current_user_from_request()
    
    if not user_info or not user_info.get('clerk_id'):
        g.current_user = None
        return None
    
    # Get or create user in Supabase
    try:
        from supabase_client import get_or_create_user
        
        db_user = get_or_create_user(
            clerk_id=user_info['clerk_id'],
            email=user_info.get('email'),
            name=user_info.get('name')
        )
        
        if db_user:
            # Merge Clerk info with DB user
            db_user['image_url'] = user_info.get('image_url')
            g.current_user = db_user
            return db_user
        
    except Exception as e:
        print(f"Error getting/creating user in Supabase: {e}")
    
    g.current_user = None
    return None


def login_required(f):
    """
    Decorator for routes that require authentication.
    Returns 401 if user is not authenticated.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function


def login_optional(f):
    """
    Decorator for routes where login is optional.
    Sets g.current_user if authenticated, otherwise None.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # This just ensures get_current_user is called
        get_current_user()
        return f(*args, **kwargs)
    return decorated_function

