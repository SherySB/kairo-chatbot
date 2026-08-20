"""
auth_helpers.py
---------------
Authentication middleware and helper functions for the Kairo API.

Provides JWT token verification, user authentication decorators, and
standardized authentication response formatting.
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable

from fastapi import HTTPException, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from . import firebase_service

logger = logging.getLogger(__name__)

# FastAPI security scheme for Bearer token authentication
security = HTTPBearer()


# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------


def verify_token(authorization: str = Header(None)) -> dict:
    """Verify Firebase ID token from Authorization header.
    
    Parameters
    ----------
    authorization:
        Authorization header in format "Bearer <token>"
        
    Returns
    -------
    dict
        Decoded Firebase token claims
        
    Raises
    ------
    HTTPException
        401 if token is missing, invalid, or expired
        500 if Firebase service error
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is required"
        )
    
    # Extract token from "Bearer <token>" format
    try:
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() != "bearer":
            raise ValueError("Invalid authorization scheme")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Use 'Bearer <token>'"
        )
    
    try:
        decoded_token = firebase_service.verify_id_token(token)
        if decoded_token is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        return decoded_token
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("verify_token: Firebase service error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        ) from exc


def get_current_user(authorization: str = Header(None)) -> dict:
    """Get current authenticated user information.
    
    Parameters
    ----------
    authorization:
        Authorization header in format "Bearer <token>"
        
    Returns
    -------
    dict
        User information with keys: uid, email, display_name, etc.
        
    Raises
    ------
    HTTPException
        401 if token is invalid
        404 if user not found
        500 if service error
    """
    token_data = verify_token(authorization)
    uid = token_data.get("uid")
    
    if not uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user ID"
        )
    
    try:
        user = firebase_service.get_user_by_uid(uid)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return user
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"get_current_user: error retrieving user '{uid}'")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User service error"
        ) from exc


def format_auth_response(
    success: bool,
    authenticated: bool = False,
    uid: str = None,
    email: str = None,
    display_name: str = None,
    provider: str = None,
    error: str = None,
    **kwargs
) -> dict:
    """Format standardized authentication response.
    
    Parameters
    ----------
    success:
        Whether the authentication operation completed without error
    authenticated:
        Whether the user is successfully authenticated
    uid:
        Firebase user ID
    email:
        User's email address
    display_name:
        User's display name
    provider:
        Authentication provider (e.g., "password", "google.com")
    error:
        Error message on failure
    **kwargs:
        Additional fields to include in response
        
    Returns
    -------
    dict
        Standardized authentication response
    """
    response = {
        "success": success,
        "authenticated": authenticated,
        "uid": uid,
        "email": email,
        "display_name": display_name,
        "provider": provider,
        "error": error,
    }
    
    # Add any additional fields
    response.update(kwargs)
    
    # Remove None values for cleaner response
    return {k: v for k, v in response.items() if v is not None}


# ---------------------------------------------------------------------------
# Decorator for protected endpoints
# ---------------------------------------------------------------------------


def require_auth(f: Callable) -> Callable:
    """Decorator to require authentication for FastAPI endpoints.
    
    Usage:
        @app.get("/protected")
        @require_auth
        async def protected_endpoint(current_user: dict = Depends(get_current_user)):
            return {"message": f"Hello {current_user['email']}"}
    """
    @wraps(f)
    async def wrapper(*args, **kwargs):
        # The actual authentication is handled by FastAPI dependencies
        # This decorator is for marking endpoints as requiring auth
        return await f(*args, **kwargs)
    return wrapper