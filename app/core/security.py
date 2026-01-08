from jose import jwt, JWTError, ExpiredSignatureError
from fastapi import HTTPException, status
from typing import Dict, Any, Optional
from datetime import datetime
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from app.core.config import settings

def decode_jwt(token: str, secret: str = settings.BETTER_AUTH_SECRET) -> Dict[str, Any]:
    """
    Decode and validate JWT token (legacy function - not used for Better Auth).

    Args:
        token: JWT token string
        secret: Secret key for verification

    Returns:
        Decoded JWT payload

    Raises:
        HTTPException: 401 if token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            options={"verify_aud": False}
        )

        # Verify required claims
        if "sub" not in payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing user ID (sub claim)"
            )

        return payload

    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {str(e)}"
        )

async def validate_session_token(token: str, session: AsyncSession) -> str:
    """
    Validate Better Auth session token and return user ID.

    Args:
        token: Better Auth session token
        session: Database session

    Returns:
        User ID from valid session

    Raises:
        HTTPException: 401 if token is invalid or expired
    """
    from sqlalchemy import text

    try:
        # Query the session table to validate the token
        query = text("""
            SELECT "userId", "expiresAt"
            FROM session
            WHERE token = :token
        """)

        result = await session.execute(query, {"token": token})
        row = result.first()

        if not row:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session token"
            )

        user_id, expires_at = row

        # Check if session has expired
        if expires_at < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session has expired"
            )

        return user_id

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Session validation failed: {str(e)}"
        )

def verify_user_access(token_user_id: str, path_user_id: str) -> None:
    """
    Verify that the authenticated user matches the path user_id.

    Args:
        token_user_id: User ID from JWT token (sub claim)
        path_user_id: User ID from URL path parameter

    Raises:
        HTTPException: 403 if user IDs don't match
    """
    if token_user_id != path_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: cannot access another user's resources"
        )
