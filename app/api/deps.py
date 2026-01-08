from typing import Annotated
from fastapi import Depends, HTTPException, status, Header, Path
from sqlmodel.ext.asyncio.session import AsyncSession
from app.core.security import validate_session_token, verify_user_access
from app.core.database import get_session

async def get_current_user(
    authorization: Annotated[str | None, Header()] = None,
    session: AsyncSession = Depends(get_session)
) -> str:
    """
    FastAPI dependency for extracting authenticated user ID from Better Auth session token.

    Args:
        authorization: Authorization header (Bearer <token>)
        session: Database session

    Returns:
        User ID from valid session

    Raises:
        HTTPException: 401 if token is missing or invalid
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract token from "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1]
    user_id = await validate_session_token(token, session)

    return user_id


async def verify_user_id(
    user_id: str = Path(..., description="User ID from URL path"),
    current_user: str = Depends(get_current_user)
) -> str:
    """
    FastAPI dependency that verifies the URL path user_id matches the authenticated user.

    Args:
        user_id: User ID from URL path parameter
        current_user: Authenticated user ID from JWT token

    Returns:
        User ID (authenticated and verified against path)

    Raises:
        HTTPException: 403 if URL user_id doesn't match authenticated user_id
    """
    verify_user_access(current_user, user_id)
    return user_id
