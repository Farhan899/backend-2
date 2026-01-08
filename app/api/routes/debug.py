from typing import Annotated
from fastapi import APIRouter, Header, Depends
from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession
from app.core.database import get_session

router = APIRouter(prefix="/api/debug", tags=["debug"])

@router.get("/check-token")
async def check_token(
    authorization: Annotated[str | None, Header()] = None
):
    """
    Debug endpoint to inspect the authorization token
    """
    if not authorization:
        return {
            "error": "No authorization header provided",
            "authorization": None
        }

    parts = authorization.split()

    return {
        "full_header": authorization,
        "parts_count": len(parts),
        "parts": parts,
        "token": parts[1] if len(parts) > 1 else None,
        "token_length": len(parts[1]) if len(parts) > 1 else 0,
        "token_segments": len(parts[1].split('.')) if len(parts) > 1 else 0,
    }

@router.post("/migrate-priority-due-date")
async def migrate_priority_due_date(
    session: Annotated[AsyncSession, Depends(get_session)]
):
    """
    Add priority and due_date columns to tasks table
    """
    try:
        # Check if columns exist
        result = await session.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'tasks' AND column_name IN ('priority', 'due_date')
        """))
        existing = [row[0] for row in result.fetchall()]

        result = {}

        # Add priority column if not exists
        if 'priority' not in existing:
            await session.execute(text("""
                ALTER TABLE tasks
                ADD COLUMN IF NOT EXISTS priority VARCHAR(20)
            """))
            result['priority'] = 'added'
        else:
            result['priority'] = 'already_exists'

        # Add due_date column if not exists
        if 'due_date' not in existing:
            await session.execute(text("""
                ALTER TABLE tasks
                ADD COLUMN IF NOT EXISTS due_date TIMESTAMP
            """))
            result['due_date'] = 'added'
        else:
            result['due_date'] = 'already_exists'

        await session.commit()
        return {"status": "success", "result": result}
    except Exception as e:
        await session.rollback()
        return {"status": "error", "message": str(e)}
