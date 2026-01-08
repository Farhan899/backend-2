from typing import AsyncGenerator
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Create async engine
async_engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.ENVIRONMENT == "development",
    future=True,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

# Session factory
async_session_maker = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

def get_async_engine() -> AsyncEngine:
    """Get async database engine"""
    return async_engine

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database session"""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def create_db_and_tables():
    """Create database tables (for testing/init)"""
    async with async_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
