import pytest
from sqlmodel import SQLModel, create_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from app.core.database import get_async_engine, get_session

@pytest.mark.asyncio
async def test_get_async_engine():
    """Test async engine creation"""
    engine = get_async_engine()
    assert engine is not None
    assert "postgresql+asyncpg" in str(engine.url)

@pytest.mark.asyncio
async def test_get_session():
    """Test async session creation"""
    async for session in get_session():
        assert isinstance(session, AsyncSession)
        break

def test_database_url_from_env(monkeypatch):
    """Test DATABASE_URL is read from environment"""
    test_url = "postgresql+asyncpg://test:test@localhost/testdb"
    monkeypatch.setenv("DATABASE_URL", test_url)
    # Re-import to pick up new env var
    from importlib import reload
    import app.core.database as db_module
    import app.core.config as config_module
    reload(config_module)
    reload(db_module)
    engine = db_module.get_async_engine()
    # Check key components (password may be masked)
    url_str = str(engine.url)
    assert "postgresql+asyncpg" in url_str
    assert "@localhost/testdb" in url_str
