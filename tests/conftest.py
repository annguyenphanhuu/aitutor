"""
Shared pytest fixtures for AITutor test suite.

Provides:
- Async DB session (in-memory SQLite)
- Mock LLM / OpenAI client
- Common test data
"""

import sys
import os
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# ── Ensure project root is on sys.path ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── Mock settings BEFORE any app import ─────────────────────────────────────
# This prevents real .env from being loaded during tests.
@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Provide safe test settings — no real API keys needed."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-unit-tests")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://aitutor:aitutorpassword@localhost:5432/aitutordb")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "./test_chroma_db")
    monkeypatch.setenv("DEBUG", "false")

    # Clear cached settings so they reload with new env vars
    from app.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ── Async database session (in-memory SQLite) ──────────────────────────────
@pytest.fixture
async def db_session():
    """Create an in-memory SQLite database and yield an async session."""
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
    from app.db.database import Base

    engine = create_async_engine("postgresql+asyncpg://aitutor:aitutorpassword@localhost:5432/aitutordb", echo=False)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ── Seed data helpers ───────────────────────────────────────────────────────
@pytest.fixture
async def seeded_db(db_session):
    """DB session seeded with a test user and some skill masteries."""
    from app.db.models import User, SkillMastery

    user = User(username="test_student", display_name="Test Student")
    db_session.add(user)
    await db_session.flush()

    skills_data = [
        ("derivative_basic",    "Đạo hàm cơ bản",     0.8,  10, 8),
        ("derivative_rules",    "Quy tắc đạo hàm",    0.5,  6,  3),
        ("integral_definite",   "Tích phân xác định",  0.2,  4,  1),
        ("probability_basic",   "Xác suất cơ bản",    0.15, 2,  0),
        ("geometry_vectors",    "Vectơ trong không gian", 0.9, 12, 11),
    ]
    for sid, sname, p, total, correct in skills_data:
        mastery = SkillMastery(
            user_id=user.id,
            skill_id=sid,
            skill_name=sname,
            p_mastery=p,
            total_attempts=total,
            correct_attempts=correct,
        )
        db_session.add(mastery)

    await db_session.flush()
    yield db_session, user


# ── Event loop for async tests ──────────────────────────────────────────────
@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
