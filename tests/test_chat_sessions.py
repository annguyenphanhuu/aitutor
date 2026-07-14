"""Tests for the shared, user-scoped conversation session helpers."""

import pytest
from fastapi import HTTPException

from app.api.routes import _get_or_create_chat_session, _load_chat_history
from app.db.models import ChatMessage


@pytest.mark.asyncio
async def test_chat_session_helper_creates_and_loads_history(db_session):
    session = await _get_or_create_chat_session(
        db_session,
        user_id=1,
        session_id=None,
        title="Đạo hàm",
    )
    db_session.add(ChatMessage(session_id=session.id, role="user", content="Câu hỏi"))
    await db_session.flush()

    history = await _load_chat_history(db_session, session.id)

    assert session.title == "Đạo hàm"
    assert history == [{"role": "user", "content": "Câu hỏi"}]


@pytest.mark.asyncio
async def test_chat_session_helper_rejects_another_users_session(db_session):
    session = await _get_or_create_chat_session(
        db_session,
        user_id=1,
        session_id=None,
        title="Riêng tư",
    )

    with pytest.raises(HTTPException) as exc_info:
        await _get_or_create_chat_session(
            db_session,
            user_id=2,
            session_id=session.id,
            title="Không dùng",
        )

    assert exc_info.value.status_code == 404
