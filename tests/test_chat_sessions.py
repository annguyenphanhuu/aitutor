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
async def test_load_chat_history_keeps_newest_messages(db_session):
    """Long conversations must keep the NEWEST messages, in chronological order."""
    session = await _get_or_create_chat_session(
        db_session,
        user_id=1,
        session_id=None,
        title="Dài",
    )
    for index in range(30):
        db_session.add(ChatMessage(
            session_id=session.id,
            role="user" if index % 2 == 0 else "assistant",
            content=f"msg-{index}",
        ))
    await db_session.flush()

    history = await _load_chat_history(db_session, session.id)

    assert len(history) == 20
    # Oldest 10 messages dropped; newest 20 kept in chronological order
    assert history[0]["content"] == "msg-10"
    assert history[-1]["content"] == "msg-29"


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
