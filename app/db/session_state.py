"""Helpers đọc/ghi SessionState — trạng thái hội thoại tường minh.

Chỉ flush, không commit (giống update_mastery) — transaction do request quản lý.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SessionState


async def get_session_state(db: AsyncSession, session_id: int) -> Optional[SessionState]:
    result = await db.execute(
        select(SessionState).where(SessionState.session_id == session_id)
    )
    return result.scalar_one_or_none()


async def upsert_session_state(
    db: AsyncSession,
    session_id: int,
    current_problem: Optional[str] = ...,
    awaiting_answer: Optional[bool] = ...,
    last_skill_ids: Optional[list[str]] = ...,
) -> SessionState:
    """Tạo/cập nhật state của một session. Tham số bỏ qua (Ellipsis) giữ nguyên giá trị cũ."""
    state = await get_session_state(db, session_id)
    if state is None:
        state = SessionState(session_id=session_id)
        db.add(state)
    if current_problem is not ...:
        state.current_problem = current_problem
    if awaiting_answer is not ...:
        state.awaiting_answer = awaiting_answer
    if last_skill_ids is not ...:
        state.last_skill_ids = last_skill_ids
    await db.flush()
    return state
