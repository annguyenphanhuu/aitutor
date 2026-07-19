"""Ownership and pending-question tests for the spaced repetition service."""

import pytest
from sqlalchemy import func, select

from app.db.models import QuizQuestion
from app.spaced_repetition.service import get_due_cards, get_or_create_card, review_card


@pytest.fixture
async def sr_card_with_question(db_session):
    card = await get_or_create_card(db_session, "derivative_basic", user_id=1)
    question = QuizQuestion(
        skill_id="derivative_basic",
        difficulty=1,
        question_latex="Tính đạo hàm của $x^2$",
        choices=["$2x$", "$x$", "$x^2$", "$2$"],
        correct_index=0,
        explanation="",
    )
    db_session.add(question)
    await db_session.flush()
    return card, question


@pytest.mark.asyncio
async def test_review_card_updates_owner_card(db_session, sr_card_with_question):
    card, question = sr_card_with_question

    result = await review_card(db_session, card.id, question.id, 0, user_id=1)

    assert "error" not in result
    assert result["is_correct"] is True
    assert card.repetitions == 1


@pytest.mark.asyncio
async def test_review_card_rejects_other_user(db_session, sr_card_with_question):
    card, question = sr_card_with_question

    result = await review_card(db_session, card.id, question.id, 0, user_id=2)

    assert "error" in result
    assert card.repetitions == 0


async def _count_questions(db_session) -> int:
    result = await db_session.execute(select(func.count(QuizQuestion.id)))
    return result.scalar() or 0


@pytest.mark.asyncio
async def test_get_due_cards_reuses_pending_question(db_session):
    """Gọi GET /review/due nhiều lần không được sinh thêm câu hỏi mới."""
    await get_or_create_card(db_session, "derivative_basic", user_id=1)

    first = await get_due_cards(db_session, user_id=1)
    count_after_first = await _count_questions(db_session)

    second = await get_due_cards(db_session, user_id=1)
    count_after_second = await _count_questions(db_session)

    assert len(first) == 1
    assert len(second) == 1
    assert count_after_second == count_after_first
    assert second[0]["question"]["id"] == first[0]["question"]["id"]


@pytest.mark.asyncio
async def test_pending_question_regenerates_after_review(db_session):
    """Sau khi review xong, lần ôn tiếp theo phải có câu hỏi mới."""
    card = await get_or_create_card(db_session, "derivative_basic", user_id=1)

    due = await get_due_cards(db_session, user_id=1)
    question_id = due[0]["question"]["id"]

    result = await review_card(db_session, card.id, question_id, 0, user_id=1)
    assert "error" not in result

    # Đưa card về trạng thái due để ôn lại ngay trong test
    from datetime import datetime, timedelta
    card.next_review = datetime.utcnow() - timedelta(days=1)
    await db_session.flush()

    due_again = await get_due_cards(db_session, user_id=1)
    assert len(due_again) == 1
    assert due_again[0]["question"]["id"] != question_id
