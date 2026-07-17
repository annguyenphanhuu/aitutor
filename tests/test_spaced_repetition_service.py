"""Ownership tests for the spaced repetition service."""

import pytest

from app.db.models import QuizQuestion
from app.spaced_repetition.service import get_or_create_card, review_card


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
