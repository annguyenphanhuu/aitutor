"""Quiz service flow tests using the in-memory database fixture."""

from collections import Counter

import pytest

from app.db.models import QuizSession
from app.quiz.service import create_quiz_session


@pytest.mark.asyncio
async def test_create_practice_quiz_keeps_requested_mcq_count(db_session):
    result = await create_quiz_session(
        db_session,
        skill_id="derivative_basic",
        difficulty=1,
        count=4,
        user_id=1,
    )

    assert result["session_type"] == "practice"
    assert len(result["questions"]) == 4
    assert {question["question_type"] for question in result["questions"]} == {"mcq"}

    session = await db_session.get(QuizSession, result["session_id"])
    assert session is not None
    assert session.exam_format is False


@pytest.mark.asyncio
async def test_create_exam_format_quiz_restores_thpt_question_mix(db_session):
    result = await create_quiz_session(
        db_session,
        skill_id="function_survey",
        difficulty=2,
        count=20,
        user_id=1,
        exam_format=True,
    )

    question_types = Counter(q["question_type"] for q in result["questions"])
    assert question_types == {"mcq": 3, "true_false": 1, "short_answer": 1}
    assert result["session_type"] == "exam"
    assert result["total_questions"] == 5
    assert result["max_score"] == pytest.approx(2.25)

    session = await db_session.get(QuizSession, result["session_id"])
    assert session is not None
    assert session.exam_format is True
    assert session.session_type == "exam"
