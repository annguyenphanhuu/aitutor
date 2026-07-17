"""Quiz service flow tests using the in-memory database fixture."""

from collections import Counter

import pytest

from app.db.models import QuizSession
from app.quiz.service import create_quiz_session, get_quiz_result, submit_quiz_answer


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


@pytest.mark.asyncio
async def test_submit_answer_rejects_resubmission(db_session):
    quiz = await create_quiz_session(
        db_session, skill_id="derivative_basic", difficulty=1, count=2, user_id=1,
    )
    question_id = quiz["questions"][0]["id"]

    first = await submit_quiz_answer(
        db_session, quiz["session_id"], question_id, selected_index=0, user_id=1,
    )
    assert "error" not in first

    second = await submit_quiz_answer(
        db_session, quiz["session_id"], question_id, selected_index=1, user_id=1,
    )
    assert "error" in second

    session = await db_session.get(QuizSession, quiz["session_id"])
    answered = [q for q in session.questions if q["answered"]]
    assert len(answered) == 1
    assert session.correct_count <= 1


@pytest.mark.asyncio
async def test_submit_answer_rejects_other_users_session(db_session):
    quiz = await create_quiz_session(
        db_session, skill_id="derivative_basic", difficulty=1, count=1, user_id=1,
    )
    question_id = quiz["questions"][0]["id"]

    result = await submit_quiz_answer(
        db_session, quiz["session_id"], question_id, selected_index=0, user_id=2,
    )
    assert "error" in result


@pytest.mark.asyncio
async def test_submit_answer_rejects_question_outside_session(db_session):
    quiz_a = await create_quiz_session(
        db_session, skill_id="derivative_basic", difficulty=1, count=1, user_id=1,
    )
    quiz_b = await create_quiz_session(
        db_session, skill_id="derivative_basic", difficulty=1, count=1, user_id=1,
    )
    foreign_question_id = quiz_b["questions"][0]["id"]

    result = await submit_quiz_answer(
        db_session, quiz_a["session_id"], foreign_question_id, selected_index=0, user_id=1,
    )
    assert "error" in result


@pytest.mark.asyncio
async def test_true_false_blank_answer_earns_no_points(db_session):
    quiz = await create_quiz_session(
        db_session,
        skill_id="function_survey",
        difficulty=2,
        user_id=1,
        exam_format=True,
    )
    tf_question = next(
        q for q in quiz["questions"] if q["question_type"] == "true_false"
    )

    result = await submit_quiz_answer(
        db_session, quiz["session_id"], tf_question["id"], tf_answers=None, user_id=1,
    )

    assert "error" not in result
    assert result["points_earned"] == 0.0
    assert result["is_correct"] is False


@pytest.mark.asyncio
async def test_quiz_result_hidden_from_other_users(db_session):
    quiz = await create_quiz_session(
        db_session, skill_id="derivative_basic", difficulty=1, count=1, user_id=1,
    )

    owner_view = await get_quiz_result(db_session, quiz["session_id"], user_id=1)
    stranger_view = await get_quiz_result(db_session, quiz["session_id"], user_id=2)

    assert "error" not in owner_view
    assert "error" in stranger_view
