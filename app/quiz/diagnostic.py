"""Diagnostic test service for entry-level adaptive assessment."""

from app.utils.time_utils import utcnow

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import QuizQuestion, QuizSession
from app.knowledge_tracing.bkt import BKTModel
from app.knowledge_tracing.service import get_or_create_mastery, update_mastery
from app.knowledge_tracing.skill_graph import SKILLS
from app.quiz.generator import generate_questions


bkt = BKTModel()


DIAGNOSTIC_SKILLS = [
    ("derivative_basic", 1),
    ("function_survey", 2),
    ("primitive_basic", 1),
    ("integral_definite", 2),
    ("complex_basic", 1),
    ("complex_operations", 2),
    ("geometry_vectors", 1),
    ("geometry_line_plane", 2),
    ("geometry_distance_angle", 2),
    ("probability_basic", 1),
    ("exponential_basic", 1),
    ("logarithm_basic", 2),
    ("sequence_basic", 1),
    ("arithmetic_sequence", 2),
    ("statistics_descriptive", 1),
    ("geometric_sequence", 2),
]


def _skill_ids_for_question(question: QuizQuestion) -> list[str]:
    return question.skill_ids or ([question.skill_id] if question.skill_id else [])


async def start_diagnostic(db: AsyncSession, user_id: int = 1) -> dict:
    """Create a diagnostic session with one representative question per skill."""
    all_questions: list[QuizQuestion] = []

    for skill_id, difficulty in DIAGNOSTIC_SKILLS:
        generated = generate_questions(skill_id, difficulty, count=1)
        if not generated:
            continue

        q_data = generated[0]
        q_skill_ids = q_data.get("skill_ids") or [q_data["skill_id"]]
        dbq = QuizQuestion(
            skill_id=q_data["skill_id"],
            skill_ids=q_skill_ids,
            formula_ids=q_data.get("formula_ids", []),
            difficulty=q_data["difficulty"],
            question_latex=q_data["question_latex"],
            choices=q_data["choices"],
            correct_index=q_data["correct_index"],
            explanation=q_data.get("explanation", ""),
            sympy_expr=q_data.get("sympy_expr", ""),
        )
        db.add(dbq)
        all_questions.append(dbq)

    await db.flush()

    session_questions = [
        {
            "question_id": question.id,
            "answered": False,
            "is_correct": None,
            "skill_id": question.skill_id,
            "skill_ids": _skill_ids_for_question(question),
            "formula_ids": question.formula_ids or [],
        }
        for question in all_questions
    ]

    session = QuizSession(
        user_id=user_id,
        session_type="diagnostic",
        skill_ids=[skill_id for skill_id, _ in DIAGNOSTIC_SKILLS],
        questions=session_questions,
        total_questions=len(all_questions),
    )
    db.add(session)
    await db.flush()

    first_q = all_questions[0] if all_questions else None

    return {
        "session_id": session.id,
        "first_question": {
            "id": first_q.id,
            "question_latex": first_q.question_latex,
            "choices": first_q.choices,
            "skill_id": first_q.skill_id,
            "skill_ids": _skill_ids_for_question(first_q),
            "formula_ids": first_q.formula_ids or [],
            "difficulty": first_q.difficulty,
        } if first_q else None,
        "total_questions": len(all_questions),
    }


async def answer_diagnostic(
    db: AsyncSession,
    session_id: int,
    question_id: int,
    selected_index: int,
    user_id: int = 1,
) -> dict:
    """Record an answer, update mastery for all question skills, and return the next item."""
    result = await db.execute(select(QuizSession).where(QuizSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        return {"error": "Diagnostic session not found."}

    result = await db.execute(select(QuizQuestion).where(QuizQuestion.id == question_id))
    question = result.scalar_one_or_none()
    if not question:
        return {"error": "Question not found."}

    is_correct = selected_index == question.correct_index

    questions = list(session.questions)
    for item in questions:
        if item["question_id"] == question_id:
            item["answered"] = True
            item["is_correct"] = is_correct
            break

    from sqlalchemy.orm.attributes import flag_modified

    session.questions = questions
    flag_modified(session, "questions")

    session.current_index += 1
    if is_correct:
        session.correct_count += 1

    q_skill_ids = _skill_ids_for_question(question)
    if q_skill_ids:
        await update_mastery(
            db,
            {skill_id: {"passed": is_correct} for skill_id in q_skill_ids},
            user_id=user_id,
        )

    unanswered = [item for item in questions if not item["answered"]]
    is_completed = not unanswered
    next_question = None

    if unanswered:
        next_result = await db.execute(
            select(QuizQuestion).where(QuizQuestion.id == unanswered[0]["question_id"])
        )
        nq = next_result.scalar_one_or_none()
        if nq:
            next_question = {
                "id": nq.id,
                "question_latex": nq.question_latex,
                "choices": nq.choices,
                "skill_id": nq.skill_id,
                "skill_ids": _skill_ids_for_question(nq),
                "formula_ids": nq.formula_ids or [],
                "difficulty": nq.difficulty,
            }

    if is_completed:
        session.is_completed = True
        session.completed_at = utcnow()

    answered_count = sum(1 for item in questions if item["answered"])
    await db.flush()

    return {
        "is_correct": is_correct,
        "correct_index": question.correct_index,
        "explanation": question.explanation,
        "next_question": next_question,
        "is_completed": is_completed,
        "progress": {
            "current": answered_count,
            "total": session.total_questions,
        },
    }


async def get_diagnostic_result(
    db: AsyncSession,
    session_id: int,
    user_id: int = 1,
) -> dict:
    """Return diagnostic results grouped by every tagged skill."""
    result = await db.execute(select(QuizSession).where(QuizSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        return {"error": "Diagnostic session not found."}

    skill_profile = []
    weak_areas = []

    for q_info in session.questions:
        q_skill_ids = q_info.get("skill_ids") or (
            [q_info.get("skill_id")] if q_info.get("skill_id") else []
        )
        for skill_id in q_skill_ids:
            skill_info = SKILLS.get(skill_id, {})
            mastery_rec = await get_or_create_mastery(db, skill_id, user_id)
            p_mastery = mastery_rec.p_mastery
            level = bkt.get_mastery_level(p_mastery)

            skill_profile.append({
                "skill_id": skill_id,
                "skill_name": skill_info.get("name", skill_id),
                "estimated_mastery": round(p_mastery, 3),
                "level": level,
                "is_correct": q_info.get("is_correct", False),
            })

            if p_mastery < 0.5:
                weak_areas.append({
                    "skill_id": skill_id,
                    "skill_name": skill_info.get("name", skill_id),
                    "recommendation": f"Review: {skill_info.get('name', skill_id)}",
                })

    overall = (
        session.correct_count / session.total_questions * 100
        if session.total_questions
        else 0
    )

    return {
        "session_id": session.id,
        "skill_profile": skill_profile,
        "weak_areas": weak_areas,
        "overall_score": round(overall, 1),
    }
