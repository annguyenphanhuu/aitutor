"""Diagnostic Test — entry-level adaptive assessment to identify knowledge gaps."""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from app.db.models import QuizQuestion, QuizSession
from app.quiz.generator import generate_questions
from app.knowledge_tracing.service import update_mastery, get_or_create_mastery
from app.knowledge_tracing.bkt import BKTModel
from app.knowledge_tracing.skill_graph import SKILLS, get_chapters, get_skills_by_chapter


bkt = BKTModel()

# Select representative skills for diagnostic (one per chapter)
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


async def start_diagnostic(db: AsyncSession, user_id: int = 1) -> dict:
    """Create a diagnostic session with one question per key skill."""
    all_questions = []

    for skill_id, difficulty in DIAGNOSTIC_SKILLS:
        qs = generate_questions(skill_id, difficulty, count=1)
        if qs:
            q_data = qs[0]
            dbq = QuizQuestion(
                skill_id=q_data["skill_id"],
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
        {"question_id": dbq.id, "answered": False, "is_correct": None, "skill_id": dbq.skill_id}
        for dbq in all_questions
    ]

    session = QuizSession(
        user_id=user_id,
        session_type="diagnostic",
        skill_ids=[s[0] for s in DIAGNOSTIC_SKILLS],
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
    """Answer a diagnostic question and get the next one."""
    # Get session
    result = await db.execute(
        select(QuizSession).where(QuizSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        return {"error": "Phiên kiểm tra không tồn tại."}

    # Get question
    result = await db.execute(
        select(QuizQuestion).where(QuizQuestion.id == question_id)
    )
    question = result.scalar_one_or_none()
    if not question:
        return {"error": "Câu hỏi không tồn tại."}

    is_correct = (selected_index == question.correct_index)

    # Update session
    questions = list(session.questions)
    for q in questions:
        if q["question_id"] == question_id:
            q["answered"] = True
            q["is_correct"] = is_correct
            break

    # CRITICAL: SQLAlchemy does NOT auto-detect mutations in JSON columns.
    # Must use flag_modified() to explicitly mark the field as changed.
    from sqlalchemy.orm.attributes import flag_modified
    session.questions = questions
    flag_modified(session, "questions")

    session.current_index += 1
    if is_correct:
        session.correct_count += 1

    # Update mastery
    if question.skill_id:
        await update_mastery(db, question.skill_id, is_correct, user_id)

    # Find next unanswered question
    next_question = None
    unanswered = [q for q in questions if not q["answered"]]
    is_completed = len(unanswered) == 0

    if not is_completed:
        next_unanswered = unanswered[0]
        nq_result = await db.execute(
            select(QuizQuestion).where(QuizQuestion.id == next_unanswered["question_id"])
        )
        nq = nq_result.scalar_one_or_none()
        if nq:
            next_question = {
                "id": nq.id,
                "question_latex": nq.question_latex,
                "choices": nq.choices,
                "skill_id": nq.skill_id,
                "difficulty": nq.difficulty,
            }

    if is_completed:
        session.is_completed = True
        session.completed_at = datetime.utcnow()

    answered_count = sum(1 for q in questions if q["answered"])

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



async def get_diagnostic_result(db: AsyncSession, session_id: int, user_id: int = 1) -> dict:
    """Get diagnostic results with skill profile and weak areas."""
    result = await db.execute(
        select(QuizSession).where(QuizSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        return {"error": "Phiên kiểm tra không tồn tại."}

    skill_profile = []
    weak_areas = []

    for q_info in session.questions:
        sid = q_info.get("skill_id", "")
        if not sid:
            continue
        skill_info = SKILLS.get(sid, {})
        mastery_rec = await get_or_create_mastery(db, sid, user_id)
        p = mastery_rec.p_mastery
        level = bkt.get_mastery_level(p)

        profile_entry = {
            "skill_id": sid,
            "skill_name": skill_info.get("name", sid),
            "estimated_mastery": round(p, 3),
            "level": level,
            "is_correct": q_info.get("is_correct", False),
        }
        skill_profile.append(profile_entry)

        if p < 0.5:
            weak_areas.append({
                "skill_id": sid,
                "skill_name": skill_info.get("name", sid),
                "recommendation": f"Cần ôn lại: {skill_info.get('name', sid)}",
            })

    overall = (session.correct_count / session.total_questions * 100) if session.total_questions else 0

    return {
        "session_id": session.id,
        "skill_profile": skill_profile,
        "weak_areas": weak_areas,
        "overall_score": round(overall, 1),
    }
