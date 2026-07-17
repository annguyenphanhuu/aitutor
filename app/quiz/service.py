"""Quiz Service — orchestrates quiz generation, sessions, and scoring.

THPT QG scoring:
- MCQ (Phần 1): 0.25 điểm/câu đúng
- True/False (Phần 2): 1 ý đúng = 0.1đ, 2 ý = 0.25đ, 3 ý = 0.5đ, 4 ý = 1.0đ
- Short answer (Phần 3): scored per question (typically 0.5đ)
"""

from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.models import QuizQuestion, QuizSession, QuestionBank
from app.quiz.generator import generate_questions
from app.quiz.grading import score_short_answer, score_true_false
from app.knowledge_tracing.service import update_mastery, get_or_create_mastery
from app.knowledge_tracing.skill_graph import SKILLS


# ── Create Quiz Session ──────────────────────────────────
async def create_quiz_session(
    db: AsyncSession,
    skill_id: str | None = None,
    difficulty: int = 1,
    count: int = 5,
    user_id: int = 1,
    chapter: str | None = None,
    exam_format: bool = False,
) -> dict:
    """Generate questions and create a quiz session.

    ``exam_format=True`` generates the THPT mix 3 MCQ + 1 True/False +
    1 short-answer question. Otherwise ``count`` MCQs are generated.
    If difficulty=0, uses adaptive engine to auto-select difficulty.
    """
    import random
    
    skill_ids_to_query = []
    if chapter:
        skill_ids_to_query = [sid for sid, info in SKILLS.items() if info["chapter"] == chapter]
    elif skill_id:
        skill_ids_to_query = [skill_id]
        
    if not skill_ids_to_query:
        return {"error": "Không có kỹ năng hoặc chương nào được chọn."}
        
    rep_skill_id = skill_id or skill_ids_to_query[0]

    # ── Adaptive difficulty (when difficulty=0 or "auto") ──
    adaptive_info = None
    if difficulty == 0:
        from app.quiz.adaptive_engine import get_adaptive_engine
        from app.knowledge_tracing.service import get_or_create_mastery as _get_mastery, get_recent_results, get_all_masteries
        engine = get_adaptive_engine()
        mastery_rec = await _get_mastery(db, rep_skill_id, user_id)
        recent = await get_recent_results(db, rep_skill_id, user_id, limit=10)
        all_masteries = await get_all_masteries(db, user_id)
        adaptive_info = engine.get_adaptive_summary(rep_skill_id, mastery_rec.p_mastery, recent, all_masteries)
        difficulty = adaptive_info["recommended_difficulty"]

    from sqlalchemy.sql.expression import func
    
    requested_by_type = (
        {"mcq": 3, "true_false": 1, "short_answer": 1}
        if exam_format
        else {"mcq": count}
    )
    raw_questions = []

    for question_type, requested_count in requested_by_type.items():
        res = await db.execute(
            select(QuestionBank)
            .where(QuestionBank.skill_id.in_(skill_ids_to_query))
            .where(QuestionBank.difficulty == difficulty)
            .where(QuestionBank.question_type == question_type)
            .where(QuestionBank.is_active.is_(True))
            .order_by(func.random())
            .limit(requested_count)
        )
        bank_questions = list(res.scalars().all())

        for bq in bank_questions:
            raw_questions.append({
                "skill_id": bq.skill_id,
                "skill_ids": [bq.skill_id],
                "difficulty": bq.difficulty,
                "question_type": bq.question_type,
                "question_latex": bq.question_latex,
                "choices": bq.choices,
                "correct_index": bq.correct_index,
                "statements": bq.statements,
                "correct_answer": bq.correct_answer,
                "points": bq.points,
                "explanation": bq.explanation,
                "sympy_expr": "",
            })

        missing = requested_count - len(bank_questions)
        for _ in range(missing):
            generated_skill = random.choice(skill_ids_to_query)
            raw_questions.extend(generate_questions(
                generated_skill,
                difficulty,
                count=1,
                question_type=question_type,
            ))

    if not raw_questions:
        return {"error": "Không thể sinh câu hỏi cho kỹ năng này."}

    # Save questions to DB
    db_questions = []
    for q in raw_questions:
        q_skill_ids = q.get("skill_ids") or [q["skill_id"]]
        dbq = QuizQuestion(
            skill_id=q["skill_id"],
            skill_ids=q_skill_ids,
            formula_ids=q.get("formula_ids", []),
            difficulty=q["difficulty"],
            question_type=q.get("question_type", "mcq"),
            question_latex=q["question_latex"],
            choices=q.get("choices"),
            correct_index=q.get("correct_index"),
            statements=q.get("statements"),
            correct_answer=q.get("correct_answer"),
            points=q.get("points", 0.25),
            explanation=q.get("explanation", ""),
            sympy_expr=q.get("sympy_expr", ""),
        )
        db.add(dbq)
        db_questions.append(dbq)

    await db.flush()

    # Build session question list
    session_questions = [
        {
            "question_id": dbq.id,
            "question_type": dbq.question_type,
            "answered": False,
            "is_correct": None,
            "points_earned": 0.0,
            "max_points": dbq.points,
            "skill_ids": dbq.skill_ids or [dbq.skill_id],
            "formula_ids": dbq.formula_ids or [],
        }
        for dbq in db_questions
    ]

    max_score = sum(dbq.points for dbq in db_questions)

    session = QuizSession(
        user_id=user_id,
        session_type="exam" if exam_format else "practice",
        exam_format=exam_format,
        skill_ids=skill_ids_to_query,
        questions=session_questions,
        total_questions=len(db_questions),
        max_score=max_score,
    )
    db.add(session)
    await db.flush()

    # Build output (hide correct answers)
    questions_out = []
    for dbq in db_questions:
        q_out = {
            "id": dbq.id,
            "question_type": dbq.question_type,
            "question_latex": dbq.question_latex,
            "skill_id": dbq.skill_id,
            "skill_ids": dbq.skill_ids or [dbq.skill_id],
            "formula_ids": dbq.formula_ids or [],
            "difficulty": dbq.difficulty,
            "points": dbq.points,
        }
        if dbq.question_type == "mcq":
            q_out["choices"] = dbq.choices
        elif dbq.question_type == "true_false":
            # Send statements text only, not correct values
            q_out["statements"] = [
                {"text": s["text"]} for s in dbq.statements
            ]
        # short_answer: no extra data needed (student types answer)
        questions_out.append(q_out)

    result_data = {
        "session_id": session.id,
        "questions": questions_out,
        "total_questions": len(db_questions),
        "session_type": session.session_type,
        "max_score": max_score,
    }

    # Include adaptive info if auto-difficulty was used
    if adaptive_info:
        result_data["adaptive"] = adaptive_info

    return result_data


# ── Submit Answer ────────────────────────────────────────
async def submit_quiz_answer(
    db: AsyncSession,
    session_id: int,
    question_id: int,
    selected_index: int | None = None,
    tf_answers: list[bool] | None = None,
    text_answer: str | None = None,
    user_id: int = 1,
) -> dict:
    """Submit student's answer. Handles all 3 question types with THPT QG scoring."""
    # Get session (scoped to the requesting user — chống IDOR)
    result = await db.execute(
        select(QuizSession).where(
            QuizSession.id == session_id,
            QuizSession.user_id == user_id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        return {"error": "Không tìm thấy phiên kiểm tra."}

    # Question must belong to this session
    questions = list(session.questions)
    q_info = next((q for q in questions if q["question_id"] == question_id), None)
    if q_info is None:
        return {"error": "Câu hỏi không thuộc phiên kiểm tra này."}
    if q_info.get("answered"):
        return {"error": "Câu hỏi này đã được trả lời."}

    # Get question
    result = await db.execute(
        select(QuizQuestion).where(QuizQuestion.id == question_id)
    )
    question = result.scalar_one_or_none()
    if not question:
        return {"error": "Không tìm thấy câu hỏi."}

    # ── Score based on question type ──
    q_type = question.question_type
    points_earned = 0.0
    is_correct = False
    response_extra = {}

    if q_type == "mcq":
        is_correct = (selected_index == question.correct_index)
        points_earned = 0.25 if is_correct else 0.0
        response_extra["correct_index"] = question.correct_index

    elif q_type == "true_false":
        correct_bools = [s["correct"] for s in question.statements]
        # Không trả lời = không điểm (đồng bộ với cách chấm đề thi)
        student_bools = (
            tf_answers if tf_answers is not None
            else [None] * len(correct_bools)
        )
        points_earned, num_correct = score_true_false(student_bools, correct_bools)
        is_correct = (num_correct == len(correct_bools))
        response_extra["correct_statements"] = correct_bools

    elif q_type == "short_answer":
        student_text = text_answer or ""
        points_earned, is_correct = score_short_answer(student_text, question.correct_answer or "")
        response_extra["correct_answer"] = question.correct_answer

    # Update session question list
    q_info["answered"] = True
    q_info["is_correct"] = is_correct
    q_info["points_earned"] = points_earned

    # CRITICAL: SQLAlchemy does NOT auto-detect mutations in JSON columns.
    from sqlalchemy.orm.attributes import flag_modified
    session.questions = questions
    flag_modified(session, "questions")

    session.current_index += 1
    session.total_score = sum(q.get("points_earned", 0) for q in questions if q["answered"])
    if is_correct:
        session.correct_count += 1

    # Check if session is complete
    answered_count = sum(1 for q in questions if q["answered"])
    if answered_count >= session.total_questions:
        session.is_completed = True
        session.completed_at = datetime.utcnow()

    # Update mastery via BKT
    new_mastery = None
    q_skill_ids = question.skill_ids or ([question.skill_id] if question.skill_id else [])
    if q_skill_ids:
        mastery_updates = await update_mastery(
            db,
            {
                skill_id: {"passed": is_correct}
                for skill_id in q_skill_ids
            },
            user_id=user_id,
        )
        if isinstance(mastery_updates, dict):
            primary_skill = q_skill_ids[0]
            new_mastery = mastery_updates.get(primary_skill)
        else:
            new_mastery = mastery_updates

    await db.flush()

    return {
        "is_correct": is_correct,
        "points_earned": points_earned,
        "max_points": question.points,
        **response_extra,
        "explanation": question.explanation,
        "new_mastery": round(new_mastery, 3) if new_mastery else None,
        "session_progress": {
            "current": answered_count,
            "total": session.total_questions,
            "score_so_far": round(session.total_score, 2),
            "max_score": session.max_score,
        },
    }


# ── Get Quiz Result ──────────────────────────────────────
async def get_quiz_result(db: AsyncSession, session_id: int, user_id: int = 1) -> dict:
    """Get the final result of a completed quiz session with THPT QG breakdown."""
    result = await db.execute(
        select(QuizSession).where(
            QuizSession.id == session_id,
            QuizSession.user_id == user_id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        return {"error": "Không tìm thấy phiên kiểm tra."}

    # Aggregate by skill and by question type
    skill_stats = {}
    part_scores = {"mcq": 0.0, "true_false": 0.0, "short_answer": 0.0}

    for q_info in session.questions:
        qid = q_info["question_id"]
        q_type = q_info.get("question_type", "mcq")
        pts = q_info.get("points_earned", 0.0)
        part_scores[q_type] = part_scores.get(q_type, 0.0) + pts

        qresult = await db.execute(
            select(QuizQuestion).where(QuizQuestion.id == qid)
        )
        question = qresult.scalar_one_or_none()
        if question:
            q_skill_ids = question.skill_ids or ([question.skill_id] if question.skill_id else [])
            for sid in q_skill_ids:
                if sid not in skill_stats:
                    skill_info = SKILLS.get(sid, {})
                    mastery_rec = await get_or_create_mastery(db, sid, user_id)
                    skill_stats[sid] = {
                        "skill_id": sid,
                        "skill_name": skill_info.get("name", sid),
                        "correct": 0,
                        "total": 0,
                        "mastery": round(mastery_rec.p_mastery, 3),
                    }
                skill_stats[sid]["total"] += 1
                if q_info.get("is_correct"):
                    skill_stats[sid]["correct"] += 1

    score_pct = (session.total_score / session.max_score * 100) if session.max_score else 0

    return {
        "session_id": session.id,
        "total_questions": session.total_questions,
        "total_score": round(session.total_score, 2),
        "max_score": session.max_score,
        "score_percent": round(score_pct, 1),
        "part_scores": {k: round(v, 2) for k, v in part_scores.items()},
        "skill_results": list(skill_stats.values()),
    }
