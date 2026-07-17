"""Spaced Repetition Service — manages review cards and due scheduling."""

from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.models import SpacedRepetitionCard, QuizQuestion
from app.spaced_repetition.engine import sm2_update
from app.quiz.generator import generate_questions
from app.knowledge_tracing.skill_graph import SKILLS


async def get_or_create_card(db: AsyncSession, skill_id: str, user_id: int = 1) -> SpacedRepetitionCard:
    """Get existing SR card or create a new one for the skill."""
    result = await db.execute(
        select(SpacedRepetitionCard).where(
            SpacedRepetitionCard.user_id == user_id,
            SpacedRepetitionCard.skill_id == skill_id,
        )
    )
    card = result.scalar_one_or_none()

    if card is None:
        card = SpacedRepetitionCard(
            user_id=user_id,
            skill_id=skill_id,
            easiness_factor=2.5,
            interval=0,
            repetitions=0,
            next_review=datetime.utcnow(),
        )
        db.add(card)
        await db.flush()

    return card


async def get_due_cards(db: AsyncSession, user_id: int = 1) -> list[dict]:
    """Get all cards that are due for review (next_review ≤ now)."""
    now = datetime.utcnow()
    result = await db.execute(
        select(SpacedRepetitionCard).where(
            SpacedRepetitionCard.user_id == user_id,
            SpacedRepetitionCard.next_review <= now,
        )
    )
    cards = result.scalars().all()

    due_list = []
    for card in cards:
        skill_info = SKILLS.get(card.skill_id, {})
        days_overdue = (now - card.next_review).days

        # Generate a fresh review question for this skill
        qs = generate_questions(card.skill_id, difficulty=1, count=1)
        if not qs:
            continue

        q = qs[0]
        q_skill_ids = q.get("skill_ids") or [q["skill_id"]]
        # Save question to DB
        dbq = QuizQuestion(
            skill_id=q["skill_id"],
            skill_ids=q_skill_ids,
            formula_ids=q.get("formula_ids", []),
            difficulty=q["difficulty"],
            question_latex=q["question_latex"],
            choices=q["choices"],
            correct_index=q["correct_index"],
            explanation=q.get("explanation", ""),
            sympy_expr=q.get("sympy_expr", ""),
        )
        db.add(dbq)
        await db.flush()

        due_list.append({
            "card_id": card.id,
            "skill_id": card.skill_id,
            "skill_name": skill_info.get("name", card.skill_id),
            "question": {
                "id": dbq.id,
                "question_latex": dbq.question_latex,
                "choices": dbq.choices,
                "skill_id": dbq.skill_id,
                "skill_ids": dbq.skill_ids or [dbq.skill_id],
                "formula_ids": dbq.formula_ids or [],
                "difficulty": dbq.difficulty,
            },
            "days_overdue": max(days_overdue, 0),
        })

    return due_list


async def review_card(
    db: AsyncSession,
    card_id: int,
    question_id: int,
    selected_index: int,
    user_id: int = 1,
) -> dict:
    """Submit a review result for a card and update SM-2 parameters.

    Card lookup is scoped to ``user_id`` so users cannot review each
    other's cards.
    """
    result = await db.execute(
        select(SpacedRepetitionCard).where(
            SpacedRepetitionCard.id == card_id,
            SpacedRepetitionCard.user_id == user_id,
        )
    )
    card = result.scalar_one_or_none()
    if not card:
        return {"error": "Không tìm thấy thẻ ôn tập."}

    q_result = await db.execute(
        select(QuizQuestion).where(QuizQuestion.id == question_id)
    )
    question = q_result.scalar_one_or_none()
    if not question:
        return {"error": "Không tìm thấy câu hỏi."}

    is_correct = (question.correct_index == selected_index)
    quality = 4 if is_correct else 1

    # Apply SM-2 algorithm
    update = sm2_update(
        easiness=card.easiness_factor,
        interval=card.interval,
        repetitions=card.repetitions,
        quality=quality,
    )

    card.easiness_factor = update["easiness"]
    card.interval = update["interval"]
    card.repetitions = update["repetitions"]
    card.next_review = update["next_review"]
    card.last_reviewed = datetime.utcnow()

    await db.flush()

    if quality >= 3:
        msg = f"✅ Tốt lắm! Ôn lại sau {update['interval']} ngày."
    else:
        msg = "📝 Cần luyện thêm. Sẽ ôn lại ngày mai!"

    return {
        "next_review_days": update["interval"],
        "new_interval": update["interval"],
        "message": msg,
        "is_correct": is_correct,
        "correct_index": question.correct_index,
        "explanation": question.explanation,
    }


async def ensure_cards_for_attempted_skills(db: AsyncSession, skill_id: str, user_id: int = 1):
    """Create an SR card for a skill if student has attempted it (auto-track)."""
    await get_or_create_card(db, skill_id, user_id)
