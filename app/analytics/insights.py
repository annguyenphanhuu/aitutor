"""Peer Insights Service — aggregate analytics for student comparison."""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.models import InteractionLog, SkillMastery, QuizSession
from app.knowledge_tracing.skill_graph import SKILLS
from app.knowledge_tracing.bkt import BKTModel


bkt = BKTModel()


async def get_insights(db: AsyncSession, user_id: int = 1) -> list[dict]:
    """Generate insight messages based on user's data."""
    insights = []

    # 1. Most difficult skills (lowest avg mastery)
    result = await db.execute(
        select(SkillMastery).where(SkillMastery.user_id == user_id)
    )
    all_masteries = result.scalars().all()

    if all_masteries:
        sorted_skills = sorted(all_masteries, key=lambda m: m.p_mastery)

        # Weakest skill
        weakest = sorted_skills[0]
        skill_info = SKILLS.get(weakest.skill_id, {})
        insights.append({
            "insight_type": "weakness",
            "message": (
                f"📊 Kỹ năng yếu nhất của em hiện tại là "
                f"**{skill_info.get('name', weakest.skill_id)}** "
                f"(mastery: {weakest.p_mastery:.0%}). "
                f"Hãy tập trung ôn luyện thêm nhé!"
            ),
            "data": {
                "skill_id": weakest.skill_id,
                "skill_name": skill_info.get("name", weakest.skill_id),
                "mastery": round(weakest.p_mastery, 3),
            },
        })

        # Strongest skill
        strongest = sorted_skills[-1]
        skill_info_s = SKILLS.get(strongest.skill_id, {})
        if strongest.p_mastery > 0.7:
            insights.append({
                "insight_type": "streak",
                "message": (
                    f"🌟 Em đang rất giỏi **{skill_info_s.get('name', strongest.skill_id)}** "
                    f"(mastery: {strongest.p_mastery:.0%}). Tuyệt vời!"
                ),
                "data": {
                    "skill_id": strongest.skill_id,
                    "skill_name": skill_info_s.get("name", strongest.skill_id),
                    "mastery": round(strongest.p_mastery, 3),
                },
            })

    # 2. Total practice stats
    result = await db.execute(
        select(func.count(InteractionLog.id)).where(InteractionLog.user_id == user_id)
    )
    total_interactions = result.scalar() or 0

    result = await db.execute(
        select(func.count(InteractionLog.id)).where(
            InteractionLog.user_id == user_id,
            InteractionLog.is_correct.is_(True),
        )
    )
    correct_interactions = result.scalar() or 0

    if total_interactions > 0:
        accuracy = correct_interactions / total_interactions
        insights.append({
            "insight_type": "comparison",
            "message": (
                f"📈 Em đã luyện tập **{total_interactions}** câu hỏi "
                f"với tỷ lệ đúng **{accuracy:.0%}**. "
                + ("Rất ấn tượng!" if accuracy > 0.7 else "Cố lên, em sẽ tiến bộ nhanh thôi!")
            ),
            "data": {
                "total": total_interactions,
                "correct": correct_interactions,
                "accuracy": round(accuracy, 3),
            },
        })

    # 3. Quiz sessions completed
    result = await db.execute(
        select(func.count(QuizSession.id)).where(
            QuizSession.user_id == user_id,
            QuizSession.is_completed.is_(True),
        )
    )
    completed_quizzes = result.scalar() or 0

    if completed_quizzes > 0:
        result = await db.execute(
            select(func.avg(QuizSession.correct_count * 100.0 / QuizSession.total_questions))
            .where(
                QuizSession.user_id == user_id,
                QuizSession.is_completed.is_(True),
            )
        )
        avg_score = result.scalar() or 0

        insights.append({
            "insight_type": "comparison",
            "message": (
                f"🎯 Em đã hoàn thành **{completed_quizzes}** bài kiểm tra "
                f"với điểm trung bình **{avg_score:.0f}%**."
            ),
            "data": {
                "completed_quizzes": completed_quizzes,
                "avg_score": round(avg_score, 1),
            },
        })

    # 4. Skill coverage
    skills_attempted = len([m for m in all_masteries if m.total_attempts > 0]) if all_masteries else 0
    total_skills = len(SKILLS)

    insights.append({
        "insight_type": "comparison",
        "message": (
            f"📚 Em đã luyện tập **{skills_attempted}/{total_skills}** kỹ năng "
            f"trong chương trình Toán 12. "
            + ("Hãy thử thêm các chuyên đề khác!" if skills_attempted < total_skills else "Phủ sóng toàn bộ! 🎉")
        ),
        "data": {
            "attempted": skills_attempted,
            "total": total_skills,
            "coverage_percent": round(skills_attempted / total_skills * 100, 1) if total_skills else 0,
        },
    })

    return insights
