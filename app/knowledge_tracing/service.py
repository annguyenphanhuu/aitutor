"""Knowledge Tracing service — bridges BKT with the database."""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models import SkillMastery
from app.knowledge_tracing.bkt import BKTModel
from app.knowledge_tracing.skill_graph import SKILLS, get_prerequisites, find_weak_prerequisites


bkt = BKTModel()


async def get_or_create_mastery(db: AsyncSession, skill_id: str, user_id: int = 1) -> SkillMastery:
    """Get existing mastery record or create a new one for the user."""
    result = await db.execute(
        select(SkillMastery).where(
            SkillMastery.user_id == user_id,
            SkillMastery.skill_id == skill_id,
        )
    )
    mastery = result.scalar_one_or_none()

    if mastery is None:
        skill_info = SKILLS.get(skill_id, {})
        mastery = SkillMastery(
            user_id=user_id,
            skill_id=skill_id,
            skill_name=skill_info.get("name", skill_id),
            p_mastery=bkt.p_init,
        )
        db.add(mastery)
        await db.flush()

    return mastery


def _coerce_skills_assessed(
    skill_or_assessment,
    is_correct: bool | None = None,
) -> dict[str, bool]:
    """Normalize legacy and multi-skill assessment shapes.

    Accepted inputs:
      - "skill_id", is_correct
      - {"skill_id": {"passed": bool, ...}}
      - {"skills_assessed": {"skill_id": {"passed": bool, ...}}}
      - [{"skill_id": "...", "is_correct": bool}, ...]
    """
    if isinstance(skill_or_assessment, str):
        return {skill_or_assessment: bool(is_correct)} if skill_or_assessment else {}

    if isinstance(skill_or_assessment, dict):
        raw = skill_or_assessment.get("skills_assessed", skill_or_assessment)
        if not isinstance(raw, dict):
            return {}

        normalized: dict[str, bool] = {}
        for skill_id, value in raw.items():
            if not skill_id:
                continue
            if isinstance(value, bool):
                normalized[str(skill_id)] = value
            elif isinstance(value, dict):
                passed = value.get("passed")
                if passed is None:
                    passed = value.get("is_correct")
                if passed is None:
                    passed = value.get("correct")
                normalized[str(skill_id)] = bool(passed)
            else:
                normalized[str(skill_id)] = bool(value)
        return normalized

    if isinstance(skill_or_assessment, list):
        normalized: dict[str, bool] = {}
        for item in skill_or_assessment:
            if not isinstance(item, dict):
                continue
            skill_id = item.get("skill_id")
            if not skill_id:
                continue
            passed = item.get("passed")
            if passed is None:
                passed = item.get("is_correct")
            if passed is None:
                passed = item.get("correct")
            normalized[str(skill_id)] = bool(passed)
        return normalized

    return {}


async def update_mastery(
    db: AsyncSession,
    skill_or_assessment,
    is_correct: bool | None = None,
    user_id: int = 1,
) -> float | dict[str, float]:
    """Update BKT mastery after a response.

    Legacy single-skill calls return ``float``. Multi-skill assessment calls
    return ``{skill_id: new_p_mastery}``.
    """
    skills_assessed = _coerce_skills_assessed(skill_or_assessment, is_correct)
    if not skills_assessed:
        return {} if not isinstance(skill_or_assessment, str) else bkt.p_init

    updated: dict[str, float] = {}
    for skill_id, passed in skills_assessed.items():
        mastery = await get_or_create_mastery(db, skill_id, user_id)
        new_p = bkt.update(mastery.p_mastery, passed)
        mastery.p_mastery = new_p
        mastery.total_attempts += 1
        if passed:
            mastery.correct_attempts += 1
        updated[skill_id] = new_p

    await db.flush()

    if isinstance(skill_or_assessment, str):
        return updated.get(skill_or_assessment, bkt.p_init)
    return updated


async def update_masteries_from_assessment(
    db: AsyncSession,
    skills_assessed: dict,
    user_id: int = 1,
) -> dict[str, float]:
    """Explicit multi-skill BKT update helper for assessor JSON output."""
    result = await update_mastery(db, skills_assessed, user_id=user_id)
    return result if isinstance(result, dict) else {}


async def get_all_masteries(db: AsyncSession, user_id: int = 1) -> dict[str, float]:
    """Get all mastery levels as {skill_id: p_mastery} for a user."""
    result = await db.execute(
        select(SkillMastery).where(SkillMastery.user_id == user_id)
    )
    records = result.scalars().all()
    return {r.skill_id: r.p_mastery for r in records}


async def get_mastery_profile(db: AsyncSession, user_id: int = 1) -> list[dict]:
    """Get full mastery profile with skill names and levels for a user."""
    result = await db.execute(
        select(SkillMastery).where(SkillMastery.user_id == user_id)
    )
    records = result.scalars().all()

    profile = []
    for r in records:
        profile.append({
            "skill_id": r.skill_id,
            "skill_name": r.skill_name,
            "p_mastery": round(r.p_mastery, 3),
            "level": bkt.get_mastery_level(r.p_mastery),
            "total_attempts": r.total_attempts,
            "correct_attempts": r.correct_attempts,
        })
    return profile


async def identify_gaps(db: AsyncSession, target_skill: str, user_id: int = 1) -> list[dict]:
    """Identify prerequisite knowledge gaps for a target skill."""
    masteries = await get_all_masteries(db, user_id)
    weak = find_weak_prerequisites(masteries, target_skill)

    gaps = []
    for skill_id in weak:
        skill_info = SKILLS.get(skill_id, {})
        gaps.append({
            "skill_id": skill_id,
            "skill_name": skill_info.get("name", skill_id),
            "current_mastery": round(masteries.get(skill_id, 0.0), 3),
            "recommendation": f"Ôn lại: {skill_info.get('name', skill_id)}",
        })
    return gaps


async def get_recent_results(db: AsyncSession, skill_id: str, user_id: int = 1, limit: int = 10) -> list[bool]:
    """Get recent quiz/interaction results for adaptive difficulty."""
    from app.db.models import InteractionLog
    result = await db.execute(
        select(InteractionLog.is_correct)
        .where(
            InteractionLog.user_id == user_id,
            InteractionLog.skill_id == skill_id,
            InteractionLog.is_correct.isnot(None),
        )
        .order_by(InteractionLog.timestamp.desc())
        .limit(limit)
    )
    rows = result.scalars().all()
    return list(rows)


async def batch_update_mastery_from_exam(
    db: AsyncSession,
    exam_items: list[dict],
    user_id: int,
) -> dict[str, float]:
    """
    Batch-update BKT mastery for all skills encountered in an exam.

    Processes each skill's answers sequentially to preserve the BKT Markov
    chain (each update depends on the previous p_mastery value).
    Performs a single DB flush at the end for efficiency.

    Args:
        exam_items: list of { "skill_id": str, "is_correct": bool } or
                    { "skill_ids": [str], "is_correct": bool } or
                    { "skills_assessed": {...} }
        user_id:    the student's user ID

    Returns:
        dict mapping skill_id -> new p_mastery (rounded to 4 decimal places)
    """
    from collections import defaultdict

    # Group answers by skill_id, preserving order of occurrence
    skill_answers: dict[str, list[bool]] = defaultdict(list)
    for item in exam_items:
        assessed = _coerce_skills_assessed(item.get("skills_assessed", {}))
        if assessed:
            for sid, passed in assessed.items():
                skill_answers[sid].append(passed)
            continue

        if item.get("skill_ids"):
            for sid in item.get("skill_ids") or []:
                if sid:
                    skill_answers[sid].append(bool(item.get("is_correct", False)))
            continue

        sid = item.get("skill_id")
        if sid:
            skill_answers[sid].append(bool(item.get("is_correct", False)))

    updated: dict[str, float] = {}

    for skill_id, answers in skill_answers.items():
        mastery = await get_or_create_mastery(db, skill_id, user_id)

        # Apply BKT updates sequentially (Markov chain — order matters)
        p = mastery.p_mastery
        for is_correct in answers:
            p = bkt.update(p, is_correct)

        # Persist final mastery value and update attempt counters
        mastery.p_mastery = p
        mastery.total_attempts += len(answers)
        mastery.correct_attempts += sum(1 for a in answers if a)

        updated[skill_id] = round(p, 4)

    await db.flush()
    return updated

