"""Adaptive Difficulty Engine — automatically selects optimal difficulty.

Uses BKT mastery + recent performance to find the "Zone of Proximal Development":
the difficulty level where students succeed ~60-70% of the time (optimal learning).

Algorithm:
1. Base difficulty from mastery level
2. Adjust based on recent streak (3 correct → harder, 2 wrong → easier)
3. Recommend next skill based on prerequisite readiness
"""

from app.knowledge_tracing.bkt import BKTModel
from app.knowledge_tracing.skill_graph import SKILLS

bkt = BKTModel()

# ── Mastery → Base Difficulty mapping ────────────────────
_MASTERY_TO_DIFFICULTY = [
    (0.30, 1),  # beginner  → easy
    (0.55, 1),  # developing (low)  → easy
    (0.70, 2),  # developing (high) → medium
    (0.85, 2),  # proficient → medium-hard
    (1.01, 3),  # mastered → hard
]


class AdaptiveDifficultyEngine:
    """Selects optimal quiz difficulty and recommends next skills to study."""

    def recommend_difficulty(
        self,
        mastery: float,
        recent_results: list[bool] | None = None,
    ) -> int:
        """
        Recommend difficulty level (1-3) based on mastery and recent performance.

        Args:
            mastery: Current BKT mastery probability [0, 1]
            recent_results: List of recent correct/incorrect results (newest first)

        Returns:
            Difficulty level: 1 (easy), 2 (medium), 3 (hard)
        """
        # Step 1: Base difficulty from mastery
        base_diff = 1
        for threshold, diff in _MASTERY_TO_DIFFICULTY:
            if mastery < threshold:
                base_diff = diff
                break

        # Step 2: Adjust based on recent streak
        if recent_results and len(recent_results) >= 2:
            # Count consecutive correct/incorrect from most recent
            streak_correct = 0
            streak_wrong = 0

            for result in recent_results[:5]:  # Look at last 5
                if result:
                    streak_correct += 1
                    if streak_wrong > 0:
                        break
                else:
                    streak_wrong += 1
                    if streak_correct > 0:
                        break

            # 3+ correct in a row → bump difficulty up
            if streak_correct >= 3:
                base_diff = min(3, base_diff + 1)
            # 2+ wrong in a row → drop difficulty down
            elif streak_wrong >= 2:
                base_diff = max(1, base_diff - 1)

        return base_diff

    def recommend_next_skill(
        self,
        masteries: dict[str, float],
        current_skill_id: str | None = None,
    ) -> str | None:
        """
        Recommend the next skill to study based on prerequisite readiness.

        Strategy:
        1. Find skills where all prerequisites are ≥ 0.5 mastery (ready to learn)
        2. Among those, pick the one with lowest mastery (most room to grow)
        3. Exclude already-mastered skills (≥ 0.85)

        Args:
            masteries: {skill_id: p_mastery} dict
            current_skill_id: Currently studying skill (will deprioritize)

        Returns:
            Recommended skill_id, or None if all mastered
        """
        candidates = []

        for skill_id, info in SKILLS.items():
            p = masteries.get(skill_id, 0.0)

            # Skip already mastered skills
            if p >= 0.85:
                continue

            # Check if prerequisites are satisfied
            prereqs = info.get("prerequisites", [])
            prereqs_ready = all(
                masteries.get(pre, 0.0) >= 0.5 for pre in prereqs
            )

            if not prereqs_ready:
                continue

            # Deprioritize current skill (slight penalty)
            priority_bonus = 0 if skill_id != current_skill_id else 0.1

            candidates.append({
                "skill_id": skill_id,
                "mastery": p + priority_bonus,
                "chapter": info["chapter"],
            })

        if not candidates:
            return None

        # Sort by mastery ascending (focus on weakest ready skill)
        candidates.sort(key=lambda c: c["mastery"])
        return candidates[0]["skill_id"]

    def get_adaptive_summary(
        self,
        skill_id: str,
        mastery: float,
        recent_results: list[bool] | None = None,
        masteries: dict[str, float] | None = None,
    ) -> dict:
        """
        Get a complete adaptive recommendation.

        Returns:
            {
                "recommended_difficulty": 1-3,
                "difficulty_label": "Dễ" | "Trung bình" | "Khó",
                "mastery_level": "beginner" | ... | "mastered",
                "next_skill_id": str | None,
                "next_skill_name": str | None,
                "explanation": str,
            }
        """
        difficulty = self.recommend_difficulty(mastery, recent_results)
        level = bkt.get_mastery_level(mastery)
        diff_labels = {1: "Dễ", 2: "Trung bình", 3: "Khó"}

        next_skill = None
        next_skill_name = None
        if masteries:
            next_skill = self.recommend_next_skill(masteries, skill_id)
            if next_skill:
                next_skill_name = SKILLS.get(next_skill, {}).get("name", next_skill)

        # Build explanation
        reasons = []
        if mastery < 0.3:
            reasons.append(f"Mastery thấp ({mastery:.0%}) → bắt đầu từ bài dễ")
        elif mastery >= 0.85:
            reasons.append(f"Mastery cao ({mastery:.0%}) → thử thách nâng cao")
        else:
            reasons.append(f"Mastery {mastery:.0%}")

        if recent_results:
            recent_correct = sum(recent_results[:5])
            recent_total = min(len(recent_results), 5)
            reasons.append(f"Gần đây: {recent_correct}/{recent_total} đúng")

        return {
            "recommended_difficulty": difficulty,
            "difficulty_label": diff_labels.get(difficulty, "Trung bình"),
            "mastery_level": level,
            "next_skill_id": next_skill,
            "next_skill_name": next_skill_name,
            "explanation": " | ".join(reasons),
        }


# ── Singleton ────────────────────────────────────────────
_engine = AdaptiveDifficultyEngine()


def get_adaptive_engine() -> AdaptiveDifficultyEngine:
    return _engine
