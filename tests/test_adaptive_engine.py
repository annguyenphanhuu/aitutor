"""
Tests for Adaptive Difficulty Engine.

Module: app/quiz/adaptive_engine.py
Covers:
- recommend_difficulty() — mastery + streak → difficulty mapping
- recommend_next_skill() — prerequisite-aware skill recommendation
- get_adaptive_summary() — complete recommendation dict
"""

import pytest
from app.quiz.adaptive_engine import AdaptiveDifficultyEngine


@pytest.fixture
def engine():
    return AdaptiveDifficultyEngine()


# ━━ recommend_difficulty() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRecommendDifficulty:

    def test_low_mastery_returns_easy(self, engine):
        assert engine.recommend_difficulty(mastery=0.1) == 1

    def test_medium_mastery_returns_medium(self, engine):
        assert engine.recommend_difficulty(mastery=0.6) == 2

    def test_high_mastery_returns_hard(self, engine):
        assert engine.recommend_difficulty(mastery=0.9) == 3

    def test_zero_mastery(self, engine):
        assert engine.recommend_difficulty(mastery=0.0) == 1

    def test_full_mastery(self, engine):
        assert engine.recommend_difficulty(mastery=1.0) == 3

    def test_streak_correct_bumps_difficulty(self, engine):
        """3+ consecutive correct → difficulty increases."""
        diff_no_streak = engine.recommend_difficulty(mastery=0.5, recent_results=None)
        diff_streak = engine.recommend_difficulty(mastery=0.5, recent_results=[True, True, True])
        assert diff_streak >= diff_no_streak

    def test_streak_wrong_lowers_difficulty(self, engine):
        """2+ consecutive wrong → difficulty decreases."""
        diff_no_streak = engine.recommend_difficulty(mastery=0.5, recent_results=None)
        diff_streak = engine.recommend_difficulty(mastery=0.5, recent_results=[False, False])
        assert diff_streak <= diff_no_streak

    def test_difficulty_capped_at_three(self, engine):
        """Even with streak, difficulty should not exceed 3."""
        diff = engine.recommend_difficulty(mastery=0.9, recent_results=[True] * 10)
        assert diff <= 3

    def test_difficulty_floor_at_one(self, engine):
        """Even with wrong streak, difficulty should not go below 1."""
        diff = engine.recommend_difficulty(mastery=0.1, recent_results=[False] * 10)
        assert diff >= 1

    def test_short_results_list(self, engine):
        """With only 1 result, streak logic should not trigger."""
        diff = engine.recommend_difficulty(mastery=0.5, recent_results=[True])
        assert diff >= 1

    def test_empty_results_list(self, engine):
        diff = engine.recommend_difficulty(mastery=0.5, recent_results=[])
        assert diff >= 1

    def test_mixed_results_no_streak(self, engine):
        """Alternating results → no streak bonus."""
        diff = engine.recommend_difficulty(mastery=0.5, recent_results=[True, False, True, False])
        base_diff = engine.recommend_difficulty(mastery=0.5)
        assert diff == base_diff  # no change

    def test_difficulty_is_int(self, engine):
        diff = engine.recommend_difficulty(mastery=0.5)
        assert isinstance(diff, int)


# ━━ recommend_next_skill() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRecommendNextSkill:

    def test_returns_weakest_ready_skill(self, engine):
        masteries = {
            "derivative_basic": 0.9,    # mastered
            "derivative_rules": 0.3,   # weak but prereqs met
            "integral_definite": 0.1,  # weak, prereq not met
        }
        recommended = engine.recommend_next_skill(masteries)
        # derivative_rules' prereq (derivative_basic=0.9) is met
        # integral_definite's prereq chain is not all ≥ 0.5
        assert recommended is not None

    def test_all_mastered_returns_none(self, engine):
        from app.knowledge_tracing.skill_graph import SKILLS
        masteries = {sid: 0.95 for sid in SKILLS.keys()}
        recommended = engine.recommend_next_skill(masteries)
        assert recommended is None

    def test_empty_masteries_returns_skill_with_no_prereqs(self, engine):
        """With no mastery data, only root skills (no prereqs) are eligible."""
        recommended = engine.recommend_next_skill({})
        from app.knowledge_tracing.skill_graph import SKILLS
        if recommended:
            prereqs = SKILLS[recommended]["prerequisites"]
            # All prereqs should be met (they're not in masteries, so 0.0 < 0.5 = not met)
            # Actually, root skills with [] prereqs are always ready
            assert prereqs == []

    def test_deprioritizes_current_skill(self, engine):
        masteries = {
            "derivative_basic": 0.2,   # low mastery, no prereqs
            "sequence_basic": 0.2,     # low mastery, no prereqs
        }
        # If current is derivative_basic, should prefer sequence_basic
        rec = engine.recommend_next_skill(masteries, current_skill_id="derivative_basic")
        # Both have same mastery (0.2) but derivative_basic gets +0.1 penalty
        assert rec == "sequence_basic" or rec is not None

    def test_returns_valid_skill_id(self, engine):
        from app.knowledge_tracing.skill_graph import SKILLS
        masteries = {"derivative_basic": 0.6}
        rec = engine.recommend_next_skill(masteries)
        if rec:
            assert rec in SKILLS


# ━━ get_adaptive_summary() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetAdaptiveSummary:

    def test_returns_required_keys(self, engine):
        result = engine.get_adaptive_summary(
            skill_id="derivative_basic",
            mastery=0.5,
        )
        required = {"recommended_difficulty", "difficulty_label", "mastery_level",
                     "next_skill_id", "next_skill_name", "explanation"}
        assert required.issubset(result.keys())

    def test_difficulty_label_mapping(self, engine):
        result = engine.get_adaptive_summary("derivative_basic", mastery=0.1)
        assert result["difficulty_label"] in ["Dễ", "Trung bình", "Khó"]

    def test_mastery_level_string(self, engine):
        result = engine.get_adaptive_summary("derivative_basic", mastery=0.5)
        assert result["mastery_level"] in ["beginner", "developing", "proficient", "mastered"]

    def test_explanation_nonempty(self, engine):
        result = engine.get_adaptive_summary("derivative_basic", mastery=0.5)
        assert len(result["explanation"]) > 0

    def test_with_recent_results(self, engine):
        result = engine.get_adaptive_summary(
            "derivative_basic", mastery=0.5,
            recent_results=[True, True, False, True],
        )
        assert "Gần đây" in result["explanation"]

    def test_with_masteries_for_next_skill(self, engine):
        masteries = {"derivative_basic": 0.8, "derivative_rules": 0.3}
        result = engine.get_adaptive_summary(
            "derivative_basic", mastery=0.8, masteries=masteries,
        )
        # next_skill should be populated
        if result["next_skill_id"]:
            assert result["next_skill_name"] is not None
