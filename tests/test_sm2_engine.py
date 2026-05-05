"""
Tests for SM-2 Spaced Repetition Engine.

Module: app/spaced_repetition/engine.py
Covers:
- sm2_update() with various quality ratings (0-5)
- Easiness factor bounds (≥ 1.3)
- Interval calculation logic
- Repetition counter behaviour
- quality_from_quiz() conversion
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
from app.spaced_repetition.engine import sm2_update, quality_from_quiz


# ━━ sm2_update() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSM2Update:
    """Test the SM-2 algorithm core."""

    def test_first_correct_interval_is_one(self):
        result = sm2_update(easiness=2.5, interval=0, repetitions=0, quality=4)
        assert result["interval"] == 1
        assert result["repetitions"] == 1

    def test_second_correct_interval_is_six(self):
        result = sm2_update(easiness=2.5, interval=1, repetitions=1, quality=4)
        assert result["interval"] == 6
        assert result["repetitions"] == 2

    def test_third_correct_uses_ef_multiplier(self):
        ef = 2.5
        result = sm2_update(easiness=ef, interval=6, repetitions=2, quality=4)
        expected_interval = round(6 * result["easiness"])
        assert result["interval"] == expected_interval
        assert result["repetitions"] == 3

    def test_incorrect_resets_repetitions(self):
        result = sm2_update(easiness=2.5, interval=10, repetitions=5, quality=2)
        assert result["repetitions"] == 0
        assert result["interval"] == 1

    def test_incorrect_quality_zero(self):
        result = sm2_update(easiness=2.5, interval=10, repetitions=5, quality=0)
        assert result["repetitions"] == 0
        assert result["interval"] == 1

    def test_incorrect_quality_one(self):
        result = sm2_update(easiness=2.5, interval=10, repetitions=5, quality=1)
        assert result["repetitions"] == 0

    def test_quality_three_is_correct(self):
        result = sm2_update(easiness=2.5, interval=1, repetitions=1, quality=3)
        assert result["repetitions"] == 2  # incremented

    def test_easiness_factor_floor(self):
        """EF should never drop below 1.3."""
        result = sm2_update(easiness=1.3, interval=1, repetitions=1, quality=0)
        assert result["easiness"] >= 1.3

    def test_quality_five_increases_ef(self):
        result = sm2_update(easiness=2.5, interval=1, repetitions=1, quality=5)
        assert result["easiness"] >= 2.5

    def test_quality_zero_decreases_ef(self):
        result = sm2_update(easiness=2.5, interval=1, repetitions=1, quality=0)
        assert result["easiness"] < 2.5

    def test_next_review_is_datetime(self):
        result = sm2_update(easiness=2.5, interval=0, repetitions=0, quality=4)
        assert isinstance(result["next_review"], datetime)

    def test_next_review_in_future(self):
        before = datetime.utcnow()
        result = sm2_update(easiness=2.5, interval=0, repetitions=0, quality=4)
        assert result["next_review"] > before

    def test_result_has_required_keys(self):
        result = sm2_update(easiness=2.5, interval=0, repetitions=0, quality=4)
        assert "easiness" in result
        assert "interval" in result
        assert "repetitions" in result
        assert "next_review" in result

    def test_all_quality_values(self):
        """Should not crash for any valid quality (0-5)."""
        for q in range(6):
            result = sm2_update(easiness=2.5, interval=1, repetitions=1, quality=q)
            assert result["easiness"] >= 1.3
            assert result["interval"] >= 1
            assert result["repetitions"] >= 0

    def test_ef_formula_exact_quality_4(self):
        """EF(new) = EF + (0.1 - (5-q) * (0.08 + (5-q)*0.02)) for q=4."""
        ef = 2.5
        q = 4
        expected_delta = 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)
        expected_ef = max(1.3, ef + expected_delta)
        result = sm2_update(easiness=ef, interval=1, repetitions=1, quality=q)
        assert abs(result["easiness"] - expected_ef) < 1e-10

    def test_ef_formula_exact_quality_0(self):
        """EF(new) = EF + (0.1 - 5*(0.08+5*0.02)) = EF - 0.8 for q=0."""
        ef = 2.5
        expected_ef = max(1.3, ef + (0.1 - 5 * (0.08 + 5 * 0.02)))
        result = sm2_update(easiness=ef, interval=1, repetitions=1, quality=0)
        assert abs(result["easiness"] - expected_ef) < 1e-10


# ━━ quality_from_quiz() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestQualityFromQuiz:
    """Test quiz result → SM-2 quality conversion."""

    def test_incorrect_returns_one(self):
        assert quality_from_quiz(is_correct=False) == 1

    def test_incorrect_ignores_time(self):
        assert quality_from_quiz(is_correct=False, time_seconds=5) == 1

    def test_correct_no_time_returns_four(self):
        assert quality_from_quiz(is_correct=True) == 4

    def test_correct_fast_returns_five(self):
        assert quality_from_quiz(is_correct=True, time_seconds=5) == 5

    def test_correct_moderate_returns_four(self):
        assert quality_from_quiz(is_correct=True, time_seconds=20) == 4

    def test_correct_slow_returns_three(self):
        assert quality_from_quiz(is_correct=True, time_seconds=60) == 3

    def test_boundary_ten_seconds(self):
        assert quality_from_quiz(is_correct=True, time_seconds=10) == 4  # >= 10

    def test_boundary_thirty_seconds(self):
        assert quality_from_quiz(is_correct=True, time_seconds=30) == 3  # >= 30

    def test_boundary_just_under_ten(self):
        assert quality_from_quiz(is_correct=True, time_seconds=9.9) == 5

    def test_boundary_just_under_thirty(self):
        assert quality_from_quiz(is_correct=True, time_seconds=29.9) == 4

    def test_quality_range(self):
        """Quality should always be in [0, 5]."""
        for correct in [True, False]:
            for t in [None, 1, 10, 30, 100]:
                q = quality_from_quiz(correct, t)
                assert 0 <= q <= 5
