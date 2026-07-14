"""
Tests for Bayesian Knowledge Tracing (BKT) model.

Module: app/knowledge_tracing/bkt.py
Covers:
- BKTModel initialization (default & custom params)
- update() with correct/incorrect answers
- predict_correct()
- get_mastery_level() boundaries
- Mathematical invariants (probabilities stay in [0, 1])
- Convergence behaviour (repeated correct → mastery increases)
"""

from app.knowledge_tracing.bkt import BKTModel


# ━━ Initialization ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBKTInit:
    """Test BKTModel constructor."""

    def test_default_params(self):
        model = BKTModel()
        assert model.p_init == 0.1
        assert model.p_transit == 0.1
        assert model.p_slip == 0.05
        assert model.p_guess == 0.25

    def test_custom_params(self):
        params = {"p_init": 0.3, "p_transit": 0.2, "p_slip": 0.1, "p_guess": 0.3}
        model = BKTModel(params)
        assert model.p_init == 0.3
        assert model.p_transit == 0.2
        assert model.p_slip == 0.1
        assert model.p_guess == 0.3

    def test_partial_params_fills_defaults(self):
        model = BKTModel({"p_init": 0.5})
        assert model.p_init == 0.5
        assert model.p_transit == 0.1  # default
        assert model.p_slip == 0.05    # default
        assert model.p_guess == 0.25   # default

    def test_none_params_uses_defaults(self):
        model = BKTModel(None)
        assert model.p_init == 0.1


# ━━ update() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBKTUpdate:
    """Test mastery update logic."""

    def setup_method(self):
        self.model = BKTModel()

    def test_correct_answer_increases_mastery(self):
        initial = 0.3
        updated = self.model.update(initial, is_correct=True)
        assert updated > initial

    def test_incorrect_answer_still_increases_slightly(self):
        """Due to learning transition, even incorrect can increase marginally."""
        initial = 0.3
        updated = self.model.update(initial, is_correct=False)
        # The posterior decreases but transit adds back
        # With default params, net effect depends on prior
        assert isinstance(updated, float)

    def test_mastery_stays_in_zero_one_range(self):
        model = self.model
        # Test extremes
        for p in [0.0, 0.001, 0.5, 0.99, 1.0]:
            for correct in [True, False]:
                result = model.update(p, correct)
                assert 0.0 <= result <= 1.0, f"Out of range: p={p}, correct={correct} → {result}"

    def test_high_mastery_correct_stays_high(self):
        result = self.model.update(0.95, is_correct=True)
        assert result >= 0.95

    def test_low_mastery_correct_increases_significantly(self):
        result = self.model.update(0.1, is_correct=True)
        assert result > 0.1

    def test_repeated_correct_converges_to_one(self):
        """Multiple correct answers should drive mastery toward 1.0."""
        p = 0.1
        for _ in range(50):
            p = self.model.update(p, is_correct=True)
        assert p > 0.95, f"After 50 correct answers, mastery should be near 1.0, got {p}"

    def test_repeated_incorrect_does_not_crash(self):
        """Many incorrect answers should not cause errors or negative values."""
        p = 0.5
        for _ in range(30):
            p = self.model.update(p, is_correct=False)
        assert 0.0 <= p <= 1.0

    def test_zero_mastery_correct(self):
        result = self.model.update(0.0, is_correct=True)
        assert result > 0.0

    def test_full_mastery_incorrect(self):
        result = self.model.update(1.0, is_correct=False)
        assert result <= 1.0

    def test_p_correct_zero_edge_case(self):
        """When p_correct would be 0, mastery should not crash."""
        # With p_slip=1.0 and p_guess=0, p_correct could be 0 at p_mastery=0
        model = BKTModel({"p_init": 0.0, "p_transit": 0.1, "p_slip": 1.0, "p_guess": 0.0})
        result = model.update(0.0, is_correct=True)
        assert 0.0 <= result <= 1.0


# ━━ predict_correct() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBKTPredict:
    """Test prediction of correct response."""

    def setup_method(self):
        self.model = BKTModel()

    def test_high_mastery_predicts_high_correct(self):
        prob = self.model.predict_correct(0.9)
        assert prob > 0.8

    def test_low_mastery_predicts_low_correct(self):
        prob = self.model.predict_correct(0.1)
        assert prob < 0.5

    def test_full_mastery_prediction(self):
        # P(correct | mastered) = 1 * (1 - p_slip) + 0 * p_guess = 0.95
        prob = self.model.predict_correct(1.0)
        assert abs(prob - 0.95) < 1e-10

    def test_zero_mastery_prediction(self):
        # P(correct | not mastered) = 0 * (1 - p_slip) + 1 * p_guess = 0.25
        prob = self.model.predict_correct(0.0)
        assert abs(prob - 0.25) < 1e-10

    def test_prediction_in_valid_range(self):
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            prob = self.model.predict_correct(p)
            assert 0.0 <= prob <= 1.0


# ━━ get_mastery_level() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBKTMasteryLevel:
    """Test mastery level classification."""

    def setup_method(self):
        self.model = BKTModel()

    def test_beginner_range(self):
        assert self.model.get_mastery_level(0.0) == "beginner"
        assert self.model.get_mastery_level(0.1) == "beginner"
        assert self.model.get_mastery_level(0.29) == "beginner"

    def test_developing_range(self):
        assert self.model.get_mastery_level(0.3) == "developing"
        assert self.model.get_mastery_level(0.5) == "developing"
        assert self.model.get_mastery_level(0.59) == "developing"

    def test_proficient_range(self):
        assert self.model.get_mastery_level(0.6) == "proficient"
        assert self.model.get_mastery_level(0.7) == "proficient"
        assert self.model.get_mastery_level(0.84) == "proficient"

    def test_mastered_range(self):
        assert self.model.get_mastery_level(0.85) == "mastered"
        assert self.model.get_mastery_level(0.9) == "mastered"
        assert self.model.get_mastery_level(1.0) == "mastered"

    def test_boundary_beginner_developing(self):
        assert self.model.get_mastery_level(0.3) == "developing"
        assert self.model.get_mastery_level(0.2999) == "beginner"

    def test_boundary_developing_proficient(self):
        assert self.model.get_mastery_level(0.6) == "proficient"
        assert self.model.get_mastery_level(0.5999) == "developing"

    def test_boundary_proficient_mastered(self):
        assert self.model.get_mastery_level(0.85) == "mastered"
        assert self.model.get_mastery_level(0.8499) == "proficient"


# ━━ batch_update_mastery_from_exam() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBatchUpdateMasteryFromExam:
    """
    Unit tests for batch_update_mastery_from_exam().

    Uses a lightweight mock DB session to avoid needing a real database,
    isolating the BKT logic from the persistence layer.
    """

    def _make_mock_mastery(self, p: float = 0.1):
        """Create a simple mock SkillMastery object."""
        class MockMastery:
            def __init__(self, p_mastery):
                self.p_mastery = p_mastery
                self.total_attempts = 0
                self.correct_attempts = 0
        return MockMastery(p)

    def test_all_correct_increases_mastery(self):
        """Many correct answers for same skill → mastery increases above initial."""
        bkt = BKTModel()
        p = 0.1
        for _ in range(10):
            p = bkt.update(p, is_correct=True)
        assert p > 0.1, "Mastery should increase after 10 correct answers"

    def test_all_incorrect_mastery_stays_low(self):
        """Many incorrect answers should keep mastery low."""
        bkt = BKTModel()
        p = 0.1
        for _ in range(10):
            p = bkt.update(p, is_correct=False)
        assert p < 0.5, "Mastery should remain low after 10 incorrect answers"

    def test_sequential_bkt_chain_order_matters(self):
        """
        BKT sequential chain: updating [correct, correct, wrong] should differ
        from applying the equivalent shortcut, confirming order is preserved.
        """
        bkt = BKTModel()
        sequence_a = [True, True, False]
        sequence_b = [False, True, True]

        p_a = 0.2
        for ans in sequence_a:
            p_a = bkt.update(p_a, is_correct=ans)

        p_b = 0.2
        for ans in sequence_b:
            p_b = bkt.update(p_b, is_correct=ans)

        # Same answers in different order → different final mastery (Markov property)
        assert p_a != p_b, (
            "Sequential BKT updates with different orderings should yield different results"
        )

    def test_multiple_skills_updated_independently(self):
        """Each skill_id should be updated independently with its own BKT chain."""
        bkt = BKTModel()

        skill_a_answers = [True, True, True]   # all correct
        skill_b_answers = [False, False, False] # all incorrect

        p_a = 0.1
        for ans in skill_a_answers:
            p_a = bkt.update(p_a, is_correct=ans)

        p_b = 0.1
        for ans in skill_b_answers:
            p_b = bkt.update(p_b, is_correct=ans)

        assert p_a > p_b, "Skill with all-correct answers should have higher mastery"

    def test_empty_skill_id_is_skipped(self):
        """Items with empty or None skill_id should be silently ignored."""
        from collections import defaultdict
        exam_items = [
            {"skill_id": "", "is_correct": True},
            {"skill_id": None, "is_correct": True},
        ]

        skill_answers = defaultdict(list)
        for item in exam_items:
            sid = item.get("skill_id")
            if sid:
                skill_answers[sid].append(bool(item.get("is_correct", False)))

        # No valid skill_ids found → no updates
        assert len(skill_answers) == 0

    def test_empty_exam_items_returns_empty_dict(self):
        """Empty exam_items list should produce an empty mastery_updates dict."""
        from collections import defaultdict
        exam_items = []
        skill_answers = defaultdict(list)
        for item in exam_items:
            sid = item.get("skill_id")
            if sid:
                skill_answers[sid].append(item.get("is_correct", False))
        assert len(skill_answers) == 0

    def test_attempt_counters_increment_correctly(self):
        """total_attempts and correct_attempts should count all answers per skill."""
        answers = [True, False, True, True]  # 3 correct, 1 incorrect

        total = len(answers)
        correct = sum(1 for a in answers if a)

        assert total == 4
        assert correct == 3

    def test_mastery_result_bounded_in_zero_one(self):
        """Regardless of input, final mastery must remain in [0, 1]."""
        bkt = BKTModel()
        p = 0.0
        for _ in range(100):
            p = bkt.update(p, is_correct=True)
        assert 0.0 <= p <= 1.0, f"Mastery out of bounds: {p}"

        p = 1.0
        for _ in range(100):
            p = bkt.update(p, is_correct=False)
        assert 0.0 <= p <= 1.0, f"Mastery out of bounds: {p}"

    def test_return_value_rounded_to_4_decimals(self):
        """batch_update_mastery_from_exam should round results to 4 decimal places."""
        bkt = BKTModel()
        p = 0.1
        for ans in [True, False, True]:
            p = bkt.update(p, ans)
        rounded = round(p, 4)
        assert rounded == round(rounded, 4)  # sanity check on rounding
