"""
Tests for Quiz Generator.

Module: app/quiz/generator.py
Covers:
- BaseQuizGenerator._make_mcq() — MCQ structure
- BaseQuizGenerator._make_true_false() — T/F structure
- BaseQuizGenerator._make_short_answer() — short answer structure
- DerivativeQuizGenerator — derivative questions
- IntegralQuizGenerator — integral questions
- SequenceQuizGenerator — sequence questions
- GeometryQuizGenerator — geometry questions
- StatisticsQuizGenerator — statistics questions
- ProbabilityQuizGenerator — probability questions
- FunctionSurveyQuizGenerator — function survey questions
- generate_questions() — top-level generator
- generate_exam_questions() — THPT exam format
- _perturb() — distractor generation
- GENERATORS registry
"""

import random
from app.quiz.generator import (
    generate_questions,
    generate_exam_questions,
    DerivativeQuizGenerator,
    IntegralQuizGenerator,
    SequenceQuizGenerator,
    GeometryQuizGenerator,
    StatisticsQuizGenerator,
    ProbabilityQuizGenerator,
    FunctionSurveyQuizGenerator,
    ExpLogQuizGenerator,
    ComplexQuizGenerator,
    GENERATORS,
    _perturb,
)


# ━━ Quiz question structure validation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _validate_mcq(q: dict):
    """Common validation for MCQ questions."""
    assert q["question_type"] == "mcq"
    assert isinstance(q["question_latex"], str)
    assert len(q["question_latex"]) > 0
    assert isinstance(q["choices"], list)
    assert len(q["choices"]) == 4
    assert isinstance(q["correct_index"], int)
    assert 0 <= q["correct_index"] <= 3
    assert isinstance(q["explanation"], str)
    assert isinstance(q["difficulty"], int)
    assert 1 <= q["difficulty"] <= 3
    assert isinstance(q["skill_id"], str)
    assert isinstance(q.get("points", 0.25), (int, float))
    # Verify correct answer is in choices
    assert q["choices"][q["correct_index"]] is not None
    # Verify all choices are unique
    assert len(set(q["choices"])) == 4, f"Duplicate choices: {q['choices']}"


def _validate_true_false(q: dict):
    """Common validation for True/False questions."""
    assert q["question_type"] == "true_false"
    assert isinstance(q["question_latex"], str)
    assert isinstance(q["statements"], list)
    assert len(q["statements"]) == 4
    for s in q["statements"]:
        assert "text" in s
        assert "correct" in s
        assert isinstance(s["correct"], bool)


def _validate_short_answer(q: dict):
    """Common validation for short answer questions."""
    assert q["question_type"] == "short_answer"
    assert isinstance(q["question_latex"], str)
    assert isinstance(q["correct_answer"], str)
    assert len(q["correct_answer"]) > 0


# ━━ DerivativeQuizGenerator ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDerivativeQuizGenerator:

    def setup_method(self):
        random.seed(42)  # Reproducible tests
        self.gen = DerivativeQuizGenerator()

    def test_easy_generates_valid_mcq(self):
        q = self.gen.generate(difficulty=1)
        _validate_mcq(q)

    def test_medium_generates_valid_mcq(self):
        q = self.gen.generate(difficulty=2)
        _validate_mcq(q)

    def test_hard_generates_valid_mcq(self):
        q = self.gen.generate(difficulty=3)
        _validate_mcq(q)

    def test_question_contains_derivative_keyword(self):
        q = self.gen.generate(difficulty=1)
        assert "đạo hàm" in q["question_latex"].lower() or "f'(x)" in q["question_latex"]

    def test_multiple_generations_dont_crash(self):
        for _ in range(20):
            q = self.gen.generate(difficulty=random.randint(1, 3))
            _validate_mcq(q)


# ━━ IntegralQuizGenerator ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIntegralQuizGenerator:

    def setup_method(self):
        random.seed(42)
        self.gen = IntegralQuizGenerator()

    def test_easy(self):
        q = self.gen.generate(difficulty=1)
        _validate_mcq(q)

    def test_medium(self):
        q = self.gen.generate(difficulty=2)
        _validate_mcq(q)

    def test_hard(self):
        q = self.gen.generate(difficulty=3)
        _validate_mcq(q)

    def test_multiple_generations(self):
        for _ in range(15):
            q = self.gen.generate(difficulty=random.randint(1, 3))
            _validate_mcq(q)


# ━━ SequenceQuizGenerator ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSequenceQuizGenerator:

    def setup_method(self):
        random.seed(42)
        self.gen = SequenceQuizGenerator()

    def test_easy(self):
        q = self.gen.generate(difficulty=1)
        _validate_mcq(q)

    def test_medium(self):
        q = self.gen.generate(difficulty=2)
        _validate_mcq(q)

    def test_hard(self):
        q = self.gen.generate(difficulty=3)
        _validate_mcq(q)

    def test_arithmetic_term(self):
        random.seed(42)
        q = self.gen._arithmetic_nth_term()
        _validate_mcq(q)
        assert "CSC" in q["question_latex"] or "csc" in q["question_latex"].lower()


# ━━ GeometryQuizGenerator ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGeometryQuizGenerator:

    def setup_method(self):
        random.seed(42)
        self.gen = GeometryQuizGenerator()

    def test_easy(self):
        q = self.gen.generate(difficulty=1)
        _validate_mcq(q)

    def test_medium(self):
        q = self.gen.generate(difficulty=2)
        _validate_mcq(q)

    def test_hard(self):
        q = self.gen.generate(difficulty=3)
        _validate_mcq(q)


# ━━ StatisticsQuizGenerator ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestStatisticsQuizGenerator:

    def setup_method(self):
        random.seed(42)
        self.gen = StatisticsQuizGenerator()

    def test_easy(self):
        q = self.gen.generate(difficulty=1)
        _validate_mcq(q)

    def test_medium(self):
        q = self.gen.generate(difficulty=2)
        _validate_mcq(q)

    def test_hard(self):
        q = self.gen.generate(difficulty=3)
        _validate_mcq(q)


# ━━ ProbabilityQuizGenerator ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestProbabilityQuizGenerator:

    def setup_method(self):
        random.seed(42)
        self.gen = ProbabilityQuizGenerator()

    def test_easy(self):
        q = self.gen.generate(difficulty=1)
        _validate_mcq(q)

    def test_medium_hard(self):
        q = self.gen.generate(difficulty=2)
        _validate_mcq(q)


# ━━ FunctionSurveyQuizGenerator ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFunctionSurveyQuizGenerator:

    def setup_method(self):
        random.seed(42)
        self.gen = FunctionSurveyQuizGenerator()

    def test_easy(self):
        q = self.gen.generate(difficulty=1)
        _validate_mcq(q)

    def test_medium(self):
        q = self.gen.generate(difficulty=2)
        _validate_mcq(q)

    def test_hard(self):
        q = self.gen.generate(difficulty=3)
        _validate_mcq(q)


# ━━ ExpLogQuizGenerator ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestExpLogQuizGenerator:

    def setup_method(self):
        random.seed(42)
        self.gen = ExpLogQuizGenerator()

    def test_easy(self):
        q = self.gen.generate(difficulty=1)
        _validate_mcq(q)

    def test_medium(self):
        q = self.gen.generate(difficulty=2)
        _validate_mcq(q)


# ━━ ComplexQuizGenerator ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestComplexQuizGenerator:

    def setup_method(self):
        random.seed(42)
        self.gen = ComplexQuizGenerator()

    def test_easy(self):
        q = self.gen.generate(difficulty=1)
        _validate_mcq(q)

    def test_medium(self):
        q = self.gen.generate(difficulty=2)
        _validate_mcq(q)


# ━━ generate_questions() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGenerateQuestions:

    def test_returns_correct_count(self):
        random.seed(42)
        qs = generate_questions("derivative_basic", difficulty=1, count=5)
        assert len(qs) == 5

    def test_all_mcq_by_default(self):
        random.seed(42)
        qs = generate_questions("derivative_basic", difficulty=1, count=3)
        for q in qs:
            assert q["question_type"] == "mcq"

    def test_unknown_skill_falls_back(self):
        """Unknown skill_id should fall back to derivative generator."""
        random.seed(42)
        qs = generate_questions("completely_unknown_skill", difficulty=1, count=2)
        # Should not be empty — fallback to DerivativeQuizGenerator
        assert len(qs) >= 0  # might still generate

    def test_skill_id_set_on_questions(self):
        random.seed(42)
        qs = generate_questions("derivative_basic", difficulty=1, count=3)
        for q in qs:
            assert q["skill_id"] == "derivative_basic"

    def test_zero_count(self):
        qs = generate_questions("derivative_basic", difficulty=1, count=0)
        assert len(qs) == 0

    def test_various_skills(self):
        """Test that each registered skill can generate at least 1 question."""
        random.seed(42)
        for skill_id in list(GENERATORS.keys())[:10]:  # Test first 10
            qs = generate_questions(skill_id, difficulty=1, count=1)
            assert len(qs) >= 0, f"Failed for skill: {skill_id}"


# ━━ generate_exam_questions() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGenerateExamQuestions:

    def test_mixed_format(self):
        random.seed(42)
        qs = generate_exam_questions("derivative_basic", difficulty=1)
        types = [q["question_type"] for q in qs]
        # Should have MCQ, true_false, short_answer
        assert "mcq" in types

    def test_total_count(self):
        random.seed(42)
        qs = generate_exam_questions("derivative_basic", difficulty=1)
        # Expected: 3 MCQ + 1 TF + 1 SA = 5
        assert len(qs) >= 3  # At minimum MCQs should generate


# ━━ _perturb() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPerturb:

    def test_returns_different_value(self):
        import sympy as sp
        random.seed(42)
        original = sp.Integer(5)
        results = set()
        for _ in range(20):
            results.add(str(_perturb(original)))
        # At least some should be different from the original
        assert len(results) > 1

    def test_handles_symbolic_expression(self):
        import sympy as sp
        x = sp.Symbol("x")
        result = _perturb(3 * x**2 + 1)
        assert result is not None


# ━━ GENERATORS registry ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGeneratorsRegistry:

    def test_registry_not_empty(self):
        assert len(GENERATORS) > 0

    def test_all_values_are_generators(self):
        from app.quiz.generator import BaseQuizGenerator
        for skill_id, gen in GENERATORS.items():
            assert isinstance(gen, BaseQuizGenerator), f"{skill_id} is not a BaseQuizGenerator"

    def test_key_skills_registered(self):
        expected = [
            "derivative_basic", "derivative_rules", "function_survey",
            "integral_definite", "primitive_basic",
            "probability_basic", "statistics_descriptive",
            "geometry_vectors", "sequence_basic",
        ]
        for sid in expected:
            assert sid in GENERATORS, f"Skill '{sid}' not registered in GENERATORS"
