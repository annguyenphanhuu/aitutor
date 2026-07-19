"""Tests cho app/agents/grading/verifier.py — pure SymPy, không mock."""

from app.agents.contracts import GradingExtraction
from app.agents.grading.verifier import (
    verify,
    verify_antiderivative,
    verify_derivative,
    verify_equation_solutions,
    verify_numeric,
)


class TestVerifyAntiderivative:
    def test_correct_antiderivative(self):
        result = verify_antiderivative("x**3/3", "x**2")
        assert result.verified is True
        assert result.verdict == "correct"

    def test_correct_with_constant(self):
        result = verify_antiderivative("x**3/3 + C", "x**2")
        assert result.verified is True
        assert result.verdict == "correct"

    def test_bug_case_x_cubed_over_two_is_incorrect(self):
        # Bug gốc: thầy nói ∫x²dx = x³/2 + C — phải bị CAS bác bỏ
        result = verify_antiderivative("x**3/2 + C", "x**2")
        assert result.verified is True
        assert result.verdict == "incorrect"

    def test_trig_antiderivative(self):
        result = verify_antiderivative("-cos(x)", "sin(x)")
        assert result.verdict == "correct"


class TestVerifyDerivative:
    def test_correct_derivative(self):
        result = verify_derivative("2*x", "x**2")
        assert result.verified is True
        assert result.verdict == "correct"

    def test_wrong_sign_derivative(self):
        # Lỗi đảo dấu kinh điển: (sin x)' = -cos x là SAI
        result = verify_derivative("-cos(x)", "sin(x)")
        assert result.verified is True
        assert result.verdict == "incorrect"


class TestVerifyEquation:
    def test_full_solution_set(self):
        result = verify_equation_solutions("2; -2", "x**2 - 4 = 0")
        assert result.verified is True
        assert result.verdict == "correct"

    def test_missing_solution(self):
        result = verify_equation_solutions("2", "x**2 - 4 = 0")
        assert result.verdict == "incorrect"

    def test_extra_solution(self):
        result = verify_equation_solutions("2; -2; 3", "x**2 - 4 = 0")
        assert result.verdict == "incorrect"


class TestVerifyNumeric:
    def test_equal_values(self):
        result = verify_numeric("1/2", "0.5")
        assert result.verdict == "correct"

    def test_unequal_values(self):
        result = verify_numeric("0.33", "1/3")
        assert result.verdict == "incorrect"


class TestVerifyDispatch:
    def test_not_gradable_returns_unverified(self):
        extraction = GradingExtraction(gradable=False)
        result = verify(extraction)
        assert result.verified is False
        assert result.verdict == "unknown"

    def test_missing_exprs_returns_unverified(self):
        extraction = GradingExtraction(gradable=True, task_kind="antiderivative")
        result = verify(extraction)
        assert result.verified is False

    def test_garbage_input_never_raises(self):
        extraction = GradingExtraction(
            gradable=True,
            task_kind="antiderivative",
            problem_expr="!!@@ not math ##",
            candidate_expr=")((",
        )
        result = verify(extraction)
        assert result.verified is False
        assert result.verdict == "unknown"

    def test_unsupported_kind(self):
        extraction = GradingExtraction(
            gradable=True, task_kind="other",
            problem_expr="x", candidate_expr="x",
        )
        result = verify(extraction)
        assert result.verified is False
        assert result.method == "unsupported_kind"

    def test_bug_case_end_to_end(self):
        extraction = GradingExtraction(
            gradable=True,
            task_kind="antiderivative",
            problem_expr="x**2",
            candidate_expr="x**3/2",
        )
        result = verify(extraction)
        assert result.verified is True
        assert result.verdict == "incorrect"
