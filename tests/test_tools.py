"""
Tests for SymPy-based Math Tools.

Module: app/agents/tools.py
Covers:
- compute_derivative() — exact symbolic differentiation
- compute_integral() — definite & indefinite
- solve_equation() — algebraic equation solving
- simplify_expression() — simplification
- evaluate_at_point() — numeric evaluation
- _safe_parse() — expression parsing
- MATH_TOOLS registry
"""

from app.agents.tools import (
    compute_derivative,
    compute_integral,
    solve_equation,
    simplify_expression,
    evaluate_at_point,
    MATH_TOOLS,
    TOOL_DESCRIPTIONS,
)


# ━━ compute_derivative() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestComputeDerivative:
    """Test symbolic differentiation."""

    def test_polynomial(self):
        result = compute_derivative("x**3 + 2*x")
        assert result["success"] is True
        assert "3" in result["result_sympy"]

    def test_constant(self):
        result = compute_derivative("5")
        assert result["success"] is True
        assert result["result_sympy"] == "0"

    def test_sin(self):
        result = compute_derivative("sin(x)")
        assert result["success"] is True
        assert "cos" in result["result_sympy"]

    def test_exp(self):
        result = compute_derivative("exp(x)")
        assert result["success"] is True
        assert "exp" in result["result_sympy"]

    def test_second_derivative(self):
        result = compute_derivative("x**3", order=2)
        assert result["success"] is True
        # d²/dx²(x³) = 6x
        assert "6" in result["result_sympy"]

    def test_different_variable(self):
        result = compute_derivative("y**2", var="y")
        assert result["success"] is True
        assert "2" in result["result_sympy"]

    def test_latex_output(self):
        result = compute_derivative("x**2")
        assert result["success"] is True
        assert "$" in result["result_latex"]

    def test_input_preserved(self):
        result = compute_derivative("x**2 + 1")
        assert result["input"] == "x**2 + 1"

    def test_invalid_expression(self):
        result = compute_derivative("@#$%^&*()")
        assert result["success"] is False
        assert "error" in result

    def test_caret_notation(self):
        """User may type ^ instead of **."""
        result = compute_derivative("x^3")
        assert result["success"] is True

    def test_product_rule(self):
        result = compute_derivative("x*sin(x)")
        assert result["success"] is True

    def test_quotient(self):
        result = compute_derivative("1/x")
        assert result["success"] is True


# ━━ compute_integral() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestComputeIntegral:
    """Test symbolic integration."""

    def test_indefinite_polynomial(self):
        result = compute_integral("x**2")
        assert result["success"] is True
        # ∫x² dx = x³/3
        assert "3" in result["result_sympy"]

    def test_definite_integral(self):
        result = compute_integral("x**2", lower="0", upper="1")
        assert result["success"] is True
        # ∫₀¹ x² dx = 1/3
        assert "1/3" in result["result_sympy"] or "Rational" in result["result_sympy"]

    def test_definite_trig(self):
        result = compute_integral("sin(x)", lower="0", upper="pi")
        assert result["success"] is True
        # ∫₀ᵖⁱ sin(x) dx = 2
        assert "2" in result["result_sympy"]

    def test_constant_integral(self):
        result = compute_integral("3")
        assert result["success"] is True
        # ∫3 dx = 3x
        assert "x" in result["result_sympy"]

    def test_exp_integral(self):
        result = compute_integral("exp(x)")
        assert result["success"] is True
        assert "exp" in result["result_sympy"]

    def test_invalid_expression(self):
        result = compute_integral("@#$%^&*()")
        assert result["success"] is False


# ━━ solve_equation() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSolveEquation:
    """Test equation solving."""

    def test_linear(self):
        result = solve_equation("2*x - 4 = 0")
        assert result["success"] is True
        assert result["count"] == 1
        assert any("2" in s for s in result["solutions_sympy"])

    def test_quadratic(self):
        result = solve_equation("x**2 - 4 = 0")
        assert result["success"] is True
        assert result["count"] == 2

    def test_implicit_zero(self):
        """'x**2 - 1' without '= 0' should be treated as x²-1=0."""
        result = solve_equation("x**2 - 1")
        assert result["success"] is True
        assert result["count"] == 2

    def test_no_solution(self):
        result = solve_equation("x**2 + 1 = 0")
        assert result["success"] is True
        # Complex solutions exist; count may be 2 or 0 depending on domain
        assert isinstance(result["count"], int)

    def test_solutions_latex(self):
        result = solve_equation("x - 5 = 0")
        assert result["success"] is True
        assert any("$" in s for s in result["solutions"])

    def test_caret_notation(self):
        result = solve_equation("x^2 - 9 = 0")
        assert result["success"] is True
        assert result["count"] == 2

    def test_invalid_equation(self):
        result = solve_equation("= = =")
        assert result["success"] is False


# ━━ simplify_expression() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSimplifyExpression:
    def test_basic_simplification(self):
        result = simplify_expression("x**2 + 2*x + 1")
        assert result["success"] is True

    def test_trig_identity(self):
        result = simplify_expression("sin(x)**2 + cos(x)**2")
        assert result["success"] is True
        assert "1" in result["result_sympy"]

    def test_fraction_simplification(self):
        result = simplify_expression("(x**2 - 1)/(x - 1)")
        assert result["success"] is True
        # Should simplify to x + 1
        assert "x + 1" in result["result_sympy"] or "x+1" in result["result_sympy"]

    def test_invalid_expression(self):
        result = simplify_expression("???")
        assert result["success"] is False


# ━━ evaluate_at_point() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEvaluateAtPoint:
    def test_polynomial_at_point(self):
        result = evaluate_at_point("x**2 + 1", var="x", value="3")
        assert result["success"] is True
        assert result["result_sympy"] == "10"

    def test_trig_at_zero(self):
        result = evaluate_at_point("sin(x)", var="x", value="0")
        assert result["success"] is True
        assert result["result_sympy"] == "0"

    def test_result_float(self):
        result = evaluate_at_point("x + 1", var="x", value="2")
        assert result["success"] is True
        assert result.get("result_float") == "3.0"

    def test_invalid_expression(self):
        result = evaluate_at_point("???invalid", var="x", value="0")
        assert result["success"] is False


# ━━ Tool Registry ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestToolRegistry:
    def test_all_tools_registered(self):
        expected = ["compute_derivative", "compute_integral", "solve_equation",
                     "simplify_expression", "evaluate_at_point"]
        for t in expected:
            assert t in MATH_TOOLS

    def test_all_tools_callable(self):
        for name, fn in MATH_TOOLS.items():
            assert callable(fn), f"Tool '{name}' is not callable"

    def test_descriptions_exist(self):
        for name in MATH_TOOLS:
            assert name in TOOL_DESCRIPTIONS
            assert len(TOOL_DESCRIPTIONS[name]) > 0
