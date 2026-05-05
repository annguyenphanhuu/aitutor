"""
Tests for Visualizer Agent.

Module: app/agents/visualizer_agent.py
Covers:
- generate_function_plot() — Plotly trace generation
- generate_desmos_url() — Desmos URL generation
- generate_derivative_visualization() — function + derivative plot
- _normalize_expr() — expression normalization
- Error handling for invalid expressions
"""

import pytest
from app.agents.visualizer_agent import VisualizerAgent


@pytest.fixture
def viz():
    return VisualizerAgent()


# ━━ _normalize_expr() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestNormalizeExpr:

    def test_strip_y_prefix(self, viz):
        assert "=" not in viz._normalize_expr("y = x**2")

    def test_strip_f_prefix(self, viz):
        assert "=" not in viz._normalize_expr("f(x) = x**2")

    def test_caret_to_doublestar(self, viz):
        result = viz._normalize_expr("x^3")
        assert "**" in result

    def test_sqrt_symbol(self, viz):
        result = viz._normalize_expr("√x")
        assert "sqrt" in result

    def test_ln_to_log(self, viz):
        result = viz._normalize_expr("ln(x)")
        assert "log" in result

    def test_whitespace_stripped(self, viz):
        result = viz._normalize_expr("  x^2  ")
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_plain_expression(self, viz):
        result = viz._normalize_expr("sin(x)")
        assert result == "sin(x)"


# ━━ generate_function_plot() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGenerateFunctionPlot:

    def test_polynomial(self, viz):
        result = viz.generate_function_plot("x^3 - 3*x + 1")
        assert result["vis_type"] == "plot"
        assert "traces" in result["data"]
        assert len(result["data"]["traces"]) == 1

    def test_trace_has_xy_data(self, viz):
        result = viz.generate_function_plot("x**2")
        trace = result["data"]["traces"][0]
        assert "x" in trace
        assert "y" in trace
        assert len(trace["x"]) == 500
        assert len(trace["y"]) == 500

    def test_trig_function(self, viz):
        result = viz.generate_function_plot("sin(x)")
        assert result["vis_type"] == "plot"

    def test_exponential(self, viz):
        result = viz.generate_function_plot("exp(x)")
        assert result["vis_type"] == "plot"

    def test_layout_present(self, viz):
        result = viz.generate_function_plot("x**2")
        layout = result["data"]["layout"]
        assert "title" in layout
        assert "xaxis" in layout
        assert "yaxis" in layout

    def test_latex_caption(self, viz):
        result = viz.generate_function_plot("x**2")
        assert result["latex_caption"] is not None
        assert len(result["latex_caption"]) > 0

    def test_y_prefix_stripped(self, viz):
        result = viz.generate_function_plot("y = x**2")
        assert result["vis_type"] == "plot"

    def test_f_prefix_stripped(self, viz):
        result = viz.generate_function_plot("f(x) = sin(x)")
        assert result["vis_type"] == "plot"

    def test_invalid_expression_returns_error(self, viz):
        result = viz.generate_function_plot("not a math expression at all")
        assert result["vis_type"] == "error"
        assert "message" in result["data"]

    def test_trace_style(self, viz):
        result = viz.generate_function_plot("x**2")
        trace = result["data"]["traces"][0]
        assert trace["type"] == "scatter"
        assert trace["mode"] == "lines"
        assert "color" in trace["line"]


# ━━ generate_desmos_url() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGenerateDesmosURL:

    def test_basic(self, viz):
        result = viz.generate_desmos_url("x^2")
        assert result["vis_type"] == "desmos"
        assert "url" in result["data"]
        assert "desmos.com" in result["data"]["url"]

    def test_embed_html(self, viz):
        result = viz.generate_desmos_url("sin(x)")
        assert "embed_html" in result["data"]
        assert "<iframe" in result["data"]["embed_html"]

    def test_caption(self, viz):
        result = viz.generate_desmos_url("x^3")
        assert result["latex_caption"] is not None


# ━━ generate_derivative_visualization() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDerivativeVisualization:

    def test_polynomial(self, viz):
        result = viz.generate_derivative_visualization("x^3 - 3*x + 1")
        assert result["vis_type"] == "plot"
        assert len(result["data"]["traces"]) == 2  # f(x) and f'(x)

    def test_trace_names(self, viz):
        result = viz.generate_derivative_visualization("x**2")
        traces = result["data"]["traces"]
        assert "f(x)" in traces[0]["name"]
        assert "f'(x)" in traces[1]["name"]

    def test_different_trace_colours(self, viz):
        result = viz.generate_derivative_visualization("x**2")
        traces = result["data"]["traces"]
        assert traces[0]["line"]["color"] != traces[1]["line"]["color"]

    def test_derivative_dashed(self, viz):
        result = viz.generate_derivative_visualization("x**2")
        assert result["data"]["traces"][1]["line"].get("dash") == "dash"

    def test_caption_has_both(self, viz):
        result = viz.generate_derivative_visualization("x**2")
        assert "f(x)" in result["latex_caption"]
        assert "f'(x)" in result["latex_caption"]

    def test_invalid_returns_error(self, viz):
        result = viz.generate_derivative_visualization("totally invalid @@@@")
        assert result["vis_type"] == "error"


# ━━ _error() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestErrorHelper:

    def test_error_structure(self, viz):
        result = viz._error("Something went wrong")
        assert result["vis_type"] == "error"
        assert result["data"]["message"] == "Something went wrong"
        assert result["latex_caption"] is None
