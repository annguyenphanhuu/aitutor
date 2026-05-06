"""
Tests for Orchestrator formatting helpers.

Module: app/agents/orchestrator.py
Covers:
- Orchestrator._format_plan() — study plan formatting
- Orchestrator._format_assessment() — assessment result formatting
- Orchestrator._extract_and_visualize() — math expression extraction
"""

import pytest
from unittest.mock import MagicMock, patch
from app.agents.orchestrator import Orchestrator


@pytest.fixture
def orchestrator():
    """Create an orchestrator with mocked LLM dependencies."""
    with patch("app.agents.orchestrator.ChatOpenAI"), \
         patch("app.agents.orchestrator.AsyncOpenAI"), \
         patch("app.agents.orchestrator.TeacherAgent"), \
         patch("app.agents.orchestrator.AssessorAgent"), \
         patch("app.agents.orchestrator.PlannerAgent"), \
         patch("app.agents.orchestrator.VisualizerAgent") as mock_viz:
        orch = Orchestrator()
        yield orch


# ━━ _format_plan() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFormatPlan:

    def test_basic_plan(self, orchestrator):
        plan = {
            "summary": "Kế hoạch ôn tập tuần này",
            "priorities": [
                {
                    "skill_name": "Đạo hàm cơ bản",
                    "action": "Ôn lại công thức",
                    "exercises": 10,
                    "urgency": "high",
                },
                {
                    "skill_name": "Tích phân",
                    "action": "Làm bài tập",
                    "exercises": 5,
                    "urgency": "medium",
                },
            ],
            "encouragement": "Em đang tiến bộ rất tốt!",
        }
        result = orchestrator._format_plan(plan)
        assert "Kế hoạch ôn tập tuần này" in result
        assert "Đạo hàm cơ bản" in result
        assert "Tích phân" in result
        assert "🔴" in result  # high urgency
        assert "🟡" in result  # medium urgency
        assert "Em đang tiến bộ rất tốt!" in result

    def test_empty_plan(self, orchestrator):
        plan = {
            "summary": "Nothing",
            "priorities": [],
            "encouragement": "",
        }
        result = orchestrator._format_plan(plan)
        assert "Nothing" in result

    def test_urgency_emojis(self, orchestrator):
        plan = {
            "summary": "Test",
            "priorities": [
                {"skill_name": "A", "action": "Do", "exercises": 1, "urgency": "high"},
                {"skill_name": "B", "action": "Do", "exercises": 1, "urgency": "medium"},
                {"skill_name": "C", "action": "Do", "exercises": 1, "urgency": "low"},
            ],
        }
        result = orchestrator._format_plan(plan)
        assert "🔴" in result
        assert "🟡" in result
        assert "🟢" in result

    def test_missing_encouragement(self, orchestrator):
        plan = {"summary": "Plan", "priorities": []}
        result = orchestrator._format_plan(plan)
        assert isinstance(result, str)


# ━━ _format_assessment() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFormatAssessment:

    def test_correct_assessment(self, orchestrator):
        assessment = {
            "is_correct": True,
            "score": 1.0,
            "feedback": "Hoàn toàn chính xác!",
            "correct_solution": "x = 5",
        }
        result = orchestrator._format_assessment(assessment)
        assert "✅" in result
        assert "100%" in result
        assert "Hoàn toàn chính xác!" in result
        assert "x = 5" in result

    def test_incorrect_assessment(self, orchestrator):
        assessment = {
            "is_correct": False,
            "score": 0.0,
            "error_type": "calculation",
            "feedback": "Sai phép tính",
            "correct_solution": "x = 3",
        }
        result = orchestrator._format_assessment(assessment)
        assert "❌" in result
        assert "Lỗi tính toán" in result
        assert "x = 3" in result

    def test_conceptual_error(self, orchestrator):
        assessment = {
            "is_correct": False,
            "score": 0.3,
            "error_type": "conceptual",
            "feedback": "Hiểu sai khái niệm",
            "correct_solution": "Giải lại...",
        }
        result = orchestrator._format_assessment(assessment)
        assert "Hiểu sai khái niệm" in result

    def test_procedural_error(self, orchestrator):
        assessment = {
            "is_correct": False,
            "score": 0.5,
            "error_type": "procedural",
            "feedback": "Sai phương pháp",
            "correct_solution": "Dùng phương pháp khác",
        }
        result = orchestrator._format_assessment(assessment)
        assert "Sai phương pháp" in result

    def test_partial_score(self, orchestrator):
        assessment = {
            "is_correct": False,
            "score": 0.5,
            "error_type": "calculation",
            "feedback": "Gần đúng",
            "correct_solution": "...",
        }
        result = orchestrator._format_assessment(assessment)
        assert "50%" in result


# ━━ _extract_and_visualize() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestExtractAndVisualize:

    @pytest.mark.asyncio
    async def test_y_equals_expression(self, orchestrator):
        """Should extract 'x^2 - 3x + 1' from 'y = x^2 - 3x + 1'."""
        orchestrator.visualizer = MagicMock()
        orchestrator.visualizer.generate_function_plot.return_value = {"vis_type": "plot"}

        with patch.object(orchestrator, "_extract_plottable_expr_llm", return_value="x**2 - 3*x + 1"):
            result = await orchestrator._extract_and_visualize("y = x^2 - 3x + 1")
            orchestrator.visualizer.generate_function_plot.assert_called_with("x**2 - 3*x + 1")

    @pytest.mark.asyncio
    async def test_f_x_equals_expression(self, orchestrator):
        orchestrator.visualizer = MagicMock()
        orchestrator.visualizer.generate_function_plot.return_value = {"vis_type": "plot"}

        with patch.object(orchestrator, "_extract_plottable_expr_llm", return_value="sin(x)"):
            result = await orchestrator._extract_and_visualize("f(x) = sin(x)")
            orchestrator.visualizer.generate_function_plot.assert_called_with("sin(x)")

    @pytest.mark.asyncio
    async def test_do_thi_prefix(self, orchestrator):
        orchestrator.visualizer = MagicMock()
        orchestrator.visualizer.generate_function_plot.return_value = {"vis_type": "plot"}

        with patch.object(orchestrator, "_extract_plottable_expr_llm", return_value="x**3 - 3*x"):
            result = await orchestrator._extract_and_visualize("đồ thị x^3 - 3x")
            orchestrator.visualizer.generate_function_plot.assert_called_with("x**3 - 3*x")

    @pytest.mark.asyncio
    async def test_ve_prefix(self, orchestrator):
        orchestrator.visualizer = MagicMock()
        orchestrator.visualizer.generate_function_plot.return_value = {"vis_type": "plot"}

        with patch.object(orchestrator, "_extract_plottable_expr_llm", return_value="x**2"):
            result = await orchestrator._extract_and_visualize("vẽ đồ thị hàm số x^2")
            orchestrator.visualizer.generate_function_plot.assert_called_with("x**2")

    @pytest.mark.asyncio
    async def test_fallback_full_message(self, orchestrator):
        orchestrator.visualizer = MagicMock()
        orchestrator.visualizer.generate_function_plot.return_value = {"vis_type": "error"}

        with patch.object(orchestrator, "_extract_plottable_expr_llm", return_value="something random"):
            result = await orchestrator._extract_and_visualize("something random")
            orchestrator.visualizer.generate_function_plot.assert_called_with("something random")
