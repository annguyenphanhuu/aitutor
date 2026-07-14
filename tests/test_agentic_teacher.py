"""Regression tests for mandatory SymPy tool routing."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.agentic_teacher import _format_tool_result, normalize_tool_result
from app.agents.teacher_agent import requires_math_tool
from app.agents.tool_schemas import TOOL_NAMES


def test_string_tool_result_is_not_misclassified_as_internal_error():
    normalized = normalize_tool_result("SymPy result: simplify(2+3) = 5")
    assert normalized["success"] is True
    assert "SymPy result" in _format_tool_result("simplify_expression", normalized)


def test_error_tool_result_is_normalized():
    normalized = normalize_tool_result("ERROR: invalid expression")
    assert normalized == {"success": False, "error": "ERROR: invalid expression"}


def test_solve_inequality_is_available_to_function_calling():
    assert "solve_inequality" in TOOL_NAMES


def test_langchain_tool_wrapper_preserves_structured_contract():
    from app.agents.tools import MATH_TOOL_LIST

    simplify_tool = next(tool for tool in MATH_TOOL_LIST if tool.name == "simplify_expression")
    result = simplify_tool.invoke({"expr_str": "2 + 3"})
    assert result["success"] is True
    assert result["result_sympy"] == "5"


def test_calculation_detector_routes_simple_arithmetic_to_tools():
    assert requires_math_tool("Tính giá trị 125 * 0.08") is True
    assert requires_math_tool("Giải thích khái niệm hàm số liên tục") is False


@pytest.mark.asyncio
async def test_agentic_loop_executes_structured_tool_result_before_answer():
    from app.agents.agentic_teacher import AgenticTeacherMixin

    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="simplify_expression",
            arguments='{"expr_str": "2 + 3"}',
        ),
    )
    first_response = SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="",
            tool_calls=[tool_call],
        ))],
    )
    final_response = SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="Kết quả đã kiểm chứng là 5.",
            tool_calls=None,
        ))],
    )

    agent = AgenticTeacherMixin()
    agent.openai_client = MagicMock()
    agent.openai_client.chat.completions.create = AsyncMock(
        side_effect=[first_response, final_response]
    )

    result = await agent.respond_agentic(
        messages=[
            {"role": "system", "content": "Giải toán."},
            {"role": "user", "content": "Tính 2 + 3."},
        ],
        question="Tính 2 + 3.",
        require_tool=True,
    )

    assert result == "Kết quả đã kiểm chứng là 5."
    calls = agent.openai_client.chat.completions.create.await_args_list
    assert calls[0].kwargs["tool_choice"] == "required"
    assert calls[1].kwargs["tool_choice"] == "auto"
    tool_messages = [
        message
        for message in calls[1].kwargs["messages"]
        if message.get("role") == "tool"
    ]
    assert tool_messages
    assert "5" in tool_messages[0]["content"]
