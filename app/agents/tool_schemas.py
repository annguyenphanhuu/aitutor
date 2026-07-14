"""OpenAI function schemas derived from the canonical SymPy tool registry."""

from __future__ import annotations

from langchain_core.utils.function_calling import convert_to_openai_tool

from app.agents.tools import MATH_TOOL_LIST, MATH_TOOLS


OPENAI_MATH_TOOLS: list[dict] = [
    convert_to_openai_tool(math_tool) for math_tool in MATH_TOOL_LIST
]
TOOL_NAMES = set(MATH_TOOLS)


def _get_math_tools() -> dict:
    """Return the plain-function registry used to execute tool calls."""
    return MATH_TOOLS
