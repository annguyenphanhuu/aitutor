"""
Tests for Reflection Engine helpers.

Module: app/agents/reflection.py
Covers:
- ReflectionResult — container class
- strip_thinking_tags() — tag removal
- extract_thinking_block() — tag extraction
- ReflectionResult.format_thinking_block()
- ReflectionResult.build_full_response()
"""

import pytest
from app.agents.reflection import (
    ReflectionResult,
    strip_thinking_tags,
    extract_thinking_block,
)


# ━━ strip_thinking_tags() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestStripThinkingTags:

    def test_removes_thinking_block(self):
        text = "<thinking>\nSome internal reasoning\n</thinking>\n\nFinal answer."
        result = strip_thinking_tags(text)
        assert "<thinking>" not in result
        assert "</thinking>" not in result
        assert "Final answer." in result

    def test_no_thinking_tags(self):
        text = "Just a normal response."
        result = strip_thinking_tags(text)
        assert result == "Just a normal response."

    def test_empty_thinking_block(self):
        text = "<thinking></thinking>Answer here."
        result = strip_thinking_tags(text)
        assert "Answer here." in result
        assert "<thinking>" not in result

    def test_multiline_thinking(self):
        text = "<thinking>\nLine 1\nLine 2\nLine 3\n</thinking>\n\nResponse."
        result = strip_thinking_tags(text)
        assert "Line 1" not in result
        assert "Response." in result

    def test_multiple_thinking_blocks(self):
        text = "<thinking>Block1</thinking>Middle<thinking>Block2</thinking>End"
        result = strip_thinking_tags(text)
        assert "Block1" not in result
        assert "Block2" not in result
        assert "Middle" in result
        assert "End" in result

    def test_preserves_surrounding_whitespace_trimmed(self):
        text = "  <thinking>x</thinking>  Answer  "
        result = strip_thinking_tags(text)
        assert "Answer" in result


# ━━ extract_thinking_block() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestExtractThinkingBlock:

    def test_extracts_thinking_content(self):
        text = "<thinking>\n🔍 Checking math...\n✅ All good.\n</thinking>\n\nFinal answer."
        thinking, clean = extract_thinking_block(text)
        assert "🔍 Checking math..." in thinking
        assert "✅ All good." in thinking
        assert "Final answer." in clean
        assert "<thinking>" not in clean

    def test_no_thinking_block(self):
        text = "Just a plain response."
        thinking, clean = extract_thinking_block(text)
        assert thinking == ""
        assert clean == "Just a plain response."

    def test_empty_thinking(self):
        text = "<thinking></thinking>Response."
        thinking, clean = extract_thinking_block(text)
        assert thinking == ""
        assert "Response." in clean

    def test_returns_tuple(self):
        result = extract_thinking_block("test")
        assert isinstance(result, tuple)
        assert len(result) == 2


# ━━ ReflectionResult ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestReflectionResult:

    def test_basic_construction(self):
        r = ReflectionResult(
            final_answer="The answer is 42.",
            thinking_log=["Step 1", "Step 2"],
            was_corrected=False,
            verifications=[],
        )
        assert r.final_answer == "The answer is 42."
        assert r.was_corrected is False
        assert len(r.thinking_log) == 2
        assert len(r.verifications) == 0

    def test_corrected_flag(self):
        r = ReflectionResult(
            final_answer="Corrected answer.",
            thinking_log=["Found error", "Fixed it"],
            was_corrected=True,
            verifications=[{"tool": "compute_derivative", "result": {"success": True}}],
        )
        assert r.was_corrected is True
        assert len(r.verifications) == 1

    def test_format_thinking_block(self):
        r = ReflectionResult(
            final_answer="Answer",
            thinking_log=["🔍 Checking", "✅ OK"],
            was_corrected=False,
            verifications=[],
        )
        block = r.format_thinking_block()
        assert block.startswith("<thinking>")
        assert block.endswith("</thinking>")
        assert "🔍 Checking" in block
        assert "✅ OK" in block

    def test_build_full_response_with_thinking(self):
        r = ReflectionResult(
            final_answer="Final text.",
            thinking_log=["Step 1"],
            was_corrected=False,
            verifications=[],
        )
        full = r.build_full_response(include_thinking=True)
        assert "<thinking>" in full
        assert "Final text." in full

    def test_build_full_response_without_thinking(self):
        r = ReflectionResult(
            final_answer="Final text.",
            thinking_log=["Step 1"],
            was_corrected=False,
            verifications=[],
        )
        full = r.build_full_response(include_thinking=False)
        assert "<thinking>" not in full
        assert "Final text." in full

    def test_empty_thinking_log(self):
        r = ReflectionResult(
            final_answer="Answer",
            thinking_log=[],
            was_corrected=False,
            verifications=[],
        )
        full = r.build_full_response(include_thinking=True)
        assert "Answer" in full
