"""Tests for shared agent context and LLM compatibility helpers."""

from app.agents.context_utils import format_formulas, format_skill_names, normalize_skill_ids
from app.utils.llm import compatible_temperature, is_reasoning_model


def test_normalize_skill_ids_prioritizes_only_a_missing_primary():
    assert normalize_skill_ids(
        ["integral_definite", "derivative_basic", "integral_definite"],
        "derivative_basic",
    ) == ["integral_definite", "derivative_basic"]
    assert normalize_skill_ids(["integral_definite"], "derivative_basic") == [
        "derivative_basic",
        "integral_definite",
    ]


def test_prompt_formatters_use_canonical_registries():
    assert "Đạo hàm cơ bản" in format_skill_names(["derivative_basic"])
    assert format_formulas([]) == "Khong co cong thuc trong tam duoc gan metadata."


def test_reasoning_model_temperature_policy_is_centralized():
    assert is_reasoning_model("o4-mini-2025-04-16") is True
    assert compatible_temperature("o4-mini-2025-04-16", 0.2) == 1.0
    assert compatible_temperature("gpt-5.4-mini", 0.2) == 0.2
