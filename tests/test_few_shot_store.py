"""
Tests for Dynamic Few-Shot Store.

Module: app/agents/few_shot_store.py
Covers:
- PedagogicalExample dataclass
- get_mastery_tier() — mastery → tier mapping
- select_few_shot() — example selection by tier and chapter
- build_few_shot_prompt() — prompt string generation
"""

import pytest
from app.agents.few_shot_store import (
    PedagogicalExample,
    get_mastery_tier,
    select_few_shot,
    build_few_shot_prompt,
    _EXAMPLES,
)


# ━━ get_mastery_tier() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetMasteryTier:
    """Test BKT mastery → pedagogical tier mapping."""

    def test_beginner(self):
        assert get_mastery_tier(0.0) == "beginner"
        assert get_mastery_tier(0.1) == "beginner"
        assert get_mastery_tier(0.29) == "beginner"

    def test_developing(self):
        assert get_mastery_tier(0.3) == "developing"
        assert get_mastery_tier(0.5) == "developing"
        assert get_mastery_tier(0.59) == "developing"

    def test_proficient(self):
        assert get_mastery_tier(0.6) == "proficient"
        assert get_mastery_tier(0.7) == "proficient"
        assert get_mastery_tier(0.84) == "proficient"

    def test_mastered(self):
        assert get_mastery_tier(0.85) == "mastered"
        assert get_mastery_tier(1.0) == "mastered"

    def test_boundary_consistency_with_bkt(self):
        """Should match BKTModel.get_mastery_level() boundaries exactly."""
        from app.knowledge_tracing.bkt import BKTModel
        bkt = BKTModel()
        test_points = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.85, 0.95, 1.0]
        for p in test_points:
            assert get_mastery_tier(p) == bkt.get_mastery_level(p)


# ━━ PedagogicalExample structure ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPedagogicalExamples:
    """Verify the pre-seeded example data."""

    def test_examples_not_empty(self):
        assert len(_EXAMPLES) > 0

    def test_all_tiers_covered(self):
        tiers = {ex.tier for ex in _EXAMPLES}
        assert "beginner" in tiers
        assert "developing" in tiers
        assert "proficient" in tiers
        assert "mastered" in tiers

    def test_example_fields(self):
        for ex in _EXAMPLES:
            assert isinstance(ex, PedagogicalExample)
            assert len(ex.tier) > 0
            assert len(ex.chapter) > 0
            assert len(ex.student_question) > 0
            assert len(ex.teacher_response) > 0
            assert len(ex.style_notes) > 0

    def test_tiers_are_valid(self):
        valid_tiers = {"beginner", "developing", "proficient", "mastered"}
        for ex in _EXAMPLES:
            assert ex.tier in valid_tiers


# ━━ select_few_shot() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSelectFewShot:
    """Test example selection logic."""

    def test_beginner_returns_beginner_example(self):
        ex = select_few_shot(p_mastery=0.1)
        assert ex is not None
        assert ex.tier == "beginner"

    def test_developing_returns_developing_example(self):
        ex = select_few_shot(p_mastery=0.4)
        assert ex is not None
        assert ex.tier == "developing"

    def test_proficient_returns_proficient_example(self):
        ex = select_few_shot(p_mastery=0.7)
        assert ex is not None
        assert ex.tier == "proficient"

    def test_mastered_returns_mastered_example(self):
        ex = select_few_shot(p_mastery=0.9)
        assert ex is not None
        assert ex.tier == "mastered"

    def test_chapter_specific_match(self):
        """When a chapter-specific example exists, prefer it."""
        ex = select_few_shot(p_mastery=0.1, chapter="Nguyên hàm và Tích phân")
        assert ex is not None
        assert ex.tier == "beginner"
        assert ex.chapter == "Nguyên hàm và Tích phân"

    def test_fallback_to_general_chapter(self):
        """When no chapter-specific example, fall back to 'general'."""
        ex = select_few_shot(p_mastery=0.1, chapter="Some Nonexistent Chapter")
        assert ex is not None
        assert ex.tier == "beginner"

    def test_chapter_none_works(self):
        ex = select_few_shot(p_mastery=0.5, chapter=None)
        assert ex is not None

    def test_developing_specific_chapter(self):
        ex = select_few_shot(p_mastery=0.45, chapter="Tổ hợp – Xác suất")
        assert ex is not None
        assert ex.tier == "developing"
        assert ex.chapter == "Tổ hợp – Xác suất"


# ━━ build_few_shot_prompt() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBuildFewShotPrompt:
    """Test prompt string builder."""

    def test_beginner_prompt(self):
        prompt = build_few_shot_prompt(p_mastery=0.1)
        assert "BEGINNER" in prompt
        assert "VÍ DỤ MẪU" in prompt
        assert len(prompt) > 50

    def test_developing_prompt(self):
        prompt = build_few_shot_prompt(p_mastery=0.4)
        assert "DEVELOPING" in prompt

    def test_proficient_prompt(self):
        prompt = build_few_shot_prompt(p_mastery=0.7)
        assert "PROFICIENT" in prompt

    def test_mastered_prompt(self):
        prompt = build_few_shot_prompt(p_mastery=0.9)
        assert "MASTERED" in prompt

    def test_contains_student_question(self):
        prompt = build_few_shot_prompt(p_mastery=0.1)
        assert "[Học sinh hỏi]" in prompt

    def test_contains_teacher_response(self):
        prompt = build_few_shot_prompt(p_mastery=0.1)
        assert "[Gia sư trả lời]" in prompt

    def test_contains_style_description(self):
        prompt = build_few_shot_prompt(p_mastery=0.1)
        assert "Phong cách yêu cầu" in prompt

    def test_mastery_percentage_in_prompt(self):
        prompt = build_few_shot_prompt(p_mastery=0.45)
        assert "45%" in prompt

    def test_with_chapter(self):
        prompt = build_few_shot_prompt(p_mastery=0.1, chapter="Nguyên hàm và Tích phân")
        assert len(prompt) > 0
