"""
Tests for Skill Graph module.

Module: app/knowledge_tracing/skill_graph.py
Covers:
- SKILLS dictionary structure & data integrity
- get_skill()
- get_all_skills()
- get_prerequisites() — direct prerequisites
- get_deep_prerequisites() — BFS transitive prerequisites
- get_skills_by_chapter()
- get_chapters()
- find_weak_prerequisites()
"""

import pytest
from app.knowledge_tracing.skill_graph import (
    SKILLS,
    get_skill,
    get_all_skills,
    get_prerequisites,
    get_deep_prerequisites,
    get_skills_by_chapter,
    get_chapters,
    find_weak_prerequisites,
)


# ━━ SKILLS dictionary integrity ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSKILLSData:
    """Verify integrity of the hardcoded SKILLS dict."""

    def test_skills_not_empty(self):
        assert len(SKILLS) > 0

    def test_all_skills_have_required_keys(self):
        required = {"name", "chapter", "description", "prerequisites"}
        for skill_id, info in SKILLS.items():
            assert required.issubset(info.keys()), f"{skill_id} missing keys: {required - info.keys()}"

    def test_prerequisites_reference_valid_skills(self):
        for skill_id, info in SKILLS.items():
            for pre in info["prerequisites"]:
                assert pre in SKILLS, f"{skill_id} references unknown prerequisite '{pre}'"

    def test_no_self_referencing_prerequisites(self):
        for skill_id, info in SKILLS.items():
            assert skill_id not in info["prerequisites"], f"{skill_id} lists itself as prerequisite"

    def test_skills_have_non_empty_names(self):
        for skill_id, info in SKILLS.items():
            assert len(info["name"]) > 0, f"{skill_id} has empty name"

    def test_skills_have_non_empty_descriptions(self):
        for skill_id, info in SKILLS.items():
            assert len(info["description"]) > 0, f"{skill_id} has empty description"

    def test_known_skills_exist(self):
        """Verify key curriculum skills are present."""
        expected = [
            "derivative_basic", "derivative_rules", "derivative_applications",
            "function_survey", "primitive_basic", "integral_definite",
            "sequence_basic", "arithmetic_sequence", "geometric_sequence",
            "geometry_vectors", "probability_basic",
        ]
        for sid in expected:
            assert sid in SKILLS, f"Expected skill '{sid}' not found in SKILLS"


# ━━ get_skill() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetSkill:
    def test_existing_skill(self):
        s = get_skill("derivative_basic")
        assert s is not None
        assert s["name"] == "Đạo hàm cơ bản"

    def test_nonexistent_skill_returns_none(self):
        assert get_skill("nonexistent_skill") is None

    def test_empty_string(self):
        assert get_skill("") is None


# ━━ get_all_skills() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetAllSkills:
    def test_returns_same_as_skills(self):
        assert get_all_skills() is SKILLS

    def test_nonempty(self):
        assert len(get_all_skills()) > 0


# ━━ get_prerequisites() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetPrerequisites:
    def test_skill_with_no_prerequisites(self):
        prereqs = get_prerequisites("derivative_basic")
        assert prereqs == []

    def test_skill_with_one_prerequisite(self):
        prereqs = get_prerequisites("derivative_rules")
        assert "derivative_basic" in prereqs

    def test_skill_returns_list(self):
        prereqs = get_prerequisites("integral_definite")
        assert isinstance(prereqs, list)

    def test_nonexistent_skill_returns_empty(self):
        prereqs = get_prerequisites("fake_skill")
        assert prereqs == []

    def test_returns_copy_not_reference(self):
        """Ensure modifying result doesn't affect SKILLS."""
        prereqs = get_prerequisites("derivative_rules")
        original_len = len(SKILLS["derivative_rules"]["prerequisites"])
        prereqs.append("injected")
        assert len(SKILLS["derivative_rules"]["prerequisites"]) == original_len


# ━━ get_deep_prerequisites() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetDeepPrerequisites:
    def test_no_prerequisites(self):
        deep = get_deep_prerequisites("derivative_basic")
        assert deep == []

    def test_single_level(self):
        deep = get_deep_prerequisites("derivative_rules")
        assert "derivative_basic" in deep

    def test_multi_level(self):
        """derivative_applications → derivative_rules → derivative_basic"""
        deep = get_deep_prerequisites("derivative_applications")
        assert "derivative_rules" in deep
        assert "derivative_basic" in deep

    def test_deep_chain(self):
        """function_survey → derivative_applications → derivative_rules → derivative_basic"""
        deep = get_deep_prerequisites("function_survey")
        assert "derivative_applications" in deep
        assert "derivative_rules" in deep
        assert "derivative_basic" in deep
        assert len(deep) == 3

    def test_no_duplicates(self):
        deep = get_deep_prerequisites("function_survey")
        assert len(deep) == len(set(deep))

    def test_nonexistent_skill(self):
        deep = get_deep_prerequisites("nonexistent")
        assert deep == []

    def test_integral_chain(self):
        """integral_applications → integral_definite → primitive_basic → derivative_basic"""
        deep = get_deep_prerequisites("integral_applications")
        assert "integral_definite" in deep
        assert "primitive_basic" in deep
        assert "derivative_basic" in deep


# ━━ get_skills_by_chapter() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetSkillsByChapter:
    def test_dao_ham_chapter(self):
        skills = get_skills_by_chapter("Đạo hàm")
        assert "derivative_basic" in skills
        assert "derivative_rules" in skills
        assert "derivative_applications" in skills
        assert "function_survey" in skills

    def test_empty_chapter(self):
        skills = get_skills_by_chapter("Nonexistent Chapter")
        assert skills == []

    def test_returns_list_of_strings(self):
        skills = get_skills_by_chapter("Đạo hàm")
        assert all(isinstance(s, str) for s in skills)

    def test_geometry_chapter(self):
        skills = get_skills_by_chapter("Hình học không gian")
        assert "geometry_vectors" in skills
        assert "geometry_sphere" in skills


# ━━ get_chapters() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetChapters:
    def test_returns_list(self):
        chapters = get_chapters()
        assert isinstance(chapters, list)

    def test_known_chapters_present(self):
        chapters = get_chapters()
        assert "Đạo hàm" in chapters
        assert "Nguyên hàm và Tích phân" in chapters
        assert "Dãy số" in chapters

    def test_no_duplicates(self):
        chapters = get_chapters()
        assert len(chapters) == len(set(chapters))

    def test_expected_count(self):
        """There should be 6 chapters."""
        chapters = get_chapters()
        assert len(chapters) == 6


# ━━ find_weak_prerequisites() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFindWeakPrerequisites:
    def test_all_strong_returns_empty(self):
        masteries = {
            "derivative_basic": 0.9,
            "derivative_rules": 0.8,
        }
        weak = find_weak_prerequisites(masteries, "derivative_applications")
        assert weak == []

    def test_weak_prerequisite_found(self):
        masteries = {
            "derivative_basic": 0.3,  # weak
            "derivative_rules": 0.8,
        }
        weak = find_weak_prerequisites(masteries, "derivative_applications")
        assert "derivative_basic" in weak

    def test_missing_mastery_treated_as_zero(self):
        """Skills not in masteries dict → mastery=0.0 → always weak."""
        weak = find_weak_prerequisites({}, "derivative_applications")
        assert len(weak) > 0

    def test_custom_threshold(self):
        masteries = {"derivative_basic": 0.6}
        # With threshold=0.5 → not weak
        weak_low = find_weak_prerequisites(masteries, "derivative_rules", threshold=0.5)
        assert "derivative_basic" not in weak_low
        # With threshold=0.7 → weak
        weak_high = find_weak_prerequisites(masteries, "derivative_rules", threshold=0.7)
        assert "derivative_basic" in weak_high

    def test_no_prerequisites_returns_empty(self):
        weak = find_weak_prerequisites({}, "derivative_basic")
        assert weak == []

    def test_deep_weak_prerequisites(self):
        """Should find transitive weak prerequisites."""
        masteries = {
            "derivative_basic": 0.1,  # weak
            "derivative_rules": 0.1,  # weak
        }
        weak = find_weak_prerequisites(masteries, "function_survey")
        assert "derivative_basic" in weak
        assert "derivative_rules" in weak
