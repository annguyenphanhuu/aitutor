"""
Tests for AssessorAgent consistency guards and history formatting.

Module: app/agents/assessor_agent.py
Covers:
- AssessorAgent._enforce_consistency() — is_correct/score coherence
- AssessorAgent._format_history() — transcript block for locating the problem
"""

from app.agents.assessor_agent import AssessorAgent


# ━━ _enforce_consistency() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEnforceConsistency:

    def test_failed_skill_forces_incorrect(self):
        result = AssessorAgent._enforce_consistency({
            "is_correct": True,
            "score": 1.0,
            "error_type": "none",
            "skills_assessed": {
                "derivative_basic": {"passed": True, "feedback": "ok"},
                "extremum": {"passed": False, "feedback": "đảo cực đại/cực tiểu"},
            },
        })
        assert result["is_correct"] is False
        assert result["score"] < 1.0

    def test_incorrect_with_full_score_gets_downgraded(self):
        result = AssessorAgent._enforce_consistency({
            "is_correct": False,
            "score": 1.0,
            "error_type": "conceptual",
            "skills_assessed": {},
        })
        assert result["score"] == 0.5

    def test_incorrect_all_skills_passed_capped_below_one(self):
        result = AssessorAgent._enforce_consistency({
            "is_correct": False,
            "score": 1.0,
            "error_type": "conceptual",
            "skills_assessed": {"extremum": {"passed": True}},
        })
        assert result["score"] <= 0.9

    def test_correct_result_untouched(self):
        result = AssessorAgent._enforce_consistency({
            "is_correct": True,
            "score": 1.0,
            "error_type": "none",
            "skills_assessed": {"extremum": {"passed": True}},
        })
        assert result["is_correct"] is True
        assert result["score"] == 1.0

    def test_legit_partial_score_kept(self):
        result = AssessorAgent._enforce_consistency({
            "is_correct": False,
            "score": 0.5,
            "error_type": "calculation",
            "skills_assessed": {"extremum": {"passed": False}},
        })
        assert result["score"] == 0.5

    def test_invalid_score_normalized(self):
        result = AssessorAgent._enforce_consistency({
            "is_correct": False,
            "score": "n/a",
            "error_type": "unknown",
            "skills_assessed": {},
        })
        assert result["score"] == 0.0

    def test_incorrect_gets_real_error_type(self):
        result = AssessorAgent._enforce_consistency({
            "is_correct": False,
            "score": 0.3,
            "error_type": "none",
            "skills_assessed": {},
        })
        assert result["error_type"] == "conceptual"


# ━━ _format_history() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFormatHistory:

    def test_empty_history(self):
        assert AssessorAgent._format_history(None) == "\n"
        assert AssessorAgent._format_history([]) == "\n"

    def test_includes_roles_and_content(self):
        block = AssessorAgent._format_history([
            {"role": "user", "content": "Tìm cực trị của f(x)=x^3-3x+1"},
            {"role": "assistant", "content": "Em tính f'(x) trước nhé."},
        ])
        assert "Hoc sinh: Tìm cực trị" in block
        assert "Gia su: Em tính f'(x)" in block
        assert "de bai goc" in block

    def test_truncates_to_last_six_messages(self):
        history = [{"role": "user", "content": f"msg-{i}"} for i in range(10)]
        block = AssessorAgent._format_history(history)
        assert "msg-3" not in block
        assert "msg-4" in block
        assert "msg-9" in block
