"""Tests cho app/agents/grading/policy.py — pure function, không mock."""

from app.agents.contracts import (
    GradingExtraction,
    PedagogyAssessment,
    SkillVerdict,
    VerifierResult,
)
from app.agents.grading.policy import decide, reconcile_assessment

THRESHOLD = 0.75

GRADABLE = GradingExtraction(
    gradable=True, task_kind="antiderivative",
    problem_expr="x**2", candidate_expr="x**3/2",
)
ABSTAINED = GradingExtraction(gradable=False, abstain_reason="third_party_claim")

CAS_INCORRECT = VerifierResult(verified=True, verdict="incorrect", method="antiderivative_roundtrip")
CAS_CORRECT = VerifierResult(verified=True, verdict="correct", method="antiderivative_roundtrip")
CAS_UNKNOWN = VerifierResult(verified=False, verdict="unknown", method="parse_error")


def _assessment(is_correct: bool, confidence: float) -> PedagogyAssessment:
    return PedagogyAssessment(
        is_correct=is_correct,
        score=1.0 if is_correct else 0.2,
        confidence=confidence,
        error_type="none" if is_correct else "conceptual",
        skills_assessed={"integral_definite": SkillVerdict(passed=is_correct)},
        feedback="test",
    )


class TestDecide:
    def test_abstain_never_writes(self):
        decision = decide(ABSTAINED, None, _assessment(True, 0.9), ["integral_definite"], THRESHOLD)
        assert decision.should_write is False
        assert decision.reason == "abstain"

    def test_no_extraction_never_writes(self):
        decision = decide(None, CAS_CORRECT, _assessment(True, 0.9), ["integral_definite"], THRESHOLD)
        assert decision.should_write is False

    def test_cas_verified_agrees_with_llm(self):
        decision = decide(GRADABLE, CAS_CORRECT, _assessment(True, 0.9), ["integral_definite"], THRESHOLD)
        assert decision.should_write is True
        assert decision.mismatch is False
        assert decision.skills_assessed == {"integral_definite": True}

    def test_cas_overrides_llm_mismatch(self):
        # Bug gốc: LLM nói ✅ nhưng CAS nói sai — CAS thắng, ghi verdict sai
        decision = decide(GRADABLE, CAS_INCORRECT, _assessment(True, 0.9), ["integral_definite"], THRESHOLD)
        assert decision.should_write is True
        assert decision.mismatch is True
        assert decision.skills_assessed == {"integral_definite": False}
        assert decision.reason == "cas_verified"

    def test_llm_only_high_confidence_writes(self):
        decision = decide(GRADABLE, CAS_UNKNOWN, _assessment(False, 0.9), ["integral_definite"], THRESHOLD)
        assert decision.should_write is True
        assert decision.reason == "llm_high_confidence"

    def test_llm_only_low_confidence_skips(self):
        decision = decide(GRADABLE, CAS_UNKNOWN, _assessment(False, 0.6), ["integral_definite"], THRESHOLD)
        assert decision.should_write is False
        assert decision.reason == "unverified_low_confidence"

    def test_missing_skill_falls_back_to_overall(self):
        assessment = PedagogyAssessment(is_correct=True, score=1.0, confidence=0.9, error_type="none")
        decision = decide(GRADABLE, CAS_CORRECT, assessment, ["integral_definite", "derivative_basic"], THRESHOLD)
        assert decision.skills_assessed == {"integral_definite": True, "derivative_basic": True}

    def test_hallucinated_skill_ids_are_filtered(self):
        # LLM bịa skill key tiếng Việt → không được lọt vào BKT write
        assessment = PedagogyAssessment(
            is_correct=True, score=1.0, confidence=0.9, error_type="none",
            skills_assessed={
                "integral_definite": SkillVerdict(passed=True),
                "giải phương trình bậc hai": SkillVerdict(passed=True),
            },
        )
        decision = decide(
            GRADABLE, CAS_CORRECT, assessment,
            ["integral_definite", "kiểm tra đầy đủ nghiệm"], THRESHOLD,
        )
        assert decision.skills_assessed == {"integral_definite": True}

    def test_mismatch_with_only_junk_skills_does_not_write(self):
        decision = decide(
            GRADABLE, CAS_INCORRECT, _assessment(True, 0.9).model_copy(update={"skills_assessed": {}}),
            ["skill không tồn tại"], THRESHOLD,
        )
        assert decision.should_write is False


class TestReconcileAssessment:
    def test_cas_incorrect_forces_failure(self):
        reconciled = reconcile_assessment(_assessment(True, 0.9), CAS_INCORRECT)
        assert reconciled.is_correct is False
        assert reconciled.score <= 0.5
        assert reconciled.error_type != "none"

    def test_cas_correct_forces_pass(self):
        reconciled = reconcile_assessment(_assessment(False, 0.9), CAS_CORRECT)
        assert reconciled.is_correct is True
        assert reconciled.score == 1.0

    def test_unverified_leaves_assessment_untouched(self):
        original = _assessment(True, 0.9)
        reconciled = reconcile_assessment(original, CAS_UNKNOWN)
        assert reconciled == original

    def test_agreement_leaves_assessment_untouched(self):
        original = _assessment(False, 0.9)
        reconciled = reconcile_assessment(original, CAS_INCORRECT)
        assert reconciled == original
