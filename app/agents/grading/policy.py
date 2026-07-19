"""Policy layer — nơi DUY NHẤT quyết định ghi BKT mastery từ kết quả chấm.

Quy tắc:
  abstain (không có bài làm)        → không ghi
  CAS đã kiểm chứng                 → ghi theo verdict CAS; LLM lệch → mismatch (CAS thắng)
  LLM-only, confidence ≥ threshold  → ghi theo LLM
  LLM-only, confidence < threshold  → không ghi (card hiển thị hedged)
"""

from __future__ import annotations

from typing import Optional

from app.agents.contracts import (
    GradingExtraction,
    MasteryDecision,
    PedagogyAssessment,
    VerifierResult,
)
from app.knowledge_tracing.skill_graph import SKILLS


def _per_skill(assessment: PedagogyAssessment, skill_ids: list[str]) -> dict[str, bool]:
    """Verdict từng kỹ năng từ assessment; skill thiếu lấy theo is_correct chung.

    Chỉ giữ skill_id có trong registry SKILLS — feedback-LLM đôi khi bịa key
    (vd "giải phương trình bậc hai"), để lọt sẽ sinh rác trong SkillMastery/BKT.
    """
    verdicts = {
        skill_id: verdict.passed
        for skill_id, verdict in assessment.skills_assessed.items()
        if skill_id in SKILLS
    }
    for skill_id in skill_ids:
        if skill_id in SKILLS:
            verdicts.setdefault(skill_id, assessment.is_correct)
    return verdicts


def decide(
    extraction: Optional[GradingExtraction],
    verification: Optional[VerifierResult],
    assessment: Optional[PedagogyAssessment],
    skill_ids: list[str],
    confidence_threshold: float,
) -> MasteryDecision:
    """Pure function — quyết định có ghi BKT không và ghi verdict nào."""
    if extraction is None or not extraction.gradable or assessment is None:
        return MasteryDecision(should_write=False, reason="abstain")

    if verification is not None and verification.verified:
        truth = verification.verdict == "correct"
        mismatch = assessment.is_correct != truth
        if mismatch:
            # CAS thắng: mọi skill nhận verdict theo CAS, bỏ per-skill của LLM
            skills = {sid: truth for sid in skill_ids if sid in SKILLS}
        else:
            skills = _per_skill(assessment, skill_ids)
        return MasteryDecision(
            should_write=bool(skills),
            skills_assessed=skills,
            reason="cas_verified",
            mismatch=mismatch,
        )

    if assessment.confidence >= confidence_threshold:
        skills = _per_skill(assessment, skill_ids)
        return MasteryDecision(
            should_write=bool(skills),
            skills_assessed=skills,
            reason="llm_high_confidence",
        )

    return MasteryDecision(should_write=False, reason="unverified_low_confidence")


def reconcile_assessment(
    assessment: PedagogyAssessment,
    verification: Optional[VerifierResult],
) -> PedagogyAssessment:
    """Ép assessment nhất quán với verdict CAS trước khi hiển thị score card.

    Chặn cả hai chiều (khác _enforce_consistency cũ chỉ chặn chiều false):
    CAS nói sai mà LLM nói đúng → hạ; CAS nói đúng mà LLM nói sai → nâng.
    """
    if verification is None or not verification.verified:
        return assessment
    truth = verification.verdict == "correct"
    if assessment.is_correct == truth:
        return assessment
    updated = assessment.model_copy()
    updated.is_correct = truth
    if truth:
        updated.score = max(updated.score, 1.0)
        updated.error_type = "none"
    else:
        updated.score = min(updated.score, 0.5)
        if updated.error_type == "none":
            updated.error_type = "conceptual"
    return updated
