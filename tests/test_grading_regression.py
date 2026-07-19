"""Regression tests cho bug "✅ 100% cho mệnh đề sai của thầy".

Bug gốc: học sinh hỏi "Thầy em bảo rằng ∫x²dx = x³/2 + C... Thầy em đúng chứ?"
→ misclassified thành assess/is_answer_submission=true → Assessor bị ép chấm
→ ✅ 100% (mâu thuẫn với feedback) → BKT mastery bị ghi TĂNG.

Các lớp phòng thủ được test ở đây (mỗi lớp giả định lớp trước đã thủng):
  Lớp 2 — extractor abstain → teach, KHÔNG ghi SkillMastery
  Lớp 3 — CAS verifier bác verdict ✅ của LLM → card ❌, ghi verdict SAI
  Lớp 4 — policy không ghi khi LLM-only confidence thấp
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from app.agents.contracts import (
    GradingExtraction,
    IntentClassification,
    PedagogyAssessment,
    SkillVerdict,
)
from app.db.models import SkillMastery

BUG_MESSAGE = "Thầy em bảo rằng ∫x²dx = x³/2 + C. Em thấy cũng hợp lý vì đạo hàm của x³ liên quan đến x². Thầy em đúng chứ?"

# Classifier CỐ TÌNH trả kết quả misclassified như bug gốc — để test các lớp sau
MISCLASSIFIED = IntentClassification(
    intent="assess",
    is_answer_submission=True,
    skill_id="integral_definite",
    skill_ids=["integral_definite"],
)


@pytest.fixture
def engine():
    with patch("app.graph.nodes.TeacherAgent") as MockTeacher, \
         patch("app.graph.nodes.PlannerAgent"), \
         patch("app.graph.nodes.VisualizerAgent"), \
         patch("app.graph.nodes.GradingExtractor") as MockExtractor, \
         patch("app.graph.nodes.AsyncOpenAI"), \
         patch("app.graph.nodes.ChatOpenAI") as MockChat:
        MockTeacher.return_value.respond = AsyncMock(
            return_value="Thầy em nhầm rồi: ∫x²dx = x³/3 + C, vì (x³/3)' = x²."
        )
        MockExtractor.return_value.extract = AsyncMock(
            return_value=GradingExtraction(
                gradable=False, abstain_reason="third_party_claim"
            )
        )
        MockChat.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value={"raw": None, "parsed": IntentClassification(), "parsing_error": None}
        )
        from app.graph.service import GraphTutorEngine
        yield GraphTutorEngine()


def _stub_classifier(engine, classification):
    engine.nodes.classifier_llm = MagicMock()
    engine.nodes.classifier_llm.ainvoke = AsyncMock(
        return_value={"raw": None, "parsed": classification, "parsing_error": None}
    )


def _stub_feedback(engine, assessment):
    engine.nodes.feedback_llm = MagicMock()
    engine.nodes.feedback_llm.ainvoke = AsyncMock(
        return_value={"raw": None, "parsed": assessment, "parsing_error": None}
    )


async def _mastery_snapshot(db, user_id):
    result = await db.execute(
        select(SkillMastery).where(SkillMastery.user_id == user_id)
    )
    return {
        m.skill_id: (m.p_mastery, m.total_attempts, m.correct_attempts)
        for m in result.scalars().all()
    }


class TestLayer2ExtractorAbstain:
    """Classifier thủng (misclassified) — extractor phải abstain và không ghi BKT."""

    async def test_bug_message_gets_explanation_not_score_card(self, engine, seeded_db):
        db, user = seeded_db
        _stub_classifier(engine, MISCLASSIFIED)

        result = await engine.handle_message(
            db=db, message=BUG_MESSAGE, user_id=user.id
        )

        assert result["mode_used"] != "assess"
        assert "Điểm:" not in result["response"]
        assert "x³/3" in result["response"]

    async def test_bug_message_does_not_touch_mastery(self, engine, seeded_db):
        db, user = seeded_db
        _stub_classifier(engine, MISCLASSIFIED)
        before = await _mastery_snapshot(db, user.id)

        await engine.handle_message(db=db, message=BUG_MESSAGE, user_id=user.id)

        after = await _mastery_snapshot(db, user.id)
        assert after == before  # KHÔNG một dòng SkillMastery nào thay đổi

    async def test_abstain_no_problem_found(self, engine, seeded_db):
        db, user = seeded_db
        _stub_classifier(engine, MISCLASSIFIED)
        engine.nodes.extractor.extract = AsyncMock(
            return_value=GradingExtraction(gradable=False, abstain_reason="no_problem_found")
        )
        before = await _mastery_snapshot(db, user.id)

        result = await engine.handle_message(db=db, message=BUG_MESSAGE, user_id=user.id)

        assert result["mode_used"] != "assess"
        assert await _mastery_snapshot(db, user.id) == before


class TestLayer3CasOverride:
    """Extractor cũng thủng (chấm nhầm) — CAS phải bác verdict ✅ của LLM."""

    def _breach_extractor(self, engine):
        engine.nodes.extractor.extract = AsyncMock(return_value=GradingExtraction(
            gradable=True,
            task_kind="antiderivative",
            problem_statement="Tính ∫x²dx",
            student_final_answer="x³/2 + C",
            problem_expr="x**2",
            candidate_expr="x**3/2",
            confidence=0.9,
        ))
        # LLM chấm SAI hoàn toàn như bug gốc: khen đúng, 100%
        _stub_feedback(engine, PedagogyAssessment(
            is_correct=True, score=1.0, confidence=0.9, error_type="none",
            skills_assessed={"integral_definite": SkillVerdict(passed=True, feedback="Tốt")},
            feedback="Chính xác!", correct_solution="∫x²dx = x³/3 + C",
        ))

    async def test_cas_forces_failing_card(self, engine, seeded_db):
        db, user = seeded_db
        _stub_classifier(engine, MISCLASSIFIED)
        self._breach_extractor(engine)

        result = await engine.handle_message(db=db, message=BUG_MESSAGE, user_id=user.id)

        # Verifier THẬT chạy: diff(x³/2) = 3x²/2 ≠ x² → card phải là ❌
        assert result["mode_used"] == "assess"
        assert result["response"].lstrip().startswith("❌")
        assert "100%" not in result["response"]

    async def test_mastery_written_in_incorrect_direction(self, engine, seeded_db):
        db, user = seeded_db
        _stub_classifier(engine, MISCLASSIFIED)
        self._breach_extractor(engine)
        before = await _mastery_snapshot(db, user.id)

        await engine.handle_message(db=db, message=BUG_MESSAGE, user_id=user.id)

        after = await _mastery_snapshot(db, user.id)
        p_before, attempts_before, correct_before = before["integral_definite"]
        p_after, attempts_after, correct_after = after["integral_definite"]
        assert p_after < p_before                    # mastery GIẢM (verdict sai)
        assert attempts_after == attempts_before + 1
        assert correct_after == correct_before       # không được tính là đúng

    async def test_mismatch_logged_to_langfuse(self, engine, seeded_db):
        db, user = seeded_db
        _stub_classifier(engine, MISCLASSIFIED)
        self._breach_extractor(engine)

        with patch("app.utils.langfuse_client.score_trace") as mock_score:
            # Có trace_id thì mismatch phải được score
            trace = MagicMock()
            trace.id = "trace-123"
            await engine.handle_message(
                db=db, message=BUG_MESSAGE, user_id=user.id, langfuse_trace=trace
            )
            mock_score.assert_called_once()
            assert mock_score.call_args.kwargs["name"] == "grading_verdict_mismatch"


class TestLayer4LowConfidencePolicy:
    """CAS không kiểm chứng được — verdict LLM confidence thấp không được ghi BKT."""

    async def test_unverifiable_low_confidence_skips_mastery_write(self, engine, seeded_db):
        db, user = seeded_db
        _stub_classifier(engine, MISCLASSIFIED)
        engine.nodes.extractor.extract = AsyncMock(return_value=GradingExtraction(
            gradable=True, task_kind="other",   # CAS không check được "other"
            problem_statement="Bài chứng minh",
            student_final_answer="...",
            confidence=0.8,
        ))
        _stub_feedback(engine, PedagogyAssessment(
            is_correct=True, score=1.0, confidence=0.4, error_type="none",
            feedback="Có vẻ đúng", correct_solution="...",
        ))
        before = await _mastery_snapshot(db, user.id)

        result = await engine.handle_message(db=db, message="Em nghĩ là ...", user_id=user.id)

        assert await _mastery_snapshot(db, user.id) == before
        # Card phải có ghi chú hedged vì không kiểm chứng được
        assert "chưa được kiểm chứng" in result["response"]


class TestPositiveControl:
    """Đối chứng dương: bài làm ĐÚNG của chính học sinh vẫn được chấm ✅ và ghi mastery."""

    async def test_correct_own_work_passes_and_writes(self, engine, seeded_db):
        db, user = seeded_db
        _stub_classifier(engine, IntentClassification(
            intent="assess", is_answer_submission=True,
            skill_id="integral_definite", skill_ids=["integral_definite"],
        ))
        engine.nodes.extractor.extract = AsyncMock(return_value=GradingExtraction(
            gradable=True, task_kind="antiderivative",
            problem_statement="Tính ∫x²dx",
            student_final_answer="x³/3 + C",
            problem_expr="x**2", candidate_expr="x**3/3",
            confidence=0.95,
        ))
        _stub_feedback(engine, PedagogyAssessment(
            is_correct=True, score=1.0, confidence=0.9, error_type="none",
            skills_assessed={"integral_definite": SkillVerdict(passed=True, feedback="Tốt")},
            feedback="Chính xác!", correct_solution="∫x²dx = x³/3 + C",
        ))
        before = await _mastery_snapshot(db, user.id)

        result = await engine.handle_message(
            db=db, message="Em tính được ∫x²dx = x³/3 + C, đúng không ạ?", user_id=user.id
        )

        assert result["response"].lstrip().startswith("✅")
        after = await _mastery_snapshot(db, user.id)
        p_before, _, correct_before = before["integral_definite"]
        p_after, _, correct_after = after["integral_definite"]
        assert p_after > p_before
        assert correct_after == correct_before + 1
