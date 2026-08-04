"""Regression tests cho các fix chất lượng gia sư (báo cáo 2026-07-20).

F-01 — ép worked micro-step khi học sinh bí lặp lại cùng một bước.
F-02 — đại số nền (PT bậc hai) KHÔNG được gắn nhãn kỹ năng giải tích;
        follow-up kế thừa skill của bài đang giải trong session.
F-04 — chế độ exam có ngân sách độ dài rõ ràng trong prompt.

Pattern mock giống test_graph_routing.py: patch class agent tại
``app.graph.nodes.*`` TRƯỚC khi khởi tạo GraphTutorEngine.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.contracts import (
    GradingExtraction,
    IntentClassification,
    PedagogyAssessment,
    VerifierResult,
)
from app.agents.grading.policy import reconcile_assessment
from app.agents.teacher_agent import TEACHER_SYSTEM_PROMPT_EXAM
from app.db.session_state import upsert_session_state
from app.graph.formatters import RETRY_NUDGE, format_assessment
from app.graph.nodes import (
    STUCK_THRESHOLD,
    build_stuck_directive,
    count_consecutive_stuck,
    is_foundation_algebra,
    _is_calculus_skill,
)


# ── F-01: phát hiện bí lặp lại (pure function) ───────────────────────────

class TestCountConsecutiveStuck:
    def test_not_stuck_returns_zero(self):
        assert count_consecutive_stuck([], "giải giúp em bài này") == 0

    def test_single_stuck_message(self):
        assert count_consecutive_stuck([], "em không biết") == 1

    def test_two_consecutive_stuck(self):
        history = [
            {"role": "user", "content": "giải x^2-5x+6=0"},
            {"role": "assistant", "content": "Em thử tìm hai số có tổng 5, tích 6?"},
            {"role": "user", "content": "em không biết"},
            {"role": "assistant", "content": "Gợi ý: các ước của 6 là gì?"},
        ]
        assert count_consecutive_stuck(history, "vẫn không biết thầy ơi") == 2

    def test_three_consecutive_stuck(self):
        history = [
            {"role": "user", "content": "không biết"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "vẫn bí"},
            {"role": "assistant", "content": "..."},
        ]
        assert count_consecutive_stuck(history, "vẫn bí đúng bước này") == 3

    def test_resets_on_non_stuck_turn(self):
        # Học sinh đã trả lời được ở giữa → chuỗi bí bị ngắt
        history = [
            {"role": "user", "content": "không biết"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "à em hiểu rồi, ước của 6 là 1,2,3,6"},
            {"role": "assistant", "content": "..."},
        ]
        assert count_consecutive_stuck(history, "giờ em lại không biết") == 1

    def test_ignores_assistant_stuck_wording(self):
        # Câu "không biết" trong lượt assistant không tính
        history = [
            {"role": "assistant", "content": "Nếu em không biết thì thử đoán xem?"},
        ]
        assert count_consecutive_stuck(history, "em chịu") == 1


class TestBuildStuckDirective:
    def test_below_threshold_returns_none(self):
        assert build_stuck_directive(0) is None
        assert build_stuck_directive(STUCK_THRESHOLD - 1) is None

    def test_at_threshold_returns_directive(self):
        directive = build_stuck_directive(STUCK_THRESHOLD)
        assert directive is not None
        assert "worked micro-step".lower() in directive.lower() or "một bước" in directive.lower()
        assert "?" in directive  # cấm để trống ô '?'

    def test_directive_includes_count(self):
        assert str(STUCK_THRESHOLD) in build_stuck_directive(STUCK_THRESHOLD)


# ── F-02: guard đại số nền (pure function) ────────────────────────────────

class TestIsFoundationAlgebra:
    def test_bare_quadratic_solve(self):
        assert is_foundation_algebra("giải x^2-5x+6=0") is True

    def test_quadratic_unicode_superscript(self):
        assert is_foundation_algebra("giải x²-5x+6=0") is True

    def test_solve_equation_phrase(self):
        assert is_foundation_algebra("giải phương trình x^2 - 1 = 0") is True

    def test_calculus_context_is_not_foundation(self):
        assert is_foundation_algebra("tính đạo hàm của x^2 - 5x + 6") is False

    def test_extremum_context_is_not_foundation(self):
        assert is_foundation_algebra("tìm cực trị của f(x) = x^2 - 5x + 6") is False

    def test_plain_chat_is_not_foundation(self):
        assert is_foundation_algebra("chào thầy ạ") is False

    def test_theory_question_is_not_foundation(self):
        assert is_foundation_algebra("đạo hàm là gì?") is False


class TestIsCalculusSkill:
    def test_derivative_is_calculus(self):
        assert _is_calculus_skill("derivative_basic") is True

    def test_integral_is_calculus(self):
        assert _is_calculus_skill("integral_definite") is True

    def test_probability_is_not_calculus(self):
        assert _is_calculus_skill("probability_basic") is False

    def test_none_is_not_calculus(self):
        assert _is_calculus_skill(None) is False


# ── F-04: prompt exam có ngân sách độ dài ─────────────────────────────────

class TestExamPromptLengthBudget:
    def test_exam_prompt_has_length_budget(self):
        assert "ĐỘ DÀI" in TEACHER_SYSTEM_PROMPT_EXAM
        assert "8-12 dòng" in TEACHER_SYSTEM_PROMPT_EXAM


# ── Integration: engine với agent đã mock ─────────────────────────────────

@pytest.fixture
def engine():
    with patch("app.graph.nodes.TeacherAgent") as MockTeacher, \
         patch("app.graph.nodes.PlannerAgent"), \
         patch("app.graph.nodes.VisualizerAgent"), \
         patch("app.graph.nodes.GradingExtractor") as MockExtractor, \
         patch("app.graph.nodes.AsyncOpenAI"), \
         patch("app.graph.nodes.ChatOpenAI") as MockChat:
        MockTeacher.return_value.respond = AsyncMock(return_value="Giải thích của thầy")
        MockExtractor.return_value.extract = AsyncMock(
            return_value=GradingExtraction(gradable=False)
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


def _stub_feedback(engine, assessment=None):
    assessment = assessment or PedagogyAssessment(
        is_correct=False, score=0.0, feedback="Em sai ở bước cuối.",
        error_type="calculation", correct_solution="= 1/3",
    )
    engine.nodes.feedback_llm = MagicMock()
    engine.nodes.feedback_llm.ainvoke = AsyncMock(
        return_value={"raw": None, "parsed": assessment, "parsing_error": None}
    )


class TestClassifyFoundationGuard:
    async def test_quadratic_labeled_derivative_gets_cleared(self, engine):
        """F-02: classifier gán 'derivative_basic' cho PT bậc hai → guard gỡ về None."""
        _stub_classifier(engine, IntentClassification(
            intent="explain", skill_id="derivative_basic",
            skill_ids=["derivative_basic"],
        ))
        out = await engine.nodes.classify(
            {"message": "giải x^2-5x+6=0"}, {"configurable": {}}
        )
        assert out["skill_id"] is None
        assert out["skill_ids"] == []

    async def test_real_derivative_question_keeps_skill(self, engine):
        """Câu giải tích thật vẫn giữ nhãn — guard không bắt nhầm."""
        _stub_classifier(engine, IntentClassification(
            intent="explain", skill_id="derivative_basic",
            skill_ids=["derivative_basic"],
        ))
        out = await engine.nodes.classify(
            {"message": "tính đạo hàm của x^2 - 5x + 6"}, {"configurable": {}}
        )
        assert out["skill_id"] == "derivative_basic"


class TestSkillInheritance:
    async def test_followup_inherits_session_skill(self, engine, db_session):
        """F-02/F-03: follow-up không nhận diện được skill → kế thừa từ session."""
        session_id = 4242
        await upsert_session_state(
            db_session, session_id,
            current_problem="Tìm cực trị f(x)=x^3-3x+1",
            awaiting_answer=True,
            last_skill_ids=["derivative_applications"],
        )
        # classifier trả None skill cho câu follow-up mơ hồ
        _stub_classifier(engine, IntentClassification(intent="explain"))
        result = await engine.handle_message(
            db=db_session,
            message="vậy bước tiếp theo là gì thầy?",
            session_id=session_id,
        )
        assert result["skill_id"] == "derivative_applications"

    async def test_no_inheritance_without_session_state(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(intent="explain"))
        result = await engine.handle_message(
            db=db_session,
            message="vậy bước tiếp theo là gì thầy?",
            session_id=999,
        )
        assert result["skill_id"] is None


class TestStuckMicroStepWiring:
    _HISTORY = [
        {"role": "user", "content": "giải x^2-5x+6=0"},
        {"role": "assistant", "content": "Em thử tìm hai số tổng 5, tích 6 nhé?"},
        {"role": "user", "content": "em không biết"},
        {"role": "assistant", "content": "Gợi ý: các ước của 6 là gì?"},
    ]

    async def test_stuck_twice_passes_directive_to_teacher(self, engine, db_session):
        """F-01: bí lần 2 cùng bước → teach truyền extra_directive vào respond."""
        _stub_classifier(engine, IntentClassification(
            intent="explain", skill_id="derivative_basic",
            skill_ids=["derivative_basic"],
        ))
        await engine.handle_message(
            db=db_session, message="vẫn không biết thầy ơi",
            chat_history=self._HISTORY,
        )
        kwargs = engine.nodes.teacher.respond.call_args.kwargs
        assert kwargs["extra_directive"] is not None

    async def test_not_stuck_passes_no_directive(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(
            intent="explain", skill_id="derivative_basic",
            skill_ids=["derivative_basic"],
        ))
        await engine.handle_message(
            db=db_session, message="à em hiểu rồi, thử với 2 và 3 ạ",
            chat_history=self._HISTORY,
        )
        kwargs = engine.nodes.teacher.respond.call_args.kwargs
        assert kwargs["extra_directive"] is None


# ── Test 2026-07-29: classifier mù ngữ cảnh ──────────────────────────────


class TestClassifierReceivesHistory:
    """Classifier phải thấy lịch sử, nếu không sẽ đoán bừa tin nhắn tiếp nối."""

    async def test_history_reaches_classifier_llm(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(intent="explain"))
        await engine.handle_message(
            db=db_session,
            message="Em không biết làm ạ",
            chat_history=[
                {"role": "user", "content": "Tính tích phân từ 0 đến 1 của x*e^x dx"},
                {"role": "assistant", "content": "Em thử chọn u và dv xem nào?"},
            ],
        )
        messages = engine.nodes.classifier_llm.ainvoke.call_args.args[0]
        contents = [m.content for m in messages]
        assert any("x*e^x" in c for c in contents), "đề bài đang dở phải tới được classifier"
        assert contents[-1].endswith("Em không biết làm ạ")

    async def test_no_history_still_works(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(intent="explain"))
        await engine.handle_message(db=db_session, message="Đạo hàm của x^3 là gì ạ?")
        messages = engine.nodes.classifier_llm.ainvoke.call_args.args[0]
        assert len(messages) == 2  # system + tin nhắn hiện tại


class TestClassifierPromptContinuationRules:
    def test_prompt_covers_stuck_and_challenge(self):
        from app.graph.nodes import CLASSIFIER_SYSTEM_PROMPT

        assert "TIN NHẮN TIẾP NỐI" in CLASSIFIER_SYSTEM_PROMPT
        assert "thầy sai rồi" in CLASSIFIER_SYSTEM_PROMPT.lower()


class TestSocraticGradingContract:
    """Chấm bài trong Socratic không được xổ lời giải hay chấm bước trung gian."""

    _ASSESSMENT = PedagogyAssessment(
        is_correct=False, score=0.2, feedback="Em tìm sai nghiệm của y'=0.",
        error_type="calculation", correct_solution="GTLN là 1/e tại x=1.",
    )

    def test_socratic_wrong_answer_hides_solution(self):
        text = format_assessment(self._ASSESSMENT, reveal_solution=False)
        assert "1/e" not in text
        assert "Lời giải đúng" not in text
        assert RETRY_NUDGE in text
        # vẫn phải chấm và chỉ ra loại lỗi
        assert "20%" in text and "Lỗi tính toán" in text

    def test_exam_wrong_answer_still_shows_solution(self):
        text = format_assessment(self._ASSESSMENT, reveal_solution=True)
        assert "Lời giải đúng" in text and "1/e" in text

    def test_default_still_reveals(self):
        assert "Lời giải đúng" in format_assessment(self._ASSESSMENT)

    def test_hedged_note_still_appended_when_hidden(self):
        text = format_assessment(self._ASSESSMENT, hedged=True, reveal_solution=False)
        assert "chưa được kiểm chứng" in text

    def test_correct_answer_in_socratic_keeps_solution(self):
        """Em làm đúng = xong bài → hiện lời giải bình thường."""
        ok = PedagogyAssessment(
            is_correct=True, score=1.0, feedback="Đúng rồi em.",
            correct_solution="GTLN là 1/e tại x=1.",
        )
        assert "Lời giải đúng" in format_assessment(ok, reveal_solution=True)


class TestCasOverrideRewritesFeedback:
    """CAS lật verdict → feedback cũ của LLM phải bị thay, tránh thẻ tự mâu thuẫn."""

    def test_cas_says_wrong_rewrites_praise(self):
        llm_said_ok = PedagogyAssessment(
            is_correct=True, score=1.0,
            feedback="Kết quả cuối cùng của em là đúng: bằng 1/3.",
            correct_solution="= 1/3",
        )
        out = reconcile_assessment(
            llm_said_ok, VerifierResult(verified=True, verdict="incorrect")
        )
        assert out.is_correct is False
        assert "đúng: bằng 1/3" not in out.feedback
        assert "CHƯA ĐÚNG" in out.feedback
        assert "1/3" not in out.feedback  # không lộ đáp án qua feedback

    def test_cas_says_correct_rewrites_criticism(self):
        llm_said_wrong = PedagogyAssessment(
            is_correct=False, score=0.0, feedback="Em làm sai bước cuối.",
            correct_solution="= 1/3",
        )
        out = reconcile_assessment(
            llm_said_wrong, VerifierResult(verified=True, verdict="correct")
        )
        assert out.is_correct is True
        assert "ĐÚNG" in out.feedback

    def test_agreement_keeps_original_feedback(self):
        ok = PedagogyAssessment(
            is_correct=True, score=1.0, feedback="Chuẩn rồi em.", correct_solution="x",
        )
        out = reconcile_assessment(ok, VerifierResult(verified=True, verdict="correct"))
        assert out.feedback == "Chuẩn rồi em."

    def test_unverified_keeps_original_feedback(self):
        a = PedagogyAssessment(
            is_correct=True, score=1.0, feedback="Chuẩn rồi em.", correct_solution="x",
        )
        assert reconcile_assessment(a, VerifierResult(verified=False)).feedback == "Chuẩn rồi em."


class TestSocraticFeedbackPrompt:
    async def test_socratic_mode_adds_no_answer_constraint(self, engine, db_session):
        engine.nodes.extractor.extract = AsyncMock(return_value=GradingExtraction(
            gradable=True, problem_statement="Tính ∫(0->1) x^2 dx",
            task_kind="numeric", candidate_expr="1/2",
        ))
        _stub_classifier(engine, IntentClassification(
            intent="assess", is_answer_submission=True, skill_ids=["integral_definite"],
        ))
        _stub_feedback(engine)
        await engine.handle_message(
            db=db_session, mode="socratic", message="Em ra kết quả cuối cùng là 1/2 ạ",
        )
        system_prompt = engine.nodes.feedback_llm.ainvoke.call_args.args[0][0].content
        assert "CHẾ ĐỘ SOCRATIC" in system_prompt

    async def test_exam_mode_has_no_constraint(self, engine, db_session):
        engine.nodes.extractor.extract = AsyncMock(return_value=GradingExtraction(
            gradable=True, problem_statement="Tính ∫(0->1) x^2 dx",
            task_kind="numeric", candidate_expr="1/2",
        ))
        _stub_classifier(engine, IntentClassification(
            intent="assess", is_answer_submission=True, skill_ids=["integral_definite"],
        ))
        _stub_feedback(engine)
        await engine.handle_message(
            db=db_session, mode="exam", message="Em ra kết quả cuối cùng là 1/2 ạ",
        )
        system_prompt = engine.nodes.feedback_llm.ainvoke.call_args.args[0][0].content
        assert "CHẾ ĐỘ SOCRATIC" not in system_prompt


class TestIntermediateStepAbstain:
    async def test_socratic_pending_problem_flags_extractor(self, engine, db_session):
        """Cờ in_socratic_dialogue phải bật khi đang chờ em trả lời một bước."""
        session_id = 777
        await upsert_session_state(
            db_session, session_id,
            current_problem="Tìm GTLN của y = x·e^(-x) trên [0;2]",
            awaiting_answer=True,
            last_skill_ids=["derivative_applications"],
        )
        _stub_classifier(engine, IntentClassification(
            intent="assess", is_answer_submission=True,
        ))
        await engine.handle_message(
            db=db_session, mode="socratic", session_id=session_id,
            message="À em hiểu rồi, y' = e^(-x) - x·e^(-x)",
        )
        kwargs = engine.nodes.extractor.extract.call_args.kwargs
        assert kwargs["in_socratic_dialogue"] is True

    async def test_exam_mode_does_not_flag(self, engine, db_session):
        session_id = 778
        await upsert_session_state(
            db_session, session_id,
            current_problem="Tìm GTLN của y = x·e^(-x) trên [0;2]",
            awaiting_answer=True,
        )
        _stub_classifier(engine, IntentClassification(
            intent="assess", is_answer_submission=True,
        ))
        await engine.handle_message(
            db=db_session, mode="exam", session_id=session_id,
            message="Em ra GTLN là 1/e ạ",
        )
        assert engine.nodes.extractor.extract.call_args.kwargs["in_socratic_dialogue"] is False

    async def test_intermediate_step_abstain_routes_to_teach(self, engine, db_session):
        """gradable=false + intermediate_step → teach, KHÔNG ra score card."""
        engine.nodes.extractor.extract = AsyncMock(return_value=GradingExtraction(
            gradable=False, abstain_reason="intermediate_step",
        ))
        _stub_classifier(engine, IntentClassification(
            intent="assess", is_answer_submission=True,
        ))
        result = await engine.handle_message(
            db=db_session, mode="socratic",
            message="À em hiểu rồi, y' = e^(-x) - x·e^(-x)",
        )
        assert result["mode_used"] == "socratic"
        assert "Điểm:" not in result["response"]
        engine.nodes.teacher.respond.assert_awaited()


class TestRequestedModeWinsOverAnswerIntent:
    """Học sinh đã chọn Socratic/Exam thì 'cho em đáp án luôn' không lách được."""

    async def test_socratic_mode_not_overridden_by_answer_intent(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(intent="answer"))
        result = await engine.handle_message(
            db=db_session, mode="socratic",
            message="Cho em đáp án luôn đi thầy, mai em thi rồi",
        )
        assert result["mode_used"] == "socratic"
        assert engine.nodes.teacher.respond.call_args.kwargs["mode"] == "socratic"

    async def test_exam_mode_not_overridden_by_answer_intent(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(intent="answer"))
        result = await engine.handle_message(
            db=db_session, mode="exam", message="Tính ∫ x^2·e^x dx từ 0 đến 1",
        )
        assert result["mode_used"] == "exam"

    async def test_auto_mode_still_honors_answer_intent(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(intent="answer"))
        result = await engine.handle_message(
            db=db_session, mode="auto", message="Đáp án của bài này là gì ạ?",
        )
        assert result["mode_used"] == "answer"
