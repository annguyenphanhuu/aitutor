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

from app.agents.contracts import IntentClassification, GradingExtraction
from app.agents.teacher_agent import TEACHER_SYSTEM_PROMPT_EXAM
from app.db.session_state import upsert_session_state
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
