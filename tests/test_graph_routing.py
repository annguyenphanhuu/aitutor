"""Tests routing + parity cho LangGraph engine (app/graph/*).

Pattern mock giống test_orchestrator.py: patch các class agent tại
``app.graph.nodes.*`` TRƯỚC khi khởi tạo TutorNodes/GraphTutorEngine.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.contracts import (
    GradingExtraction,
    IntentClassification,
    PedagogyAssessment,
    SkillVerdict,
)
from app.graph.nodes import (
    _history_messages_for_classifier,
    _is_challenge_message,
    find_original_question,
    route_gradable,
    route_intent,
    should_stay_on_problem,
)
from app.graph.state import SessionSnapshot


# ── Pure routing functions ──────────────────────────────────────────────


class TestRouteIntent:
    def _state(self, intent, is_submission=False):
        return {
            "intent": intent,
            "classification": IntentClassification(
                intent=intent, is_answer_submission=is_submission
            ),
        }

    def test_assess_with_submission_goes_to_grading(self):
        assert route_intent(self._state("assess", True)) == "grade_extract"

    def test_assess_without_submission_goes_to_teach(self):
        # Bug gốc lớp 1: "thầy em bảo X đúng không" không được vào grading
        assert route_intent(self._state("assess", False)) == "teach"

    def test_explain_goes_to_teach(self):
        assert route_intent(self._state("explain")) == "teach"

    def test_answer_goes_to_teach(self):
        assert route_intent(self._state("answer")) == "teach"

    @pytest.mark.parametrize(
        "intent", ["off_topic", "plan", "quiz", "review", "diagnostic", "visualize"]
    )
    def test_static_intents_route_to_own_node(self, intent):
        assert route_intent(self._state(intent)) == intent

    @pytest.mark.parametrize("intent", ["greeting", "motivation"])
    def test_social_intents_route_to_social_node(self, intent):
        assert route_intent(self._state(intent)) == "social"

    def test_missing_intent_defaults_to_teach(self):
        assert route_intent({}) == "teach"


# ── Guard: giữ học sinh ở lại bài đang dở ───────────────────────────────


def _pending(intent, message):
    """State có một bài đang dở trong session."""
    return {
        "intent": intent,
        "message": message,
        "classification": IntentClassification(intent=intent),
        "session_state": SessionSnapshot(
            current_problem="Tính tích phân từ 0 đến 1 của x*e^x dx",
            awaiting_answer=True,
        ),
    }


class TestStayOnProblemGuard:
    @pytest.mark.parametrize("intent", ["greeting", "motivation", "off_topic"])
    def test_stuck_message_mid_problem_goes_to_teach(self, intent):
        """Lỗi gốc: 'em không biết làm ạ' → greeting → social xổ trọn đáp án."""
        assert route_intent(_pending(intent, "Em không biết làm ạ")) == "teach"

    def test_challenge_message_mid_problem_goes_to_teach(self):
        """Lỗi gốc: 'thầy sai rồi' → off_topic → canned text bỏ rơi tranh luận."""
        assert route_intent(_pending("off_topic", "Thầy sai rồi, em không đồng ý")) == "teach"

    def test_real_off_topic_mid_problem_still_off_topic(self):
        # Guard hẹp: câu lạc đề thật vẫn phải vào off_topic dù đang dở bài
        state = _pending("off_topic", "Thầy ơi tối nay đá bóng đội nào thắng ạ?")
        assert route_intent(state) == "off_topic"

    def test_real_greeting_mid_problem_still_social(self):
        assert route_intent(_pending("greeting", "Chào thầy ạ")) == "social"

    def test_stuck_message_without_pending_problem_stays_social(self):
        # Chưa có bài nào đang dở → 'em chịu' đúng là tâm sự, không phải bí bài
        state = {
            "intent": "motivation",
            "message": "Em chịu rồi thầy ơi",
            "classification": IntentClassification(intent="motivation"),
            "session_state": SessionSnapshot(),
        }
        assert route_intent(state) == "social"

    def test_no_session_state_stays_social(self):
        state = {
            "intent": "greeting",
            "message": "Em không biết làm ạ",
            "classification": IntentClassification(intent="greeting"),
        }
        assert should_stay_on_problem(state) is False
        assert route_intent(state) == "social"

    def test_explain_intent_unaffected(self):
        assert should_stay_on_problem(_pending("explain", "Em không biết làm ạ")) is False

    @pytest.mark.parametrize("text", [
        "Thầy sai rồi", "em không đồng ý", "Sách em ghi khác mà", "em vẫn nghĩ là cos(2x)",
    ])
    def test_challenge_patterns(self, text):
        assert _is_challenge_message(text) is True

    @pytest.mark.parametrize("text", ["", "Em cảm ơn thầy", "Đáp án là 2"])
    def test_non_challenge_messages(self, text):
        assert _is_challenge_message(text) is False


# ── Ngữ cảnh truyền cho classifier ──────────────────────────────────────


class TestClassifierHistoryMessages:
    def test_empty_history(self):
        assert _history_messages_for_classifier(None) == []
        assert _history_messages_for_classifier([]) == []

    def test_keeps_roles_in_order(self):
        msgs = _history_messages_for_classifier([
            {"role": "user", "content": "Tính ∫x·e^x dx"},
            {"role": "assistant", "content": "Em đặt u là phần nào?"},
        ])
        assert [m.type for m in msgs] == ["human", "ai"]
        assert msgs[0].content == "Tính ∫x·e^x dx"

    def test_keeps_only_last_turns(self):
        history = [{"role": "user", "content": f"m{i}"} for i in range(10)]
        msgs = _history_messages_for_classifier(history)
        assert len(msgs) == 4
        assert msgs[-1].content == "m9"

    def test_strips_thinking_block(self):
        msgs = _history_messages_for_classifier([
            {"role": "assistant", "content": "<thinking>log nội bộ</thinking>Câu hỏi của thầy"},
        ])
        assert msgs[0].content == "Câu hỏi của thầy"

    def test_drops_empty_after_strip(self):
        msgs = _history_messages_for_classifier([
            {"role": "assistant", "content": "<thinking>chỉ có log</thinking>"},
            {"role": "user", "content": "em không biết"},
        ])
        assert len(msgs) == 1
        assert msgs[0].type == "human"

    def test_truncates_long_turn(self):
        msgs = _history_messages_for_classifier([
            {"role": "user", "content": "x" * 5000},
        ])
        assert len(msgs[0].content) == 400


class TestRouteGradable:
    def test_gradable_goes_to_verify(self):
        state = {"extraction": GradingExtraction(gradable=True)}
        assert route_gradable(state) == "grade_verify"

    def test_abstain_goes_to_teach(self):
        # Bug gốc lớp 2: extractor abstain → giải thích thay vì chấm
        state = {
            "extraction": GradingExtraction(
                gradable=False, abstain_reason="third_party_claim"
            )
        }
        assert route_gradable(state) == "teach"

    def test_missing_extraction_goes_to_teach(self):
        assert route_gradable({}) == "teach"


class TestFindOriginalQuestion:
    def test_skips_assessment_outputs(self):
        history = [
            {"role": "assistant", "content": "Giải phương trình x² = 4"},
            {"role": "user", "content": "x = 2"},
            {"role": "assistant", "content": "✅ **Điểm: 100%**\n..."},
        ]
        assert find_original_question(history, "fallback") == "Giải phương trình x² = 4"

    def test_fallback_when_no_history(self):
        assert find_original_question([], "fallback") == "fallback"


# ── Graph end-to-end với agent đã mock ──────────────────────────────────


@pytest.fixture
def engine():
    """GraphTutorEngine với mọi agent/LLM được mock (không gọi mạng)."""
    with patch("app.graph.nodes.TeacherAgent") as MockTeacher, \
         patch("app.graph.nodes.PlannerAgent") as MockPlanner, \
         patch("app.graph.nodes.VisualizerAgent") as MockVisualizer, \
         patch("app.graph.nodes.GradingExtractor") as MockExtractor, \
         patch("app.graph.nodes.AsyncOpenAI"), \
         patch("app.graph.nodes.ChatOpenAI") as MockChat:
        MockTeacher.return_value.respond = AsyncMock(return_value="Giải thích của thầy")
        MockPlanner.return_value.create_plan = AsyncMock(return_value={
            "summary": "Kế hoạch", "priorities": [], "encouragement": ""
        })
        MockVisualizer.return_value.generate_function_plot = MagicMock(
            return_value={"vis_type": "plot", "data": {}}
        )
        MockExtractor.return_value.extract = AsyncMock(
            return_value=GradingExtraction(gradable=False, abstain_reason="other")
        )
        MockChat.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value={"raw": None, "parsed": IntentClassification(), "parsing_error": None}
        )

        from app.graph.service import GraphTutorEngine
        yield GraphTutorEngine()


def _stub_classifier(engine, classification: IntentClassification):
    engine.nodes.classifier_llm = MagicMock()
    engine.nodes.classifier_llm.ainvoke = AsyncMock(
        return_value={"raw": None, "parsed": classification, "parsing_error": None}
    )


def _stub_feedback(engine, assessment: PedagogyAssessment):
    engine.nodes.feedback_llm = MagicMock()
    engine.nodes.feedback_llm.ainvoke = AsyncMock(
        return_value={"raw": None, "parsed": assessment, "parsing_error": None}
    )


class TestGraphParity:
    """So khớp key + giá trị result dict với contract của Orchestrator cũ."""

    async def test_off_topic_result_keys(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(intent="off_topic"))
        result = await engine.handle_message(db=db_session, message="hôm nay trời đẹp quá")
        assert set(result.keys()) == {
            "response", "skill_id", "skill_name", "mastery_level", "mode_used"
        }
        assert result["mode_used"] == "off_topic"
        assert result["skill_id"] is None

    async def test_greeting_uses_social_llm_response(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(intent="greeting"))
        engine.nodes.social_llm = MagicMock()
        engine.nodes.social_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="Chào Minh! Thầy đây, em muốn học phần nào?")
        )
        result = await engine.handle_message(db=db_session, message="Em tên Minh, chào thầy ạ")
        assert result["mode_used"] == "greeting"
        assert "Chào Minh" in result["response"]
        assert set(result.keys()) == {
            "response", "skill_id", "skill_name", "mastery_level", "mode_used"
        }

    async def test_motivation_llm_error_falls_back_to_static_text(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(intent="motivation"))
        engine.nodes.social_llm = MagicMock()
        engine.nodes.social_llm.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        result = await engine.handle_message(db=db_session, message="em nản quá thầy ơi")
        assert result["mode_used"] == "motivation"
        assert "chẩn đoán năng lực" in result["response"]

    async def test_explain_result_keys(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(
            intent="explain", skill_id="derivative_basic",
            skill_ids=["derivative_basic"],
        ))
        result = await engine.handle_message(db=db_session, message="đạo hàm là gì?")
        assert set(result.keys()) == {
            "response", "skill_id", "skill_ids", "formula_ids",
            "skill_name", "mastery_level", "mode_used", "visualization",
        }
        assert result["response"] == "Giải thích của thầy"
        assert result["mode_used"] in ("socratic", "exam")

    async def test_answer_result_has_no_visualization_key(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(
            intent="answer", skill_id="derivative_basic",
            skill_ids=["derivative_basic"],
        ))
        result = await engine.handle_message(db=db_session, message="đáp án là gì?")
        assert "visualization" not in result
        assert result["mode_used"] == "answer"

    async def test_quiz_result(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(intent="quiz"))
        result = await engine.handle_message(db=db_session, message="cho em làm quiz")
        assert result["skill_id"] == "derivative_basic"
        assert result["mode_used"] == "quiz"
        assert "Bài kiểm tra" in result["response"]

    async def test_plan_review_diagnostic(self, engine, db_session):
        for intent, marker in [
            ("plan", "📋"), ("review", "Ôn tập chống quên lãng"), ("diagnostic", "Chẩn Đoán"),
        ]:
            _stub_classifier(engine, IntentClassification(intent=intent))
            result = await engine.handle_message(db=db_session, message="...")
            assert marker in result["response"]
            assert result["mode_used"] == intent

    async def test_mode_auto_resolves_socratic_for_low_mastery(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(
            intent="explain", skill_id="derivative_basic",
            skill_ids=["derivative_basic"],
        ))
        result = await engine.handle_message(
            db=db_session, message="giúp em bài này", mode="auto"
        )
        assert result["mode_used"] == "socratic"

    async def test_explicit_mode_respected(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(intent="explain"))
        result = await engine.handle_message(
            db=db_session, message="giúp em", mode="exam"
        )
        assert result["mode_used"] == "exam"


class TestGraphGradingFlow:
    async def test_abstain_routes_to_teach(self, engine, db_session):
        """assess + is_answer_submission=true nhưng extractor abstain → teach."""
        _stub_classifier(engine, IntentClassification(
            intent="assess", is_answer_submission=True,
            skill_id="integral_definite", skill_ids=["integral_definite"],
        ))
        # extractor fixture mặc định trả gradable=False
        result = await engine.handle_message(
            db=db_session, message="Thầy em bảo ∫x²dx = x³/2 + C, đúng chứ?"
        )
        assert result["mode_used"] != "assess"
        assert "Điểm:" not in result["response"]
        assert result["response"] == "Giải thích của thầy"

    async def test_gradable_flow_produces_score_card(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(
            intent="assess", is_answer_submission=True,
            skill_id="integral_definite", skill_ids=["integral_definite"],
        ))
        engine.nodes.extractor.extract = AsyncMock(return_value=GradingExtraction(
            gradable=True, task_kind="antiderivative",
            problem_statement="Tính ∫x²dx",
            problem_expr="x**2", candidate_expr="x**3/3",
            confidence=0.95,
        ))
        _stub_feedback(engine, PedagogyAssessment(
            is_correct=True, score=1.0, confidence=0.9, error_type="none",
            skills_assessed={"integral_definite": SkillVerdict(passed=True, feedback="Tốt")},
            feedback="Chính xác!", correct_solution="∫x²dx = x³/3 + C",
        ))
        result = await engine.handle_message(
            db=db_session, message="Em tính được ∫x²dx = x³/3 + C, đúng không ạ?"
        )
        assert result["mode_used"] == "assess"
        assert set(result.keys()) == {
            "response", "skill_id", "skill_ids", "formula_ids",
            "skill_name", "mastery_level", "mode_used",
        }
        assert "✅" in result["response"]
        assert "100%" in result["response"]
