"""
Tests for Pydantic Schemas validation.

Module: app/db/schemas.py
Covers:
- ChatRequest / ChatResponse — chat flow schemas
- AssessmentRequest / AssessmentResponse — assessment schemas
- QuizGenerateRequest — quiz generation params
- QuizQuestionOut — question output
- QuizAnswerRequest / QuizAnswerResponse — answer flow
- DiagnosticStartResponse / DiagnosticAnswerRequest
- ReviewCardOut / ReviewSubmitRequest
- VisualizationData
- RoadmapNodeOut / RoadmapResponse
- InsightOut
- ConversationSessionOut / ChatMessageOut
"""

import pytest
from datetime import datetime
from pydantic import ValidationError
from app.db.schemas import (
    ChatRequest,
    ChatResponse,
    AssessmentRequest,
    AssessmentResponse,
    StudyPlanResponse,
    SkillMasteryResponse,
    DashboardResponse,
    QuizGenerateRequest,
    QuizQuestionOut,
    QuizSessionOut,
    QuizAnswerRequest,
    QuizAnswerResponse,
    QuizResultResponse,
    DiagnosticStartResponse,
    DiagnosticAnswerRequest,
    DiagnosticAnswerResponse,
    DiagnosticResultResponse,
    ReviewCardOut,
    ReviewDueResponse,
    ReviewSubmitRequest,
    ReviewSubmitResponse,
    VisualizationData,
    RoadmapNodeOut,
    RoadmapResponse,
    InsightOut,
    ChatMessageOut,
    ConversationSessionOut,
    ConversationHistoryResponse,
)


# ━━ ChatRequest ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestChatRequest:

    def test_basic(self):
        r = ChatRequest(message="Hello")
        assert r.message == "Hello"
        assert r.mode == "auto"
        assert r.session_id is None

    def test_with_mode(self):
        r = ChatRequest(message="Question", mode="socratic")
        assert r.mode == "socratic"

    def test_with_session_id(self):
        r = ChatRequest(message="Hi", session_id=5)
        assert r.session_id == 5

    def test_missing_message_fails(self):
        with pytest.raises(ValidationError):
            ChatRequest()


# ━━ ChatResponse ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestChatResponse:

    def test_minimal(self):
        r = ChatResponse(response="Answer text")
        assert r.response == "Answer text"
        assert r.mode_used == "socratic"
        assert r.skill_id is None

    def test_full(self):
        r = ChatResponse(
            response="Answer",
            skill_id="derivative_basic",
            skill_name="Đạo hàm cơ bản",
            mastery_level=0.75,
            mode_used="exam",
            visualization={"vis_type": "plot"},
            session_id=42,
        )
        assert r.mastery_level == 0.75
        assert r.visualization["vis_type"] == "plot"


# ━━ AssessmentRequest / Response ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestAssessmentSchemas:

    def test_request(self):
        r = AssessmentRequest(question="What is 2+2?", student_answer="4")
        assert r.question == "What is 2+2?"
        assert r.student_answer == "4"

    def test_response(self):
        r = AssessmentResponse(
            is_correct=True,
            score=1.0,
            feedback="Chính xác!",
            correct_solution="2+2=4",
        )
        assert r.is_correct is True
        assert r.score == 1.0


# ━━ QuizGenerateRequest ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestQuizGenerateRequest:

    def test_defaults(self):
        r = QuizGenerateRequest(skill_id="derivative_basic")
        assert r.difficulty == 1
        assert r.count == 5
        assert r.exam_format is False

    def test_custom(self):
        r = QuizGenerateRequest(skill_id="integral_definite", difficulty=3, count=10, exam_format=True)
        assert r.difficulty == 3
        assert r.count == 10
        assert r.exam_format is True


# ━━ QuizQuestionOut ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestQuizQuestionOut:

    def test_mcq(self):
        q = QuizQuestionOut(
            id=1,
            question_type="mcq",
            question_latex="$f'(x) = ?$",
            choices=["$2x$", "$x^2$", "$3x$", "$x$"],
            skill_id="derivative_basic",
            difficulty=1,
        )
        assert q.question_type == "mcq"
        assert len(q.choices) == 4

    def test_true_false(self):
        q = QuizQuestionOut(
            id=2,
            question_type="true_false",
            question_latex="Statements about f(x)",
            statements=[{"text": "A", "correct": True}],
            skill_id="derivative_basic",
            difficulty=1,
        )
        assert q.question_type == "true_false"

    def test_short_answer(self):
        q = QuizQuestionOut(
            id=3,
            question_type="short_answer",
            question_latex="Compute...",
            skill_id="integral_definite",
            difficulty=2,
        )
        assert q.points == 0.25  # default


# ━━ QuizAnswerRequest ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestQuizAnswerRequest:

    def test_mcq_answer(self):
        r = QuizAnswerRequest(session_id=1, question_id=1, selected_index=2)
        assert r.selected_index == 2

    def test_tf_answer(self):
        r = QuizAnswerRequest(session_id=1, question_id=2, tf_answers=[True, False, True, False])
        assert len(r.tf_answers) == 4

    def test_short_answer(self):
        r = QuizAnswerRequest(session_id=1, question_id=3, text_answer="42")
        assert r.text_answer == "42"


# ━━ ReviewSubmitRequest ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestReviewSubmitRequest:

    def test_valid(self):
        r = ReviewSubmitRequest(card_id=1, quality=4)
        assert r.quality == 4

    def test_quality_range(self):
        """Quality 0-5 should all be valid."""
        for q in range(6):
            r = ReviewSubmitRequest(card_id=1, quality=q)
            assert r.quality == q


# ━━ VisualizationData ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestVisualizationData:

    def test_plot(self):
        v = VisualizationData(
            vis_type="plot",
            data={"traces": [], "layout": {}},
        )
        assert v.vis_type == "plot"
        assert v.latex_caption is None

    def test_desmos(self):
        v = VisualizationData(
            vis_type="desmos",
            data={"url": "https://desmos.com"},
            latex_caption="y = x^2",
        )
        assert v.latex_caption == "y = x^2"


# ━━ RoadmapNodeOut ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRoadmapNodeOut:

    def test_valid(self):
        n = RoadmapNodeOut(
            skill_id="derivative_basic",
            skill_name="Đạo hàm cơ bản",
            chapter="Đạo hàm",
            p_mastery=0.8,
            level="proficient",
            prerequisites=[],
        )
        assert n.skill_id == "derivative_basic"
        assert n.prerequisites == []


# ━━ InsightOut ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestInsightOut:

    def test_valid(self):
        i = InsightOut(
            insight_type="weakness",
            message="Kỹ năng yếu nhất là đạo hàm",
        )
        assert i.insight_type == "weakness"
        assert i.data is None


# ━━ ChatMessageOut ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestChatMessageOut:

    def test_valid(self):
        m = ChatMessageOut(id=1, role="user", content="Hello")
        assert m.role == "user"
        assert m.skill_id is None


# ━━ ConversationSessionOut ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestConversationSessionOut:

    def test_valid(self):
        s = ConversationSessionOut(id=1)
        assert s.is_active is True
        assert s.message_count == 0
        assert s.title is None


# ━━ DiagnosticSchemas ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDiagnosticSchemas:

    def test_answer_request(self):
        r = DiagnosticAnswerRequest(session_id=1, question_id=5, selected_index=2)
        assert r.selected_index == 2

    def test_result_response(self):
        r = DiagnosticResultResponse(
            session_id=1,
            skill_profile=[],
            weak_areas=[],
            overall_score=75.5,
        )
        assert r.overall_score == 75.5
