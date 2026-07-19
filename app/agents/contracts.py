"""Pydantic contracts cho I/O giữa các agent trong LangGraph pipeline.

Mọi agent trả về structured output theo các model dưới đây (qua
``with_structured_output(..., method="function_calling")``), thay cho
kiểu "CHỈ TRẢ VỀ JSON" + parse chuỗi. Mỗi kết quả mang ``confidence``
và (với grading) đường abstain tường minh — nền tảng cho policy layer
quyết định có ghi BKT hay không.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

IntentName = Literal[
    "explain", "answer", "assess", "plan", "quiz",
    "review", "diagnostic", "visualize", "off_topic",
]


class IntentClassification(BaseModel):
    """Kết quả phân loại ý định của classifier node."""

    intent: IntentName = "explain"
    skill_id: Optional[str] = None
    skill_ids: list[str] = Field(default_factory=list)
    formula_ids: list[str] = Field(default_factory=list)
    # true CHỈ KHI tin nhắn chứa lời giải/đáp án của CHÍNH học sinh
    is_answer_submission: bool = False
    # true khi tin nhắn chứa một đề bài mới (điều khiển SessionState.awaiting_answer)
    contains_problem: bool = False
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    def resolved_skill_ids(self) -> list[str]:
        """skill_ids đã chuẩn hóa (fallback về skill_id đơn nếu list rỗng)."""
        if self.skill_ids:
            return self.skill_ids
        return [self.skill_id] if self.skill_id else []


AbstainReason = Literal["third_party_claim", "no_final_answer", "no_problem_found", "other"]

TaskKind = Literal["antiderivative", "derivative", "equation", "numeric", "other"]


class GradingExtraction(BaseModel):
    """Kết quả trích xuất của grade_extract node.

    ``gradable=False`` là đường abstain: tin nhắn không chứa bài làm
    của chính học sinh (ví dụ hỏi xác minh mệnh đề của thầy/bạn) —
    pipeline chuyển sang teach, KHÔNG chấm, KHÔNG ghi BKT.
    """

    gradable: bool = False
    abstain_reason: Optional[AbstainReason] = None
    problem_statement: Optional[str] = None
    # Kết luận cuối cùng CỦA HỌC SINH (không phải của thầy/bạn/đề bài)
    student_final_answer: Optional[str] = None
    task_kind: TaskKind = "other"
    # Biểu thức dạng sympy-parseable, ví dụ integrand "x**2"
    problem_expr: Optional[str] = None
    # Đáp án học sinh dạng sympy-parseable, ví dụ "x**3/2"
    candidate_expr: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class VerifierResult(BaseModel):
    """Kết quả kiểm chứng deterministic bằng SymPy (không LLM)."""

    verified: bool = False          # False = CAS không kiểm chứng được
    verdict: Literal["correct", "incorrect", "unknown"] = "unknown"
    method: str = ""
    details: str = ""


class SkillVerdict(BaseModel):
    passed: bool = False
    feedback: str = ""


class PedagogyAssessment(BaseModel):
    """Nhận xét sư phạm của grade_feedback node (thay dict thô của Assessor)."""

    is_correct: bool = False
    score: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    error_type: Literal["none", "calculation", "conceptual", "procedural", "unknown"] = "unknown"
    skills_assessed: dict[str, SkillVerdict] = Field(default_factory=dict)
    feedback: str = ""
    overall_feedback: str = ""
    correct_solution: str = ""


class MasteryDecision(BaseModel):
    """Quyết định của policy layer — nơi DUY NHẤT được phép ghi BKT."""

    should_write: bool = False
    skills_assessed: dict[str, bool] = Field(default_factory=dict)
    reason: str = ""
    # LLM và CAS mâu thuẫn nhau (CAS thắng) — log lên Langfuse để theo dõi
    mismatch: bool = False
