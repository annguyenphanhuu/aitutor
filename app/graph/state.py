"""TutorState — state schema của tutor graph.

TypedDict (không phải Pydantic) để LangGraph merge partial updates từ các
node một cách tự nhiên. Các object contract Pydantic nằm bên trong state.

Runtime handles (db session, langfuse trace, user_id, streaming flag,
session_id) KHÔNG nằm trong state — chúng đi qua ``config["configurable"]``
để state luôn serializable (sẵn sàng gắn checkpointer sau này).
"""

from __future__ import annotations

from typing import Optional, TypedDict

from pydantic import BaseModel

from app.agents.contracts import (
    GradingExtraction,
    IntentClassification,
    MasteryDecision,
    PedagogyAssessment,
    VerifierResult,
)


class SessionSnapshot(BaseModel):
    """Bản chụp SessionState từ DB tại thời điểm hydrate."""

    current_problem: Optional[str] = None
    awaiting_answer: bool = False
    last_skill_ids: list[str] = []


class TutorState(TypedDict, total=False):
    # ── inputs ──────────────────────────────────────────────
    message: str
    requested_mode: str                     # "auto" | "socratic" | "exam"
    chat_history: list[dict]

    # ── hydrated context ────────────────────────────────────
    session_state: SessionSnapshot
    masteries: dict[str, float]

    # ── classification / routing ────────────────────────────
    classification: IntentClassification
    intent: str
    skill_id: Optional[str]
    skill_ids: list[str]
    formula_ids: list[str]
    current_mastery: float
    mastery_level: str                      # nhãn BKT: beginner/developing/...
    mode: str                               # mode đã resolve

    # ── grading pipeline ────────────────────────────────────
    extraction: Optional[GradingExtraction]
    verification: Optional[VerifierResult]
    assessment: Optional[PedagogyAssessment]
    mastery_decision: Optional[MasteryDecision]
    abstained: bool
    new_mastery: Optional[float]

    # ── outputs ─────────────────────────────────────────────
    response_text: str
    visualization: Optional[dict]
    mode_used: str
    streamed_tokens: bool                   # teach đã stream token → done frame gọn
    result: dict                            # dict cuối cùng theo contract ChatResponse
