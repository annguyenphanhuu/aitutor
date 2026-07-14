"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Chat ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    mode: str = "auto"  # auto, socratic, exam
    session_id: Optional[int] = None  # None = create new session


class ChatResponse(BaseModel):
    response: str
    skill_id: Optional[str] = None
    skill_ids: list[str] = Field(default_factory=list)
    skill_name: Optional[str] = None
    mastery_level: Optional[float] = None
    mode_used: str = "socratic"
    visualization: Optional[dict] = None  # Plotly/Desmos data if present
    session_id: Optional[int] = None      # conversation session ID
    formula_ids: list[str] = Field(default_factory=list)


# ── Assessment ───────────────────────────────────────────
class AssessmentRequest(BaseModel):
    question: str
    student_answer: str


class AssessmentResponse(BaseModel):
    is_correct: bool
    score: float
    error_type: Optional[str] = None
    feedback: str
    correct_solution: str
    skill_id: Optional[str] = None
    skill_ids: list[str] = Field(default_factory=list)
    skills_assessed: dict = Field(default_factory=dict)
    formula_ids: list[str] = Field(default_factory=list)
    new_mastery: Optional[float] = None


# ── Study Plan ───────────────────────────────────────────
class StudyPlanResponse(BaseModel):
    recommendations: list
    created_at: datetime
    is_active: bool

    class Config:
        from_attributes = True


# ── Skill Mastery ────────────────────────────────────────
class SkillMasteryResponse(BaseModel):
    skill_id: str
    skill_name: str
    p_mastery: float
    total_attempts: int
    correct_attempts: int

    class Config:
        from_attributes = True


# ── Dashboard ────────────────────────────────────────────
class DashboardResponse(BaseModel):
    skills: list[SkillMasteryResponse]
    recent_interactions: list[dict]
    study_plan: Optional[StudyPlanResponse] = None
    overall_mastery: float


# ── Quiz ─────────────────────────────────────────────
class QuizGenerateRequest(BaseModel):
    skill_id: Optional[str] = None
    skill_ids: Optional[list[str]] = None
    chapter: Optional[str] = None
    formula_ids: list[str] = Field(default_factory=list)
    difficulty: int = 1         # 1-easy, 2-medium, 3-hard
    count: int = 5              # number of questions


class QuizQuestionOut(BaseModel):
    id: int
    question_type: str = "mcq"  # mcq | true_false | short_answer
    question_latex: str
    choices: Optional[list[str]] = None   # MCQ only
    statements: Optional[list[dict]] = None  # True/False only: [{text, ...}]
    skill_id: str
    skill_ids: list[str] = Field(default_factory=list)
    formula_ids: list[str] = Field(default_factory=list)
    difficulty: int
    points: float = 0.25

    class Config:
        from_attributes = True


class QuizSessionOut(BaseModel):
    session_id: int
    questions: list[QuizQuestionOut]
    total_questions: int
    session_type: str
    max_score: float = 10.0


class QuizAnswerRequest(BaseModel):
    session_id: int
    question_id: int
    # MCQ
    selected_index: Optional[int] = None      # 0-3 for MCQ
    # True/False
    tf_answers: Optional[list[bool]] = None   # [True, False, True, False] for 4 statements
    # Short answer
    text_answer: Optional[str] = None         # numeric/text answer


class QuizAnswerResponse(BaseModel):
    is_correct: bool               # fully correct?
    points_earned: float           # 0.1 / 0.25 / 0.5 / 1.0 for T/F, or 0.25 for MCQ
    max_points: float              # max possible for this question
    correct_index: Optional[int] = None
    correct_statements: Optional[list[bool]] = None  # T/F correct answers
    correct_answer: Optional[str] = None              # short answer
    explanation: Optional[str] = None
    new_mastery: Optional[float] = None
    session_progress: dict         # {current, total, score_so_far, max_score}


class QuizResultResponse(BaseModel):
    session_id: int
    total_questions: int
    total_score: float             # accumulated points
    max_score: float               # max possible
    score_percent: float
    part_scores: Optional[dict] = None  # {mcq: x, true_false: y, short_answer: z}
    skill_results: list[dict]   # [{skill_id, skill_name, correct, total, mastery}]


# ── Diagnostic ───────────────────────────────────────────
class DiagnosticStartResponse(BaseModel):
    session_id: int
    first_question: QuizQuestionOut
    total_questions: int


class DiagnosticAnswerRequest(BaseModel):
    session_id: int
    question_id: int
    selected_index: int


class DiagnosticAnswerResponse(BaseModel):
    is_correct: bool
    correct_index: int
    explanation: Optional[str] = None
    next_question: Optional[QuizQuestionOut] = None
    is_completed: bool
    progress: dict              # {current, total}


class DiagnosticResultResponse(BaseModel):
    session_id: int
    skill_profile: list[dict]   # [{skill_id, skill_name, estimated_mastery, level}]
    weak_areas: list[dict]      # [{skill_id, skill_name, recommendation}]
    overall_score: float


# ── Spaced Repetition ────────────────────────────────────
class ReviewCardOut(BaseModel):
    card_id: int
    skill_id: str
    skill_name: str
    question: QuizQuestionOut
    days_overdue: int

    class Config:
        from_attributes = True


class ReviewDueResponse(BaseModel):
    due_count: int
    cards: list[ReviewCardOut]


class ReviewSubmitRequest(BaseModel):
    card_id: int
    question_id: Optional[int] = None
    selected_index: Optional[int] = None
    quality: Optional[int] = None


# ── Exam Practice ─────────────────────────────────────────
class ExamGradeRequest(BaseModel):
    exam_id: str
    answers: dict  # { "question_id": { "mcq_answer": "B" | "tf_answers": {...} | "sa_answer": "42" } }


class ReviewSubmitResponse(BaseModel):
    next_review_days: int
    new_interval: int
    message: str
    is_correct: bool
    correct_index: int
    explanation: Optional[str] = None


# ── Visualization ────────────────────────────────────────
class VisualizationData(BaseModel):
    vis_type: str               # "plot", "desmos", "geometry_3d"
    data: dict                  # Plotly trace data or Desmos config
    latex_caption: Optional[str] = None


# ── Roadmap ──────────────────────────────────────────────
class RoadmapNodeOut(BaseModel):
    skill_id: str
    skill_name: str
    chapter: str
    p_mastery: float
    level: str                  # beginner, developing, proficient, mastered
    prerequisites: list[str]


class RoadmapResponse(BaseModel):
    nodes: list[RoadmapNodeOut]
    chapters: list[str]


# ── Insights ─────────────────────────────────────────────
class InsightOut(BaseModel):
    insight_type: str           # "comparison", "streak", "weakness"
    message: str
    data: Optional[dict] = None


# ── Conversation Memory ──────────────────────────────────
class ChatMessageOut(BaseModel):
    id: int
    role: str
    content: str
    skill_id: Optional[str] = None
    skill_ids: list[str] = Field(default_factory=list)
    formula_ids: list[str] = Field(default_factory=list)
    timestamp: Optional[datetime] = None

    class Config:
        from_attributes = True


class ConversationSessionOut(BaseModel):
    id: int
    title: Optional[str] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_active: Optional[datetime] = None
    message_count: int = 0

    class Config:
        from_attributes = True


class ConversationHistoryResponse(BaseModel):
    session: ConversationSessionOut
    messages: list[ChatMessageOut]


# ── Exam Solver ──────────────────────────────────────────────
class ExamSolverQuestionOut(BaseModel):
    question_number: str
    question_type: str          # mcq, true_false, short_answer, essay
    content: str
    skill_id: Optional[str] = None
    skill_name: Optional[str] = None
    solution: str
    error: Optional[str] = None


class ExamSolverResultOut(BaseModel):
    total_questions: int
    questions: list[ExamSolverQuestionOut] = Field(default_factory=list)
    report_markdown: str = ""
    skill_stats: dict = Field(default_factory=dict)
    raw_ocr: str = ""
    ocr_engine_used: str = "cloud"
    elapsed_seconds: float = 0.0

