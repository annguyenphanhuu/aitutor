"""SQLAlchemy ORM models."""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, UniqueConstraint
from app.db.database import Base
from app.utils.time_utils import utcnow


# ── User ─────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), nullable=False, unique=True)
    display_name = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=utcnow)
    last_login = Column(DateTime, default=utcnow)


class SkillMastery(Base):
    __tablename__ = "skill_masteries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, default=1)
    skill_id = Column(String(50), nullable=False)
    skill_name = Column(String(200), nullable=False)
    p_mastery = Column(Float, default=0.1)  # BKT mastery probability
    total_attempts = Column(Integer, default=0)
    correct_attempts = Column(Integer, default=0)
    last_updated = Column(DateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "skill_id", name="uq_user_skill"),
    )


class InteractionLog(Base):
    __tablename__ = "interaction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, default=1)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    agent_response = Column(Text, nullable=True)
    skill_id = Column(String(50), nullable=True)
    skill_ids = Column(JSON, nullable=True)       # list of assessed or targeted skills
    formula_ids = Column(JSON, nullable=True)     # list of formula registry ids used
    is_correct = Column(Boolean, nullable=True)
    error_type = Column(String(50), nullable=True)  # calculation, conceptual, procedural
    response_mode = Column(String(20), default="socratic")  # socratic, exam
    timestamp = Column(DateTime, default=utcnow)


class StudyPlan(Base):
    __tablename__ = "study_plans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, default=1)
    recommendations = Column(JSON, nullable=False)  # list of skill recommendations
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)


# ── Conversation Memory ─────────────────────────────
class ConversationSession(Base):
    __tablename__ = "conversation_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, default=1)
    title = Column(String(200), nullable=True)       # auto-generated from first message
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)
    last_active = Column(DateTime, default=utcnow, onupdate=utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, nullable=False, index=True)  # FK to conversation_sessions.id
    role = Column(String(20), nullable=False)         # "user" | "assistant"
    content = Column(Text, nullable=False)
    skill_id = Column(String(50), nullable=True)
    skill_ids = Column(JSON, nullable=True)
    formula_ids = Column(JSON, nullable=True)
    mode_used = Column(String(20), nullable=True)
    timestamp = Column(DateTime, default=utcnow)


class ChatVisualization(Base):
    """Rich visualization payload attached to an assistant chat message.

    Kept in a separate table so ``create_all()`` can add it to an existing
    deployment without an ``ALTER TABLE chat_messages`` migration.
    """

    __tablename__ = "chat_visualizations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, nullable=False, index=True)
    message_id = Column(Integer, nullable=False, unique=True, index=True)
    payload = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=utcnow)


class SessionState(Base):
    """Trạng thái hội thoại tường minh cho routing (LangGraph engine).

    Lưu đề bài đang giải / cờ chờ-trả-lời thay cho việc tái dựng state
    bằng cách quét text các lượt chat cũ. Bảng riêng (không thêm cột vào
    conversation_sessions) để ``create_all()`` áp dụng được cho deployment
    hiện có mà không cần ALTER TABLE — cùng lý do với ChatVisualization.
    """

    __tablename__ = "session_states"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, nullable=False, unique=True, index=True)  # FK to conversation_sessions.id
    current_problem = Column(Text, nullable=True)
    awaiting_answer = Column(Boolean, default=False)
    last_skill_ids = Column(JSON, nullable=True)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


# ── Quiz & Diagnostic ───────────────────────────────
class QuizQuestion(Base):
    __tablename__ = "quiz_questions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    skill_id = Column(String(50), nullable=False)
    skill_ids = Column(JSON, nullable=True)           # list of required skills
    formula_ids = Column(JSON, nullable=True)         # list of formula registry ids
    difficulty = Column(Integer, default=1)           # 1-easy, 2-medium, 3-hard
    question_type = Column(String(20), default="mcq") # mcq | true_false | short_answer
    question_latex = Column(Text, nullable=False)      # LaTeX formatted question
    # MCQ fields
    choices = Column(JSON, nullable=True)              # list of 4 answer strings (MCQ only)
    correct_index = Column(Integer, nullable=True)     # 0-3 (MCQ only)
    # True/False fields
    statements = Column(JSON, nullable=True)           # [{text, correct: bool}, ...] (4 statements)
    # Short-answer fields
    correct_answer = Column(Text, nullable=True)       # numeric/text answer
    # Common
    points = Column(Float, default=0.25)               # max points for this question
    explanation = Column(Text, nullable=True)           # Step-by-step solution
    sympy_expr = Column(Text, nullable=True)            # Original SymPy expression
    created_at = Column(DateTime, default=utcnow)


class QuestionBank(Base):
    __tablename__ = "question_bank"

    id = Column(Integer, primary_key=True, autoincrement=True)
    skill_id = Column(String(50), nullable=False)
    cognitive_level = Column(String(50), nullable=True) # nhan_biet, thong_hieu, van_dung, van_dung_cao
    difficulty = Column(Integer, default=1)           # 1-easy, 2-medium, 3-hard
    question_type = Column(String(20), default="mcq") # mcq | true_false | short_answer
    question_latex = Column(Text, nullable=False)
    choices = Column(JSON, nullable=True)
    correct_index = Column(Integer, nullable=True)
    statements = Column(JSON, nullable=True)
    correct_answer = Column(Text, nullable=True)
    points = Column(Float, default=0.25)
    explanation = Column(Text, nullable=True)
    source = Column(String(200), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)


class QuizSession(Base):
    __tablename__ = "quiz_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, default=1)
    session_type = Column(String(20), default="practice")  # practice, diagnostic, exam
    exam_format = Column(Boolean, default=False)            # True = THPT QG format
    skill_ids = Column(JSON, nullable=True)                 # list of targeted skill_ids
    questions = Column(JSON, nullable=False)                # [{question_id, answered, points_earned, ...}]
    total_questions = Column(Integer, default=0)
    correct_count = Column(Integer, default=0)
    total_score = Column(Float, default=0.0)                # accumulated score
    max_score = Column(Float, default=10.0)                # max possible score
    current_index = Column(Integer, default=0)
    is_completed = Column(Boolean, default=False)
    started_at = Column(DateTime, default=utcnow)
    completed_at = Column(DateTime, nullable=True)


# ── Spaced Repetition (SM-2) ────────────────────────
class SpacedRepetitionCard(Base):
    __tablename__ = "spaced_repetition_cards"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, default=1)
    skill_id = Column(String(50), nullable=False)
    easiness_factor = Column(Float, default=2.5)       # SM-2 EF (≥ 1.3)
    interval = Column(Integer, default=0)               # days until next review
    repetitions = Column(Integer, default=0)            # consecutive correct reviews
    next_review = Column(DateTime, default=utcnow)
    last_reviewed = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "skill_id", name="uq_user_sr_skill"),
    )


class SpacedRepetitionPendingQuestion(Base):
    """Câu hỏi ôn tập đang chờ trả lời của một SR card.

    Cho phép ``GET /review/due`` tái sử dụng câu hỏi đã sinh thay vì
    INSERT một QuizQuestion mới mỗi lần gọi. Bảng riêng (không thêm cột
    vào spaced_repetition_cards) để ``create_all()`` áp dụng được cho
    deployment hiện có mà không cần ALTER TABLE — cùng lý do với
    ChatVisualization.
    """

    __tablename__ = "sr_pending_questions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    card_id = Column(Integer, nullable=False, unique=True, index=True)  # FK to spaced_repetition_cards.id
    question_id = Column(Integer, nullable=False)  # FK to quiz_questions.id
    created_at = Column(DateTime, default=utcnow)
