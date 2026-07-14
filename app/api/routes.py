"""API routes for the AI Tutor application."""

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func, delete
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import json
import time

from app.db.database import get_db
from app.db.models import (
    InteractionLog, StudyPlan,
    ConversationSession, ChatMessage, ChatVisualization,
)
from app.db.schemas import (
    ChatRequest, ChatResponse,
    QuizGenerateRequest, QuizAnswerRequest,
    DiagnosticAnswerRequest,
    ReviewSubmitRequest,
    ExamGradeRequest,
)
from app.agents.orchestrator import Orchestrator
from app.agents.planner_agent import PlannerAgent
from app.knowledge_tracing.service import get_mastery_profile, get_all_masteries
from app.knowledge_tracing.skill_graph import SKILLS, get_chapters
from app.knowledge_tracing.bkt import BKTModel
from app.ocr.ocr_strategy import get_ocr_engine
from app.quiz.service import create_quiz_session, submit_quiz_answer, get_quiz_result
from app.quiz.diagnostic import start_diagnostic, answer_diagnostic, get_diagnostic_result
from app.quiz.exam_service import list_exams, load_exam, grade_exam_and_update
from app.spaced_repetition.service import get_due_cards, review_card, ensure_cards_for_attempted_skills
from app.analytics.insights import get_insights
from app.auth.service import login_or_register, get_current_user_id
from app.guardrails.input_validator import mask_pii, validate_input
from app.guardrails.rate_limiter import get_rate_limiter
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["tutor"])

orchestrator = Orchestrator()
planner = PlannerAgent()
bkt = BKTModel()

# Maximum chat history messages to load from DB
MAX_HISTORY_LOAD = 20


async def _save_chat_visualization(
    db: AsyncSession,
    message: ChatMessage,
    visualization: Optional[dict],
) -> None:
    """Persist a rich chat payload after its ChatMessage has received an id."""
    if not visualization:
        return
    await db.flush()
    db.add(ChatVisualization(
        session_id=message.session_id,
        message_id=message.id,
        payload=visualization,
    ))


async def _get_or_create_chat_session(
    db: AsyncSession,
    *,
    user_id: int,
    session_id: Optional[int],
    title: str,
) -> ConversationSession:
    """Return a user-owned session or create a new one."""
    if session_id is not None:
        result = await db.execute(
            select(ConversationSession).where(
                ConversationSession.id == session_id,
                ConversationSession.user_id == user_id,
            )
        )
        session = result.scalar_one_or_none()
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        session.last_active = datetime.utcnow()
        return session

    session = ConversationSession(user_id=user_id, title=title)
    db.add(session)
    await db.flush()
    return session


async def _load_chat_history(
    db: AsyncSession,
    session_id: int,
) -> list[dict]:
    """Load the bounded conversation history in chronological order."""
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.timestamp.asc())
        .limit(MAX_HISTORY_LOAD)
    )
    return [
        {"role": message.role, "content": message.content}
        for message in result.scalars().all()
    ]


# ── Auth ─────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str


@router.post("/auth/login")
async def login(data: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Login or register with username. Returns user info."""
    try:
        result = await login_or_register(db, data.username)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/auth/me")
async def get_me(
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Get current user info from X-User-Id header."""
    from app.db.models import User
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return {
        "user_id": user.id,
        "username": user.username,
        "display_name": user.display_name or user.username,
    }


# ── Adaptive Difficulty ──────────────────────────────────
@router.get("/quiz/adaptive")
async def get_adaptive_recommendation(
    skill_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Get adaptive difficulty recommendation for a skill."""
    from app.quiz.adaptive_engine import get_adaptive_engine
    from app.knowledge_tracing.service import get_or_create_mastery, get_recent_results, get_all_masteries
    engine = get_adaptive_engine()
    mastery_rec = await get_or_create_mastery(db, skill_id, user_id)
    recent = await get_recent_results(db, skill_id, user_id, limit=10)
    all_masteries = await get_all_masteries(db, user_id)
    summary = engine.get_adaptive_summary(skill_id, mastery_rec.p_mastery, recent, all_masteries)
    return summary


# ── Chat ─────────────────────────────────────────────────
@router.post("/chat", response_model=ChatResponse)
async def chat(
    data: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Send a question and get an adaptive response with conversation memory."""
    # ── Rate limiting ──────────────────────────────────────────────────
    allowed, reason = get_rate_limiter().is_allowed(user_id)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)

    # ── Input validation + sanitization ───────────────────────────────
    validation = validate_input(data.message, user_id=user_id)
    if not validation.is_safe:
        raise HTTPException(status_code=400, detail=validation.rejection_reason)
    # Dùng message đã sanitized (truncated nếu quá dài)
    sanitized_message = validation.message

    # ── Session management ────────────────────────────────────────────
    title = sanitized_message[:80] + ("..." if len(sanitized_message) > 80 else "")
    session = await _get_or_create_chat_session(
        db,
        user_id=user_id,
        session_id=data.session_id,
        title=title,
    )
    session_id = session.id

    # ── Load chat history ─────────────────────────────────────────
    chat_history = await _load_chat_history(db, session_id)

    # ── Save user message ─────────────────────────────────────────
    user_msg = ChatMessage(
        session_id=session_id,
        role="user",
        content=sanitized_message,
    )
    db.add(user_msg)

    # ── Process through orchestrator with Langfuse tracing ────────────
    from app.utils.langfuse_client import new_trace, update_trace
    trace = new_trace(
        name="chat",
        user_id=str(user_id),
        session_id=str(session_id),
        metadata={"mode": data.mode},
        input_text=mask_pii(sanitized_message),
    )

    t0 = time.monotonic()
    response = await orchestrator.handle_message(
        db=db,
        message=sanitized_message,
        mode=data.mode,
        chat_history=chat_history,
        user_id=user_id,
        langfuse_trace=trace,   # truyền trace xuống để các span con gắn vào
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    update_trace(
        trace,
        output=response.get("response", ""),
        metadata={
            "skill_id": response.get("skill_id"),
            "skill_ids": response.get("skill_ids"),
            "mode_used": response.get("mode_used"),
            "latency_ms": latency_ms,
            "intent": response.get("intent"),
        },
    )

    # ── Save assistant response ───────────────────────────────────────────
    assistant_msg = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=response["response"],
        skill_id=response.get("skill_id"),
        mode_used=response.get("mode_used"),
    )
    db.add(assistant_msg)
    await _save_chat_visualization(db, assistant_msg, response.get("visualization"))

    # Log the interaction (backward compatible)
    log = InteractionLog(
        user_id=user_id,
        question=mask_pii(sanitized_message),
        agent_response=response["response"],
        skill_id=response.get("skill_id"),
        response_mode=response.get("mode_used", "socratic"),
    )
    db.add(log)

    # Auto-create SR card for attempted skills
    if response.get("skill_id"):
        await ensure_cards_for_attempted_skills(db, response["skill_id"], user_id)

    response["session_id"] = session_id
    return ChatResponse(**response)


# ── Chat Stream (SSE) ─────────────────────────────────────────────────
@router.post("/chat/stream")
async def chat_stream(
    data: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Streaming chat endpoint dùng Server-Sent Events (SSE).

    Frontend kết nối bằng fetch() + ReadableStream (không dùng EventSource vì
    EventSource chỉ hỗ trợ GET và không gửi headers custom).

    Mỗi SSE frame:
      data: {"type": "meta"|"token"|"done"|"error", ...}\n\n
    """
    # ── Guardrails ──────────────────────────────────────────────────
    allowed, reason = get_rate_limiter().is_allowed(user_id)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)

    validation = validate_input(data.message, user_id=user_id)
    if not validation.is_safe:
        raise HTTPException(status_code=400, detail=validation.rejection_reason)
    sanitized_message = validation.message

    # ── Session management ─────────────────────────────────────────
    title = sanitized_message[:80] + ("..." if len(sanitized_message) > 80 else "")
    session = await _get_or_create_chat_session(
        db,
        user_id=user_id,
        session_id=data.session_id,
        title=title,
    )
    session_id = session.id

    # ── Load chat history ──────────────────────────────────────────
    chat_history = await _load_chat_history(db, session_id)

    # ── Save user message ngay (trước khi stream bắt đầu) ──────────────
    db.add(ChatMessage(session_id=session_id, role="user", content=sanitized_message))
    await db.commit()

    captured_session_id = session_id

    # ── SSE event generator ─────────────────────────────────────────
    async def event_generator():
        full_response = ""
        skill_id_cap = None
        mode_used_cap = None
        visualization_cap = None

        try:
            async for frame_json in orchestrator.handle_message_stream(
                db=db,
                message=sanitized_message,
                mode=data.mode,
                chat_history=chat_history,
                user_id=user_id,
            ):
                try:
                    frame = json.loads(frame_json)
                    ftype = frame.get("type")
                    if ftype == "meta":
                        skill_id_cap = frame.get("skill_id")
                        mode_used_cap = frame.get("mode_used")
                    elif ftype == "done":
                        full_response = frame.get("full_response", "")
                        # Final metadata is authoritative for non-streamable
                        # intents (e.g. visualize), whose mode can differ from
                        # the preliminary meta frame.
                        skill_id_cap = frame.get("skill_id", skill_id_cap)
                        mode_used_cap = frame.get("mode_used", mode_used_cap)
                        visualization_cap = frame.get("visualization")
                except Exception:
                    pass

                yield f"data: {frame_json}\n\n"

        except Exception as e:
            err = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
            yield f"data: {err}\n\n"
            return

        # ── Lưu assistant message + log sau khi stream xong ────────────
        if full_response:
            from app.db.database import async_session
            async with async_session() as save_db:
                async with save_db.begin():
                    assistant_message = ChatMessage(
                        session_id=captured_session_id,
                        role="assistant",
                        content=full_response,
                        skill_id=skill_id_cap,
                        mode_used=mode_used_cap,
                    )
                    save_db.add(assistant_message)
                    await _save_chat_visualization(
                        save_db, assistant_message, visualization_cap,
                    )
                    save_db.add(InteractionLog(
                        user_id=user_id,
                        question=mask_pii(sanitized_message),
                        agent_response=full_response,
                        skill_id=skill_id_cap,
                        response_mode=mode_used_cap or "stream",
                    ))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "X-Session-Id": str(session_id),
        },
    )


# ── Image Upload — Multi-Image Hybrid Vision (OCR + Vision) ──
@router.post("/upload")
async def upload_image(
    files: List[UploadFile] = File(...),
    user_text: str = Form(default=""),
    session_id: Optional[int] = Form(default=None),
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """
    Upload 1 hoặc nhiều ảnh → Multi-Image Hybrid Vision flow:
    1. OCR song song tất cả ảnh bằng GPT-4o-mini
    2. Gộp tất cả OCR text với label [Ảnh 1], [Ảnh 2]...
    3. Gọi orchestrator.handle_image_message() với ảnh đầu + OCR gộp + user_text
    4. Lưu vào session DB (tạo mới nếu chưa có)
    """
    if not files:
        raise HTTPException(status_code=400, detail="Không có file nào được gửi")

    # ── Rate limiting ──────────────────────────────────────────────────
    allowed, reason = get_rate_limiter().is_allowed(user_id)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)

    # ── Validate user_text nếu có ────────────────────────────────────
    if user_text:
        validation = validate_input(user_text, user_id=user_id)
        if not validation.is_safe:
            raise HTTPException(status_code=400, detail=validation.rejection_reason)
        user_text = validation.message

    # Validate tất cả file là ảnh
    for f in files:
        if not f.content_type or not f.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"File '{f.filename}' không phải ảnh")

    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Tối đa 5 ảnh mỗi lần gửi")

    # ── Đọc tất cả file ───────────────────────────────────────────
    images: list[tuple[bytes, str]] = []
    for f in files:
        data = await f.read()
        mime = f.content_type or "image/jpeg"
        images.append((data, mime))

    # ── Step 1: OCR song song tất cả ảnh ─────────────────────────
    ocr_engine = get_ocr_engine()
    ocr_results: list[str] = await asyncio.gather(
        *[ocr_engine.ocr_image(b, mime_type=m) for b, m in images]
    )

    # Gộp OCR results với label "Ảnh 1/2/..."
    if len(ocr_results) == 1:
        ocr_text = ocr_results[0]
    else:
        parts = []
        for i, txt in enumerate(ocr_results, 1):
            parts.append(f"[Ảnh {i}]\n{txt}")
        ocr_text = "\n\n".join(parts)

    # ── Step 2: Session management ────────────────────────────────
    title = user_text[:80] if user_text else (ocr_text[:60] + "…")
    session = await _get_or_create_chat_session(
        db,
        user_id=user_id,
        session_id=session_id,
        title=f"📸 {title}",
    )
    sid = session.id

    # ── Step 3: Load chat history ─────────────────────────────────
    chat_history = await _load_chat_history(db, sid)

    # ── Step 4: Gọi orchestrator với ảnh đầu + OCR gộp ───────────
    # Nếu nhiều ảnh, gửi ảnh đầu + OCR gộp (LLM sẽ có full context)
    primary_bytes, primary_mime = images[0]
    response = await orchestrator.handle_image_message(
        db=db,
        image_bytes=primary_bytes,
        image_mime=primary_mime,
        ocr_text=ocr_text,
        user_text=user_text,
        mode="auto",
        chat_history=chat_history,
        user_id=user_id,
    )

    # ── Step 5: Lưu vào DB ────────────────────────────────────────
    n = len(images)
    user_content = f"[Ảnh đính kèm: {n} ảnh]" if n > 1 else "[Ảnh đính kèm]"
    if user_text:
        user_content += f"\n{user_text}"
    db.add(ChatMessage(session_id=sid, role="user", content=user_content))

    assistant_message = ChatMessage(
        session_id=sid,
        role="assistant",
        content=response["response"],
        skill_id=response.get("skill_id"),
        mode_used=response.get("mode_used"),
    )
    db.add(assistant_message)
    await _save_chat_visualization(
        db, assistant_message, response.get("visualization"),
    )

    db.add(InteractionLog(
        user_id=user_id,
        question=mask_pii(user_content),
        agent_response=response["response"],
        skill_id=response.get("skill_id"),
        response_mode=response.get("mode_used", "hybrid_vision"),
    ))

    response["ocr_text"] = ocr_text
    response["image_count"] = n
    response["session_id"] = sid
    return response


# ── Quiz ─────────────────────────────────────────────────
@router.post("/quiz/generate")
async def generate_quiz(
    data: QuizGenerateRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Generate quiz questions for a specific skill."""
    result = await create_quiz_session(
        db, 
        skill_id=data.skill_id, 
        difficulty=data.difficulty, 
        count=data.count,
        user_id=user_id,
        chapter=data.chapter,
        exam_format=data.exam_format,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/quiz/submit")
async def submit_answer(
    data: QuizAnswerRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Submit an answer for a quiz question."""
    result = await submit_quiz_answer(
        db, data.session_id, data.question_id,
        selected_index=data.selected_index,
        tf_answers=data.tf_answers,
        text_answer=data.text_answer,
        user_id=user_id,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # Auto-create SR card
    from app.db.models import QuizQuestion
    q_result = await db.execute(select(QuizQuestion).where(QuizQuestion.id == data.question_id))
    q = q_result.scalar_one_or_none()
    if q and q.skill_id:
        await ensure_cards_for_attempted_skills(db, q.skill_id, user_id)

    return result


@router.get("/quiz/result/{session_id}")
async def quiz_result(session_id: int, db: AsyncSession = Depends(get_db)):
    """Get the results of a completed quiz session."""
    result = await get_quiz_result(db, session_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ── Diagnostic ───────────────────────────────────────────
@router.post("/diagnostic/start")
async def diagnostic_start(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Start a diagnostic assessment."""
    result = await start_diagnostic(db, user_id)
    return result


@router.post("/diagnostic/answer")
async def diagnostic_answer(
    data: DiagnosticAnswerRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Answer a diagnostic question and get the next one."""
    result = await answer_diagnostic(db, data.session_id, data.question_id, data.selected_index, user_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/diagnostic/result/{session_id}")
async def diagnostic_result(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Get diagnostic results with skill profile."""
    result = await get_diagnostic_result(db, session_id, user_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ── Spaced Repetition ────────────────────────────────────
@router.get("/review/due")
async def review_due(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Get cards due for review today."""
    due = await get_due_cards(db, user_id)
    return {"due_count": len(due), "cards": due}


@router.post("/review/submit")
async def review_submit(data: ReviewSubmitRequest, db: AsyncSession = Depends(get_db)):
    """Submit a review result for a spaced repetition card."""
    result = await review_card(db, data.card_id, data.question_id, data.selected_index)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ── Dashboard & Profile ──────────────────────────────────
@router.get("/dashboard")
async def get_dashboard(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Get dashboard data."""
    # Get mastery profile
    profile = await get_mastery_profile(db, user_id)

    # Get recent interactions
    result = await db.execute(
        select(InteractionLog)
        .where(InteractionLog.user_id == user_id)
        .order_by(desc(InteractionLog.timestamp))
        .limit(20)
    )
    interactions = result.scalars().all()

    # Get active study plan
    result = await db.execute(
        select(StudyPlan)
        .where(StudyPlan.user_id == user_id, StudyPlan.is_active.is_(True))
        .order_by(desc(StudyPlan.created_at))
        .limit(1)
    )
    plan = result.scalar_one_or_none()

    # Calculate overall mastery
    overall = sum(s["p_mastery"] for s in profile) / len(profile) if profile else 0

    return {
        "skills": profile,
        "recent_interactions": [
            {
                "question": i.question[:100],
                "skill_id": i.skill_id,
                "is_correct": i.is_correct,
                "response_mode": i.response_mode,
                "timestamp": i.timestamp.isoformat() if i.timestamp else None,
            }
            for i in interactions
        ],
        "study_plan": {
            "recommendations": plan.recommendations,
            "created_at": plan.created_at.isoformat(),
            "is_active": plan.is_active,
        } if plan else None,
        "overall_mastery": round(overall, 3),
    }


@router.get("/profile")
async def get_profile(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Get mastery profile."""
    profile = await get_mastery_profile(db, user_id)
    return {"skills": profile}


@router.post("/plan")
async def generate_plan(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Generate a new study plan."""
    profile = await get_mastery_profile(db, user_id)
    plan_data = await planner.create_plan(profile)

    # Deactivate old plans
    result = await db.execute(
        select(StudyPlan).where(StudyPlan.user_id == user_id, StudyPlan.is_active.is_(True))
    )
    for old_plan in result.scalars().all():
        old_plan.is_active = False

    # Save new plan
    plan = StudyPlan(
        user_id=user_id,
        recommendations=plan_data,
        is_active=True,
    )
    db.add(plan)

    return plan_data


# ── Roadmap (Knowledge Graph + Mastery) ──────────────────
@router.get("/roadmap")
async def get_roadmap(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Get skill graph with mastery overlay for dashboard visualization."""
    masteries = await get_all_masteries(db, user_id)

    nodes = []
    for skill_id, info in SKILLS.items():
        p = masteries.get(skill_id, 0.0)
        level = bkt.get_mastery_level(p)
        nodes.append({
            "skill_id": skill_id,
            "skill_name": info["name"],
            "chapter": info["chapter"],
            "description": info.get("description", ""),
            "p_mastery": round(p, 3),
            "level": level,
            "prerequisites": info["prerequisites"],
        })

    return {
        "nodes": nodes,
        "chapters": get_chapters(),
    }


# ── Insights ─────────────────────────────────────────────
@router.get("/insights")
async def get_peer_insights(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Get insights and analytics."""
    insights = await get_insights(db, user_id)
    return {"insights": insights}


# ── Curriculum Info ──────────────────────────────────────
@router.get("/skills")
async def list_skills():
    """List all skills in the curriculum."""
    return {
        "skills": [
            {"id": k, "name": v["name"], "chapter": v["chapter"], "description": v["description"]}
            for k, v in SKILLS.items()
        ],
        "chapters": get_chapters(),
    }


# ── Conversation Sessions ──────────────────────────────────
@router.post("/session/new")
async def create_session(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Create a new conversation session."""
    session = ConversationSession(user_id=user_id)
    db.add(session)
    await db.flush()
    return {"session_id": session.id}


@router.get("/sessions")
async def list_sessions(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """List recent conversation sessions."""
    result = await db.execute(
        select(ConversationSession)
        .where(ConversationSession.user_id == user_id)
        .order_by(desc(ConversationSession.last_active))
        .limit(20)
    )
    sessions = result.scalars().all()

    out = []
    for s in sessions:
        count_result = await db.execute(
            select(func.count(ChatMessage.id))
            .where(ChatMessage.session_id == s.id)
        )
        msg_count = count_result.scalar() or 0
        out.append({
            "id": s.id,
            "title": s.title,
            "is_active": s.is_active,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "last_active": s.last_active.isoformat() if s.last_active else None,
            "message_count": msg_count,
        })

    return {"sessions": out}


@router.get("/session/{session_id}/history")
async def get_session_history(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Get full chat history for a session."""
    result = await db.execute(
        select(ConversationSession).where(
            ConversationSession.id == session_id,
            ConversationSession.user_id == user_id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    msg_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.timestamp.asc())
    )
    messages = msg_result.scalars().all()

    visualization_result = await db.execute(
        select(ChatVisualization)
        .where(ChatVisualization.session_id == session_id)
    )
    visualizations = {
        item.message_id: item.payload
        for item in visualization_result.scalars().all()
    }

    return {
        "session": {
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at.isoformat() if session.created_at else None,
        },
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "skill_id": m.skill_id,
                "visualization": visualizations.get(m.id),
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
            }
            for m in messages
        ],
    }


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Delete a specific chat session and its messages."""
    result = await db.execute(
        select(ConversationSession).where(
            ConversationSession.id == session_id,
            ConversationSession.user_id == user_id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await db.execute(delete(ChatVisualization).where(ChatVisualization.session_id == session_id))
    await db.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
    await db.execute(delete(ConversationSession).where(ConversationSession.id == session_id))
    await db.commit()
    return {"status": "ok"}


@router.delete("/sessions/all")
async def clear_all_sessions(db: AsyncSession = Depends(get_db), user_id: int = Depends(get_current_user_id)):
    """Delete all chat sessions and messages for the current user."""
    # Find all sessions for user
    result = await db.execute(select(ConversationSession.id).where(ConversationSession.user_id == user_id))
    session_ids = [row[0] for row in result.all()]

    if session_ids:
        await db.execute(delete(ChatVisualization).where(ChatVisualization.session_id.in_(session_ids)))
        await db.execute(delete(ChatMessage).where(ChatMessage.session_id.in_(session_ids)))
        await db.execute(delete(ConversationSession).where(ConversationSession.user_id == user_id))
        await db.commit()
    return {"status": "ok"}


# ── Exam Practice (pre-built exam sets) ──────────────────
@router.get("/exams")
async def get_exams():
    """List available exam sets from data/exams/."""
    exams = list_exams()
    return {"exams": exams}


@router.get("/exams/{exam_id}")
async def get_exam(exam_id: str):
    """Load an exam for test-taking (answers hidden)."""
    result = load_exam(exam_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy đề thi")
    return result


@router.post("/exams/grade")
async def grade_exam_endpoint(
    data: ExamGradeRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Grade a completed exam and update BKT mastery, logs, and SR cards."""
    if not data.exam_id:
        raise HTTPException(status_code=400, detail="Thiếu exam_id")
    result = await grade_exam_and_update(db, data.exam_id, data.answers, user_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy đề thi")
    await db.commit()
    return result


# ── Evaluation / RAGAS ───────────────────────────────────

@router.get("/evaluation/hallucination-report")
async def hallucination_report():
    """
    Trả về hallucination metrics từ ReflectionEngine runtime.

    Metrics:
    - correction_rate  : % lần LLM cần self-correct (mục tiêu < 10%)
    - sympy_error_rate : % phép tính SymPy verify thất bại (mục tiêu < 5%)
    - per_skill        : breakdown theo skill_id
    """
    from app.evaluation.hallucination_tracker import ReflectionMetrics
    return ReflectionMetrics.report()


@router.get("/evaluation/ragas-report")
async def ragas_report():
    """
    Trả về kết quả RAGAS evaluation cuối cùng (nếu đã chạy scripts/run_evaluation.py).
    File: data/ragas_report.json
    """
    report_path = Path("data/ragas_report.json")
    if not report_path.exists():
        return {
            "status": "not_found",
            "message": (
                "Chưa có báo cáo RAGAS. Chạy: "
                "python scripts/run_evaluation.py --build --limit 20"
            ),
        }
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["status"] = "ok"
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đọc report: {e}")


@router.post("/evaluation/build-dataset")
async def build_eval_dataset(limit: int = 20):
    """
    Trigger xây dựng RAGAS dataset từ exam JSON (background task).
    Trả về ngay lập tức, dataset được lưu vào data/ragas_dataset.json.
    """
    import asyncio
    from app.evaluation.dataset_builder import (
        load_exam_questions,
    )

    questions = load_exam_questions(exam_dir="data/exams", limit=limit)
    if not questions:
        raise HTTPException(status_code=400, detail="Không tìm thấy câu hỏi exam")

    # Chạy trong background
    asyncio.create_task(_build_dataset_task(questions))

    return {
        "status": "building",
        "n_questions": len(questions),
        "output": "data/ragas_dataset.json",
        "message": "Dataset đang được build. Kiểm tra lại sau vài phút.",
    }


async def _build_dataset_task(questions: list[dict]):
    """Background task: build và lưu RAGAS dataset."""
    from app.evaluation.dataset_builder import build_ragas_samples, save_samples
    samples = await build_ragas_samples(questions, k=5)
    save_samples(samples, "data/ragas_dataset.json")


# ── Guardrails Admin ─────────────────────────────────────────────────────────

@router.get("/guardrails/rate-limit-stats")
async def rate_limit_stats(
    user_id: int = Depends(get_current_user_id),
):
    """Xem trạng thái rate limit của user hiện tại."""
    return get_rate_limiter().get_stats(user_id)


@router.get("/guardrails/config")
async def guardrails_config():
    """Xem cấu hình guardrails hiện tại."""
    from app.config import get_settings
    s = get_settings()
    return {
        "guardrails_enabled": s.GUARDRAILS_ENABLED,
        "rate_limit_per_minute": s.RATE_LIMIT_PER_MINUTE,
        "rate_limit_per_hour": s.RATE_LIMIT_PER_HOUR,
        "max_message_length": s.MAX_MESSAGE_LENGTH,
        "reranker_enabled": s.RERANKER_ENABLED,
        "reranker_model": s.RERANKER_MODEL,
        "use_function_calling": s.USE_FUNCTION_CALLING,
    }


# ── Exam Solver (OCR + Per-Question RAG Pipeline) ────────────────────────────

@router.post("/exam-solver/solve")
async def solve_exam(
    file: UploadFile = File(...),
    ocr_engine: str = Form(default="cloud"),
    raw_ocr_text: Optional[str] = Form(default=None),
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Upload exam image/PDF → OCR → Split → Solve each question → Stream results.

    SSE stream format:
      data: {"type": "ocr", "message": "Đang OCR..."}
      data: {"type": "split", "total": 40}
      data: {"type": "progress", "current": 3, "total": 40, "question": "Câu 3"}
      data: {"type": "solution", "question_number": "Câu 3", "solution": "..."}
      data: {"type": "done", "result": {...}}
      data: {"type": "error", "message": "..."}
    """
    # Validate file type
    mime = file.content_type or ""
    if mime.startswith("image/"):
        file_type = "image"
    elif mime == "application/pdf":
        file_type = "pdf"
    else:
        raise HTTPException(400, "Chỉ hỗ trợ file ảnh hoặc PDF")

    # Validate OCR engine
    if ocr_engine not in ("cloud", "local"):
        ocr_engine = "cloud"

    file_bytes = await file.read()

    # Validate file size (max 20MB)
    if len(file_bytes) > 20 * 1024 * 1024:
        raise HTTPException(400, "File quá lớn (tối đa 20MB)")

    async def event_generator():
        """SSE event generator — streams progress updates per question."""
        from app.exam_solver.solver import ExamSolver, ExamSolverResult

        try:
            yield f"data: {json.dumps({'type': 'ocr', 'message': f'Đang OCR đề thi (engine: {ocr_engine})...'})}\n\n"

            solver = ExamSolver(ocr_engine=ocr_engine)

            # Progress tracking via list (mutable in closure)
            progress_events: list[str] = []

            def on_progress(current: int, total: int, question_num: str):
                progress_events.append(
                    json.dumps({
                        "type": "progress",
                        "current": current,
                        "total": total,
                        "question": question_num,
                    })
                )

            result: ExamSolverResult = await solver.solve(
                file_bytes=file_bytes,
                file_type=file_type,
                mime_type=mime,
                user_id=user_id,
                db=db,
                on_progress=on_progress,
                raw_ocr_text=raw_ocr_text,
            )

            # Emit progress events that accumulated during solving
            for evt in progress_events:
                yield f"data: {evt}\n\n"

            # Emit final result
            result_dict = {
                "type": "done",
                "result": {
                    "total_questions": result.total_questions,
                    "questions": [
                        {
                            "question_number": s.question_number,
                            "question_type": s.question_type,
                            "content": s.content,
                            "skill_id": s.skill_id,
                            "skill_name": s.skill_name,
                            "solution": s.solution,
                            "error": s.error,
                        }
                        for s in result.solutions
                    ],
                    "report_markdown": result.report_markdown,
                    "skill_stats": result.skill_stats,
                    "raw_ocr": result.raw_ocr,
                    "ocr_engine_used": result.ocr_engine_used,
                    "elapsed_seconds": result.elapsed_seconds,
                },
            }
            yield f"data: {json.dumps(result_dict, ensure_ascii=False)}\n\n"

        except Exception as e:
            import traceback
            logger.error("Exam Solver error: %s\n%s", e, traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/exam-solver/ocr")
async def extract_ocr_only(
    file: UploadFile = File(...),
    ocr_engine: str = Form(default="cloud"),
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Only perform OCR on the exam file."""
    mime = file.content_type or ""
    if mime.startswith("image/"):
        file_type = "image"
    elif mime == "application/pdf":
        file_type = "pdf"
    else:
        raise HTTPException(400, "Chỉ hỗ trợ file ảnh hoặc PDF")

    file_bytes = await file.read()
    if len(file_bytes) > 20 * 1024 * 1024:
        raise HTTPException(400, "File quá lớn (tối đa 20MB)")

    from app.ocr.ocr_strategy import get_ocr_engine
    ocr = get_ocr_engine(ocr_engine)
    
    if file_type == "pdf":
        page_texts = await ocr.ocr_pdf(file_bytes)
        raw_ocr = "\n\n---\n\n".join(
            f"[Trang {i+1}]\n{text}" for i, text in enumerate(page_texts)
        )
    else:
        raw_ocr = await ocr.ocr_image(file_bytes, mime, is_exam=True)
        
    return {"raw_ocr": raw_ocr}


@router.get("/exam-solver/ocr-engines")
async def list_ocr_engines():
    """List available OCR engines and their status."""
    engines = [
        {
            "id": "cloud",
            "name": "☁️ Cloud VLM (GPT Vision)",
            "description": "Chính xác nhất — dùng GPT-4o-mini Vision. Tốn phí, cần internet.",
            "available": True,
            "recommended": True,
        },
    ]

    # Check if GOT-OCR is available
    try:
        from app.ocr.got_ocr import get_got_ocr
        got = get_got_ocr()
        available = got.is_available
        device = got.device_info
        engines.append({
            "id": "local",
            "name": "🖥️ Local OCR (GOT-OCR2.0)",
            "description": f"Miễn phí, offline. Device: {device}.",
            "available": available,
            "recommended": False,
        })
    except Exception:
        engines.append({
            "id": "local",
            "name": "🖥️ Local OCR (GOT-OCR2.0)",
            "description": "Không khả dụng — cần cài torch và transformers.",
            "available": False,
            "recommended": False,
        })

    return {"engines": engines}

