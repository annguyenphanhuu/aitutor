"""Langfuse client singleton — LLM observability & tracing.

Usage:
    from app.utils.langfuse_client import get_langfuse, new_trace, new_generation

    # Tạo trace cho một request
    trace = new_trace(name="chat", user_id="42", metadata={"skill_id": "derivative_basic"})

    # Tạo generation span cho một LLM call
    gen = new_generation(name="teacher_llm", model="gpt-4o", input_text="Câu hỏi...")
    # ... LLM call ...
    gen and gen.end(output="Trả lời...", usage={"input": 100, "output": 200})

    # Update trace khi xong
    trace and trace.update(output="final_answer", metadata={"latency_ms": 1200})

Graceful fallback: Nếu LANGFUSE_ENABLED=false hoặc keys thiếu → trả về None.
App KHÔNG bao giờ crash vì Langfuse.
"""

import logging
from functools import lru_cache
from typing import Optional, Any

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_langfuse():
    """Return Langfuse client singleton, hoặc None nếu disabled/misconfigured.

    Cached: chỉ khởi tạo 1 lần trong lifetime của process.
    Dùng Langfuse v2 SDK (v2.x).
    """
    from app.config import get_settings
    settings = get_settings()

    if not settings.LANGFUSE_ENABLED:
        return None
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        logger.warning(
            "⚠️  Langfuse ENABLED nhưng thiếu PUBLIC_KEY/SECRET_KEY — bỏ qua tracing."
        )
        return None

    try:
        from langfuse import Langfuse
        client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
            flush_at=1,           # Flush ngay sau mỗi event (dev mode)
            flush_interval=2,     # Hoặc flush mỗi 2 giây
        )
        # Kiểm tra kết nối ngay khi khởi tạo
        ok = client.auth_check()
        if ok:
            logger.info("✅ Langfuse v2 connected → %s", settings.LANGFUSE_HOST)
        else:
            logger.warning("⚠️ Langfuse auth_check failed — traces sẽ không được ghi")
        return client
    except Exception as e:
        logger.warning("Langfuse init failed (graceful no-op): %s", e)
        return None


def new_trace(
    name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    input_text: Optional[str] = None,
) -> Optional[Any]:
    """Tạo Langfuse trace mới. Trả về None nếu Langfuse disabled."""
    lf = get_langfuse()
    if not lf:
        return None
    try:
        kwargs: dict = {"name": name}
        if user_id is not None:
            kwargs["user_id"] = str(user_id)
        if session_id is not None:
            kwargs["session_id"] = str(session_id)
        if metadata:
            kwargs["metadata"] = metadata
        if input_text:
            kwargs["input"] = input_text
        return lf.trace(**kwargs)
    except Exception as e:
        logger.debug("Langfuse new_trace failed (ignoring): %s", e)
        return None


def new_span(
    name: str,
    trace_id: Optional[str] = None,
    parent_observation_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    input_data: Optional[Any] = None,
) -> Optional[Any]:
    """Tạo một Span con trong một trace để group các bước logic.

    Khác với generation (dùng cho LLM calls), span dùng cho các bước
    non-LLM như RAG retrieval, reranking, tool execution, v.v.

    Parameters
    ----------
    name : str
        Tên của span, ví dụ: "rag.graph_retrieval", "tool.compute_derivative".
    trace_id : str | None
        ID của trace cha. Nếu None, span sẽ là root-level observation.
    parent_observation_id : str | None
        ID của span/generation cha (để lồng nhau nhiều cấp).
    metadata : dict | None
        Metadata tuỳ chỉnh đính kèm span.
    input_data : Any | None
        Dữ liệu input của bước này (sẽ hiển thị trên Langfuse UI).
    """
    lf = get_langfuse()
    if not lf:
        return None
    try:
        kwargs: dict = {"name": name}
        if trace_id:
            kwargs["trace_id"] = trace_id
        if parent_observation_id:
            kwargs["parent_observation_id"] = parent_observation_id
        if metadata:
            kwargs["metadata"] = metadata
        if input_data is not None:
            kwargs["input"] = input_data
        return lf.span(**kwargs)
    except Exception as e:
        logger.debug("Langfuse new_span failed (ignoring): %s", e)
        return None


def new_generation(
    name: str,
    model: str,
    input_text: Optional[str] = None,
    metadata: Optional[dict] = None,
    trace_id: Optional[str] = None,
    parent_observation_id: Optional[str] = None,
) -> Optional[Any]:
    """Tạo Langfuse generation span. Trả về None nếu Langfuse disabled."""
    lf = get_langfuse()
    if not lf:
        return None
    try:
        kwargs: dict = {"name": name, "model": model}
        if input_text:
            kwargs["input"] = input_text[:1000]   # truncate để tránh payload quá to
        if metadata:
            kwargs["metadata"] = metadata
        if trace_id:
            kwargs["trace_id"] = trace_id
        if parent_observation_id:
            kwargs["parent_observation_id"] = parent_observation_id
        return lf.generation(**kwargs)
    except Exception as e:
        logger.debug("Langfuse new_generation failed (ignoring): %s", e)
        return None


def end_generation(
    generation: Optional[Any],
    output: Optional[str] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Kết thúc một generation span. Safe no-op nếu generation=None."""
    if not generation:
        return
    try:
        kwargs: dict = {}
        if output:
            kwargs["output"] = output[:1000]
        if input_tokens or output_tokens:
            kwargs["usage"] = {"input": input_tokens, "output": output_tokens}
        generation.end(**kwargs)
    except Exception as e:
        logger.debug("Langfuse end_generation failed (ignoring): %s", e)


def end_span(
    span: Optional[Any],
    output: Optional[Any] = None,
    metadata: Optional[dict] = None,
    level: str = "DEFAULT",
) -> None:
    """Kết thúc một span. Safe no-op nếu span=None.

    Parameters
    ----------
    span : Any | None
        Span object trả về từ new_span().
    output : Any | None
        Kết quả của bước này (dict, str, v.v.).
    metadata : dict | None
        Metadata bổ sung khi kết thúc span.
    level : str
        "DEFAULT" | "WARNING" | "ERROR" — để highlight lỗi trên UI.
    """
    if not span:
        return
    try:
        kwargs: dict = {}
        if output is not None:
            kwargs["output"] = output
        if metadata:
            kwargs["metadata"] = metadata
        if level != "DEFAULT":
            kwargs["level"] = level
        span.end(**kwargs)
    except Exception as e:
        logger.debug("Langfuse end_span failed (ignoring): %s", e)


def score_trace(
    trace_id: str,
    name: str,
    value: float,
    comment: Optional[str] = None,
    data_type: str = "NUMERIC",
) -> None:
    """Push một score vào trace trên Langfuse.

    Dùng để ghi kết quả từ MathJudge, RAGAS, hallucination tracker
    vào trace tương ứng. Giúp theo dõi chất lượng mô hình theo thời gian.

    Parameters
    ----------
    trace_id : str
        ID của trace cần gán score.
    name : str
        Tên metric, ví dụ: "accuracy", "step_clarity", "hallucination_rate".
    value : float
        Giá trị score (thường 0.0–1.0).
    comment : str | None
        Chú thích ngắn, ví dụ: "type=exam_mcq".
    data_type : str
        "NUMERIC" (mặc định) hoặc "BOOLEAN" hoặc "CATEGORICAL".
    """
    lf = get_langfuse()
    if not lf or not trace_id:
        return
    try:
        kwargs: dict = {
            "trace_id": trace_id,
            "name": name,
            "value": value,
            "data_type": data_type,
        }
        if comment:
            kwargs["comment"] = comment
        lf.score(**kwargs)
    except Exception as e:
        logger.debug("Langfuse score_trace failed (ignoring): %s", e)


def get_prompt(name: str, fallback: str = "", label: str = "production") -> str:
    """Fetch prompt từ Langfuse Prompt Management.

    Dùng để decouple system prompts khỏi codebase — Data Scientist có thể
    chỉnh sửa prompt trực tiếp trên Langfuse UI mà không cần deploy lại.

    Parameters
    ----------
    name : str
        Tên prompt đã đăng ký trên Langfuse, ví dụ: "aitutor-socratic-v1".
    fallback : str
        Nội dung prompt dự phòng nếu không fetch được từ Langfuse.
    label : str
        Label version của prompt, mặc định "production".

    Returns
    -------
    str
        Nội dung prompt (string raw, chưa format). Trả về fallback nếu lỗi.
    """
    lf = get_langfuse()
    if not lf:
        return fallback
    try:
        prompt_obj = lf.get_prompt(name, label=label)
        # Langfuse v2: prompt_obj.prompt là string hoặc list of messages
        raw = prompt_obj.prompt
        if isinstance(raw, str):
            return raw
        # Nếu là list (chat prompt format), nối lại thành string
        if isinstance(raw, list):
            return "\n\n".join(
                msg.get("content", "") for msg in raw if isinstance(msg, dict)
            )
        return fallback
    except Exception as e:
        logger.warning("Langfuse get_prompt '%s' failed, using fallback: %s", name, e)
        return fallback


def update_trace(
    trace: Optional[Any],
    output: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Update một trace và flush ngay. Safe no-op nếu trace=None."""
    if not trace:
        return
    try:
        kwargs: dict = {}
        if output:
            kwargs["output"] = output[:1000]
        if metadata:
            kwargs["metadata"] = metadata
        trace.update(**kwargs)
        # Flush ngay — đảm bảo data được gửi lên Langfuse Cloud
        lf = get_langfuse()
        if lf:
            lf.flush()
    except Exception as e:
        logger.debug("Langfuse update_trace failed (ignoring): %s", e)


def langfuse_auth_check() -> bool:
    """Kiểm tra kết nối Langfuse. Dùng khi startup để verify config."""
    lf = get_langfuse()
    if not lf:
        logger.info("ℹ️  Langfuse disabled — bỏ qua tracing.")
        return False
    try:
        ok = lf.auth_check()
        if ok:
            logger.info("✅ Langfuse auth OK")
        else:
            logger.warning("⚠️ Langfuse auth FAILED")
        return ok
    except Exception as e:
        logger.warning("Langfuse auth_check error: %s", e)
        return False
