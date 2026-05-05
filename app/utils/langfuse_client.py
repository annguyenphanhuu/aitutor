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
        if metadata:
            kwargs["metadata"] = metadata
        if input_text:
            kwargs["input"] = input_text
        return lf.trace(**kwargs)
    except Exception as e:
        logger.debug("Langfuse new_trace failed (ignoring): %s", e)
        return None


def new_generation(
    name: str,
    model: str,
    input_text: Optional[str] = None,
    metadata: Optional[dict] = None,
    trace_id: Optional[str] = None,
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
