"""Input Validator — lọc tin nhắn không an toàn trước khi gửi vào pipeline.

Threat models:
    1. Prompt injection  — cố tình override system prompt
    2. Jailbreak         — yêu cầu LLM bỏ qua rules
    3. Message quá dài   — token bomb, tốn chi phí
    4. Empty message     — gây lỗi downstream

PII masking:
    Che CMND/CCCD (9/12 số), số điện thoại VN, email trước khi log.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Injection detection patterns ──────────────────────────────────────────────
# Dùng các cụm từ thường xuất hiện trong prompt injection / jailbreak tiếng Việt + tiếng Anh
_INJECTION_PATTERNS: list[re.Pattern] = [
    # English jailbreak classics
    re.compile(r"ignore\s+(all\s+)?(previous|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|above|prior)\s+instructions?", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(previous|your)\s+instructions?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an|the)\s+\w+\s*(ai|bot|assistant|model)?", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+)?(you\s+are|a|an)\s+\w+", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)", re.IGNORECASE),
    re.compile(r"do\s+not\s+follow\s+(your\s+)?(instructions?|rules?|guidelines?)", re.IGNORECASE),
    re.compile(r"override\s+(system|prompt|instructions?)", re.IGNORECASE),
    re.compile(r"\[system\]|\[assistant\]|\[admin\]|\[root\]", re.IGNORECASE),
    # Vietnamese variants
    re.compile(r"b[oỏ]?\s+qua\s+(m[oọ]i\s+)?(h[uướ][oớ]ng?\s+d[aẫ]n|quy\s+t[aắ]c)", re.IGNORECASE),
    re.compile(r"gi[aả]\s+v[oờ]\s+(b[aạ]n\s+l[aà]|m[iì]nh\s+l[aà])", re.IGNORECASE),
    re.compile(r"b[aả]y\s+gi[oờ]\s+b[aạ]n\s+l[aà]", re.IGNORECASE),
]

# ── PII patterns ──────────────────────────────────────────────────────────────
_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    # CMND/CCCD VN: 9 hoặc 12 chữ số liên tiếp
    (re.compile(r"\b\d{9}\b|\b\d{12}\b"), "[CMND/CCCD]"),
    # SĐT VN: 0xx hoặc +84xx, 10-11 số
    (re.compile(r"(?:\+84|0)[3-9]\d{8}"), "[SĐT]"),
    # Email
    (re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), "[EMAIL]"),
]


@dataclass
class ValidationResult:
    """Kết quả validation của một tin nhắn."""
    is_safe: bool
    message: str                      # message đã được xử lý (truncate, v.v.)
    rejection_reason: Optional[str]   # lý do từ chối nếu is_safe=False
    warning: Optional[str]            # cảnh báo nhẹ (vẫn cho qua)


def validate_input(
    message: str,
    user_id: Optional[int] = None,
) -> ValidationResult:
    """
    Validate và sanitize tin nhắn đầu vào.

    Parameters
    ----------
    message : str
        Nội dung tin nhắn của học sinh.
    user_id : int | None
        Để log context.

    Returns
    -------
    ValidationResult
        is_safe=False → API trả về 400 Bad Request.
        is_safe=True  → message đã xử lý, sẵn sàng vào pipeline.
    """
    if not settings.GUARDRAILS_ENABLED:
        return ValidationResult(is_safe=True, message=message,
                                rejection_reason=None, warning=None)

    # ── 1. Empty check ────────────────────────────────────────────────────
    stripped = message.strip()
    if not stripped:
        return ValidationResult(
            is_safe=False,
            message=message,
            rejection_reason="Tin nhắn trống. Vui lòng nhập câu hỏi của bạn.",
            warning=None,
        )

    # ── 2. Length truncation ──────────────────────────────────────────────
    warning: Optional[str] = None
    max_len = settings.MAX_MESSAGE_LENGTH
    if len(stripped) > max_len:
        stripped = stripped[:max_len]
        warning = (
            f"Tin nhắn quá dài (>{max_len} ký tự) — đã tự động cắt bớt. "
            "Vui lòng chia nhỏ câu hỏi nếu cần."
        )
        logger.warning(
            "⚠️  Message truncated (user_id=%s, original_len=%d)",
            user_id, len(message),
        )

    # ── 3. Injection detection ────────────────────────────────────────────
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(stripped):
            logger.warning(
                "🚨 Prompt injection detected (user_id=%s): pattern=%s",
                user_id,
                pattern.pattern[:60],
            )
            # ── Log blocked attempt lên Langfuse ────────────────────────
            try:
                from app.utils.langfuse_client import new_trace, score_trace
                block_trace = new_trace(
                    name="guardrails.injection_blocked",
                    user_id=str(user_id) if user_id else None,
                    metadata={
                        "reason": "prompt_injection",
                        "pattern": pattern.pattern[:60],
                        "message_preview": mask_pii(stripped[:100]),
                    },
                )
                if block_trace:
                    score_trace(
                        trace_id=block_trace.id,
                        name="safety_pass",
                        value=0.0,
                        comment="injection_blocked",
                        data_type="NUMERIC",
                    )
            except Exception:
                pass  # Guardrails không được phép crash vì Langfuse
            return ValidationResult(
                is_safe=False,
                message=message,
                rejection_reason=(
                    "Tin nhắn chứa nội dung không phù hợp. "
                    "Mình chỉ hỗ trợ câu hỏi về Toán 12 nhé!"
                ),
                warning=None,
            )

    return ValidationResult(
        is_safe=True,
        message=stripped,
        rejection_reason=None,
        warning=warning,
    )


def mask_pii(text: str) -> str:
    """
    Thay thế thông tin cá nhân nhạy cảm bằng placeholder trước khi log.

    Parameters
    ----------
    text : str
        Nội dung cần mask.

    Returns
    -------
    str
        Text đã được mask PII.
    """
    masked = text
    for pattern, replacement in _PII_PATTERNS:
        masked = pattern.sub(replacement, masked)
    return masked
