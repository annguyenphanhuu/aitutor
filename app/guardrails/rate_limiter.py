"""Rate Limiter — Sliding window rate limiting per user_id.

Giới hạn:
    - 30 requests / 1 phút
    - 200 requests / 1 giờ

In-memory implementation dùng deque (O(1) amortized per check).
Thread-safe với threading.Lock.

Lưu ý: Reset khi restart server — dùng Redis nếu cần persistence.
"""

from __future__ import annotations

import threading
import time
import logging
from collections import defaultdict, deque
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RateLimiter:
    """
    Sliding window rate limiter per user_id.

    Usage:
        limiter = get_rate_limiter()
        allowed, reason = limiter.is_allowed(user_id=42)
        if not allowed:
            raise HTTPException(429, detail=reason)
    """

    def __init__(
        self,
        per_minute: int | None = None,
        per_hour: int | None = None,
    ):
        self._per_minute = per_minute or settings.RATE_LIMIT_PER_MINUTE
        self._per_hour = per_hour or settings.RATE_LIMIT_PER_HOUR

        # user_id → deque of timestamps (float, seconds since epoch)
        self._minute_windows: dict[int, deque] = defaultdict(deque)
        self._hour_windows: dict[int, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def is_allowed(self, user_id: int) -> tuple[bool, Optional[str]]:
        """
        Kiểm tra xem user có được phép gửi request không.

        Parameters
        ----------
        user_id : int

        Returns
        -------
        (True, None)         — cho phép
        (False, reason_str)  — từ chối, kèm lý do
        """
        if not settings.GUARDRAILS_ENABLED:
            return True, None

        now = time.time()
        minute_cutoff = now - 60.0
        hour_cutoff = now - 3600.0

        with self._lock:
            minute_q = self._minute_windows[user_id]
            hour_q = self._hour_windows[user_id]

            # Slide windows: loại bỏ timestamps cũ
            while minute_q and minute_q[0] <= minute_cutoff:
                minute_q.popleft()
            while hour_q and hour_q[0] <= hour_cutoff:
                hour_q.popleft()

            # Kiểm tra giới hạn
            if len(minute_q) >= self._per_minute:
                wait = int(60 - (now - minute_q[0])) + 1
                logger.warning("🚦 Rate limit (minute) hit: user_id=%d", user_id)
                return False, (
                    f"Bạn đang gửi quá nhanh ({self._per_minute} tin/phút). "
                    f"Vui lòng đợi ~{wait} giây."
                )

            if len(hour_q) >= self._per_hour:
                wait_min = int((3600 - (now - hour_q[0])) / 60) + 1
                logger.warning("🚦 Rate limit (hour) hit: user_id=%d", user_id)
                return False, (
                    f"Bạn đã đạt giới hạn {self._per_hour} tin/giờ. "
                    f"Vui lòng thử lại sau ~{wait_min} phút."
                )

            # Ghi timestamp mới
            minute_q.append(now)
            hour_q.append(now)

        return True, None

    def get_stats(self, user_id: int) -> dict:
        """Trả về stats hiện tại của user (dùng cho debug / admin)."""
        now = time.time()
        with self._lock:
            minute_count = sum(1 for t in self._minute_windows[user_id] if t > now - 60)
            hour_count = sum(1 for t in self._hour_windows[user_id] if t > now - 3600)
        return {
            "user_id": user_id,
            "requests_last_minute": minute_count,
            "requests_last_hour": hour_count,
            "limit_per_minute": self._per_minute,
            "limit_per_hour": self._per_hour,
        }

    def reset_user(self, user_id: int) -> None:
        """Reset rate limit counter của một user (dùng cho tests/admin)."""
        with self._lock:
            self._minute_windows[user_id].clear()
            self._hour_windows[user_id].clear()


# ── Singleton ─────────────────────────────────────────────────────────────────
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
