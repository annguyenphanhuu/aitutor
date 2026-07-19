"""Time helpers shared across the app."""

from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Naive UTC now — thay thế ``datetime.utcnow()`` (deprecated từ Python 3.12).

    Giữ naive datetime (không tzinfo) để tương thích với dữ liệu DateTime
    hiện có trong DB (SQLite/PostgreSQL đều đang lưu naive-UTC). Nếu sau
    này chuyển sang timezone-aware, đổi tại một chỗ này.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)
