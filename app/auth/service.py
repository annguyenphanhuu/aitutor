"""Simple username-based authentication service.

No passwords — just username login. Creates user if not exists.
User ID is passed via X-User-Id header from frontend (stored in localStorage).
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
from fastapi import Header, HTTPException, Depends

from app.db.models import User
from app.db.database import get_db


async def login_or_register(db: AsyncSession, username: str) -> dict:
    """Login with username. Create account if not exists."""
    username = username.strip()
    if not username or len(username) < 2:
        raise ValueError("Tên đăng nhập phải có ít nhất 2 ký tự.")

    result = await db.execute(
        select(User).where(User.username == username)
    )
    user = result.scalar_one_or_none()

    if user:
        # Existing user — update last login
        user.last_login = datetime.utcnow()
        await db.flush()
        return {
            "user_id": user.id,
            "username": user.username,
            "display_name": user.display_name or user.username,
            "is_new": False,
        }
    else:
        # New user — create
        user = User(
            username=username,
            display_name=username,
        )
        db.add(user)
        await db.flush()
        return {
            "user_id": user.id,
            "username": user.username,
            "display_name": user.display_name,
            "is_new": True,
        }


async def get_current_user_id(
    x_user_id: str = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> int:
    """FastAPI dependency: extract user_id from X-User-Id header.
    
    If no header provided, returns default user_id=1 for backward compatibility.
    """
    if not x_user_id:
        return 1  # default user for backward compat

    try:
        uid = int(x_user_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="X-User-Id phải là số nguyên.")

    # Verify user exists
    result = await db.execute(select(User).where(User.id == uid))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User không tồn tại. Vui lòng đăng nhập lại.")

    return uid
