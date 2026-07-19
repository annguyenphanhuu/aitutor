"""GraphTutorEngine — adapter cùng chữ ký với Orchestrator.

routes.py chỉ cần đổi engine (qua flag USE_LANGGRAPH) mà không đổi call site:
  - handle_message(...)        → dict giống hệt Orchestrator.handle_message
  - handle_message_stream(...) → async generator yield các JSON frame string
    (meta → token* → done/error) giống hệt Orchestrator.handle_message_stream
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.builder import build_tutor_graph
from app.graph.nodes import TutorNodes
from app.graph.state import TutorState

logger = logging.getLogger(__name__)


class GraphTutorEngine:
    """Engine LangGraph — thay thế Orchestrator if/elif."""

    def __init__(self):
        self.nodes = TutorNodes()
        self.graph = build_tutor_graph(self.nodes)

    @staticmethod
    def _initial_state(message: str, mode: str, chat_history: Optional[list[dict]]) -> TutorState:
        return {
            "message": message,
            "requested_mode": mode,
            "chat_history": chat_history or [],
        }

    @staticmethod
    def _config(
        db: AsyncSession,
        user_id: int,
        session_id: Optional[int],
        trace_id: Optional[str],
        streaming: bool,
    ) -> dict:
        return {
            "configurable": {
                "db": db,
                "user_id": user_id,
                "session_id": session_id,
                "trace_id": trace_id,
                "streaming": streaming,
            }
        }

    async def handle_message(
        self,
        db: AsyncSession,
        message: str,
        mode: str = "auto",
        chat_history: Optional[list[dict]] = None,
        user_id: int = 1,
        langfuse_trace=None,
        session_id: Optional[int] = None,
        **_,
    ) -> dict:
        trace_id: Optional[str] = getattr(langfuse_trace, "id", None)
        final_state = await self.graph.ainvoke(
            self._initial_state(message, mode, chat_history),
            config=self._config(db, user_id, session_id, trace_id, streaming=False),
        )
        return final_state["result"]

    async def handle_message_stream(
        self,
        db: AsyncSession,
        message: str,
        mode: str = "auto",
        chat_history: Optional[list[dict]] = None,
        user_id: int = 1,
        session_id: Optional[int] = None,
        **_,
    ):
        """Async generator — yield JSON frame strings cho SSE endpoint."""
        try:
            async for frame in self.graph.astream(
                self._initial_state(message, mode, chat_history),
                config=self._config(db, user_id, session_id, None, streaming=True),
                stream_mode="custom",
            ):
                # stream_mode="custom" yield đúng giá trị writer() đã emit (JSON string)
                yield frame
        except Exception as exc:
            logger.error("GraphTutorEngine stream error: %s", exc)
            yield json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False)
