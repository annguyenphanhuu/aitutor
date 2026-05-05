"""LangChain-compatible retrievers using hybrid search.

Two retrievers exposed:
  • TheoryRetriever — tìm lý thuyết SGK
  • ExamRetriever   — tìm câu hỏi đề thi thật
  • UnifiedRetriever — kết hợp cả hai (dùng cho Teacher Agent)
"""

from langchain.schema import BaseRetriever, Document
from pydantic import Field
from typing import Any, Optional

from app.rag.knowledge_base import KnowledgeBase, get_knowledge_base


# ── Theory Retriever ──────────────────────────────────────

class TheoryRetriever(BaseRetriever):
    """Retriever cho lý thuyết SGK Toán 12."""

    kb: Any = Field(default=None)
    k: int = Field(default=5)
    alpha: float = Field(default=0.6)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, k: int = 5, alpha: float = 0.6, **kwargs):
        super().__init__(k=k, alpha=alpha, **kwargs)
        self.kb = get_knowledge_base()

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        results = self.kb.search_theory(query, k=self.k, alpha=self.alpha)
        return [
            Document(
                page_content=r["content"],
                metadata={**r["metadata"], "hybrid_score": r["hybrid_score"]},
            )
            for r in results
        ]

    async def _aget_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        return self._get_relevant_documents(query, **kwargs)


# ── Exam Retriever ────────────────────────────────────────

class ExamRetriever(BaseRetriever):
    """
    Retriever cho câu hỏi đề thi thật.
    Hỗ trợ filter: year, exam_source, difficulty_part, skill_id, question_type.
    """

    kb: Any = Field(default=None)
    k: int = Field(default=5)
    alpha: float = Field(default=0.5)
    year: Optional[int] = Field(default=None)
    exam_source: Optional[str] = Field(default=None)
    difficulty_part: Optional[int] = Field(default=None)
    skill_id: Optional[str] = Field(default=None)
    question_type: Optional[str] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        k: int = 5,
        alpha: float = 0.5,
        year: Optional[int] = None,
        exam_source: Optional[str] = None,
        difficulty_part: Optional[int] = None,
        skill_id: Optional[str] = None,
        question_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            k=k, alpha=alpha,
            year=year, exam_source=exam_source,
            difficulty_part=difficulty_part,
            skill_id=skill_id, question_type=question_type,
            **kwargs,
        )
        self.kb = get_knowledge_base()

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        results = self.kb.search_exams(
            query, k=self.k, alpha=self.alpha,
            year=self.year,
            exam_source=self.exam_source,
            difficulty_part=self.difficulty_part,
            skill_id=self.skill_id,
            question_type=self.question_type,
        )
        return [
            Document(
                page_content=r["content"],
                metadata={**r["metadata"], "hybrid_score": r["hybrid_score"]},
            )
            for r in results
        ]

    async def _aget_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        return self._get_relevant_documents(query, **kwargs)


# ── Unified Retriever ─────────────────────────────────────

class UnifiedRetriever(BaseRetriever):
    """
    Retriever tìm đồng thời cả lý thuyết lẫn đề thi.
    Hữu ích cho Teacher Agent khi cần ngữ cảnh đa dạng.
    """

    kb: Any = Field(default=None)
    k: int = Field(default=5)
    alpha: float = Field(default=0.55)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, k: int = 5, alpha: float = 0.55, **kwargs):
        super().__init__(k=k, alpha=alpha, **kwargs)
        self.kb = get_knowledge_base()

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        results = self.kb.search_all(query, k=self.k, alpha=self.alpha)
        return [
            Document(
                page_content=r["content"],
                metadata={**r["metadata"], "hybrid_score": r["hybrid_score"]},
            )
            for r in results
        ]

    async def _aget_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        return self._get_relevant_documents(query, **kwargs)


# ── Backward-compatible alias ─────────────────────────────
# Giữ tên HybridRetriever để không phá vỡ code cũ đang import
HybridRetriever = UnifiedRetriever
