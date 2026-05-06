"""LangChain-compatible retrievers for theory-only RAG."""

from __future__ import annotations

from typing import Any

from langchain.schema import BaseRetriever, Document
from pydantic import Field

from app.rag.knowledge_base import get_knowledge_base


class TheoryRetriever(BaseRetriever):
    """Retriever for Grade 12 math theory documents."""

    kb: Any = Field(default=None)
    k: int = Field(default=5)
    alpha: float = Field(default=0.6)
    skill_ids: list[str] | None = Field(default=None)
    chapter: str | None = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        k: int = 5,
        alpha: float = 0.6,
        skill_ids: list[str] | None = None,
        chapter: str | None = None,
        **kwargs,
    ):
        super().__init__(
            k=k,
            alpha=alpha,
            skill_ids=skill_ids,
            chapter=chapter,
            **kwargs,
        )
        self.kb = get_knowledge_base()

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        results = self.kb.search_theory(
            query,
            k=self.k,
            alpha=self.alpha,
            skill_ids=self.skill_ids,
            chapter=self.chapter,
        )
        return [
            Document(
                page_content=result["content"],
                metadata={
                    **result.get("metadata", {}),
                    "hybrid_score": result.get("hybrid_score", 0.0),
                },
            )
            for result in results
        ]

    async def _aget_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        return self._get_relevant_documents(query, **kwargs)


class UnifiedRetriever(TheoryRetriever):
    """Backward-compatible name for theory-only retrieval."""


class ExamRetriever(TheoryRetriever):
    """Deprecated compatibility alias.

    Exam questions are no longer embedded in RAG. This class intentionally
    falls back to theory retrieval to preserve imports without leaking exams.
    """


HybridRetriever = UnifiedRetriever
