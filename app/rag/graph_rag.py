"""GraphRAG — Knowledge-Graph-Enhanced Retrieval-Augmented Generation.

Combines the Skill Graph (prerequisite dependencies) with Vector+BM25
hybrid search to create *hyper-personalised* context for each student.

Key idea (inspired by Microsoft GraphRAG, 2024):
  When a student asks about Skill A, we don't just search the vector DB
  for Skill A documents.  We also traverse the prerequisite graph:
    1. Find all prerequisite skills of A  (via BFS in skill_graph).
    2. Check BKT mastery scores — keep only *weak* prerequisites.
    3. Issue secondary searches for each weak prerequisite.
    4. Merge & deduplicate, then rank by a combined score.

  The result: if a student asks about "Tích phân từng phần" but is
  weak at "Đạo hàm", the context will *automatically* include
  derivative theory — and the Teacher Agent can remind the student.

Usage
-----
    retriever = GraphRAGRetriever(k=5)
    docs = retriever.retrieve(
        query="Tính tích phân ...",
        skill_id="integral_definite",
        masteries={"derivative_basic": 0.3, ...},
    )
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain.schema import Document

from app.rag.knowledge_base import KnowledgeBase, get_knowledge_base
from app.knowledge_tracing.skill_graph import (
    SKILLS,
    get_deep_prerequisites,
    find_weak_prerequisites,
)

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
PREREQUISITE_MASTERY_THRESHOLD = 0.5   # Below this → weak
MAX_PREREQUISITE_QUERIES = 3           # Limit extra searches to control cost
PREREQUISITE_DOCS_PER_QUERY = 2        # Docs retrieved per prerequisite query


class GraphRAGRetriever:
    """Graph-enhanced retriever: main query + prerequisite gap queries.

    This retriever is *not* a LangChain BaseRetriever for simplicity —
    it is called directly by the TeacherAgent which needs the richer
    return type anyway.
    """

    def __init__(
        self,
        k: int = 5,
        alpha: float = 0.55,
        gap_threshold: float = PREREQUISITE_MASTERY_THRESHOLD,
    ):
        self.kb: KnowledgeBase = get_knowledge_base()
        self.k = k
        self.alpha = alpha
        self.gap_threshold = gap_threshold

    # ── Public API ───────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        skill_id: Optional[str] = None,
        masteries: Optional[dict[str, float]] = None,
    ) -> GraphRAGResult:
        """Retrieve documents with automatic prerequisite enrichment.

        Parameters
        ----------
        query : str
            The student's question or search text.
        skill_id : str | None
            The classified skill for this query (from Orchestrator).
        masteries : dict | None
            Current BKT mastery scores ``{skill_id: p_mastery}``.

        Returns
        -------
        GraphRAGResult
            Contains main documents, prerequisite documents, and
            a structured gap summary for prompt injection.
        """
        masteries = masteries or {}

        # ── Step 1: Primary retrieval with QueryExpander + soft boost ──
        from app.rag.query_expander import expand_question
        from app.rag.reranker import get_reranker
        from app.config import get_settings as _get_settings

        _settings = _get_settings()
        reranker = get_reranker()
        raw_pool_k = _settings.RERANKER_CANDIDATE_K if _settings.RERANKER_ENABLED else self.k * 2
        pool_k = max(raw_pool_k, self.k * 4)

        # QueryExpander infers chapters/skills from question content
        expansion = expand_question(query)

        # Query Rewriting: dùng rag_query (thuật ngữ học thuật) thay cho câu hỏi gốc.
        # Lý do: Câu hỏi gốc dạng bài toán thực tế có vector embedding xa với lý thuyết.
        # rag_query có các thuật ngữ đúng hơn nên vector gần KB hơn.
        search_query = expansion.rag_query if (not expansion.error and expansion.rag_query) else query
        candidates = self.kb.search_theory(search_query, k=pool_k, alpha=self.alpha)

        # Soft boost đủ mạnh (0.12/0.07) để docs đúng skill/chapter luôn vào pool.
        SKILL_BOOST   = 0.12
        CHAPTER_BOOST = 0.07
        expanded_skills   = {s.lower() for s in expansion.skill_ids}
        # Normalize unicode dash: ChromaDB có thể dùng en-dash (U+2013) nhưng LLM trả hyphen
        def _norm_ch(s: str) -> str:
            return s.replace("–", "-").replace("—", "-").lower().strip()
        expanded_chapters = {_norm_ch(c) for c in expansion.chapters}

        for r in candidates:
            bonus = 0.0
            meta  = r.get("metadata", {})
            chunk_skill   = str(meta.get("skill_id", "")).lower().strip()
            chunk_chapter = _norm_ch(str(meta.get("chapter",  "")))

            if chunk_skill and chunk_skill in expanded_skills:
                bonus += SKILL_BOOST
            if chunk_chapter and any(
                chunk_chapter in exp_ch or exp_ch in chunk_chapter
                for exp_ch in expanded_chapters
            ):
                bonus += CHAPTER_BOOST

            if bonus > 0:
                r["hybrid_score"] = min(r.get("hybrid_score", 0) + bonus, 1.0)
                r["_boosted"] = True

        main_results = reranker.rerank(query, candidates, k=self.k)

        # Debug log: count boosted docs
        n_boosted = sum(1 for r in main_results if r.get("_boosted"))
        if n_boosted:
            logger.debug(
                "Soft Boost: %d/%d docs boosted (via QueryExpander)",
                n_boosted, len(main_results),
            )

        main_docs = [
            Document(
                page_content=r["content"],
                metadata={**r["metadata"], "hybrid_score": r["hybrid_score"]},
            )
            for r in main_results
        ]

        # ── Step 2: Graph traversal — find weak prerequisites ────────
        prereq_docs: list[Document] = []
        weak_skills: list[dict] = []

        if skill_id and skill_id in SKILLS:
            weak_ids = find_weak_prerequisites(
                masteries, skill_id, threshold=self.gap_threshold
            )
            # Limit the number of extra queries
            weak_ids = weak_ids[:MAX_PREREQUISITE_QUERIES]

            for wid in weak_ids:
                skill_info = SKILLS.get(wid, {})
                skill_name = skill_info.get("name", wid)
                mastery_val = masteries.get(wid, 0.0)

                weak_skills.append({
                    "skill_id": wid,
                    "skill_name": skill_name,
                    "mastery": round(mastery_val, 3),
                })

                # ── Step 3: Secondary search for prerequisite theory ─
                prereq_query = f"{skill_name} lý thuyết cơ bản"
                prereq_results = self.kb.search_theory(
                    prereq_query,
                    k=PREREQUISITE_DOCS_PER_QUERY,
                    alpha=self.alpha,
                )

                for r in prereq_results:
                    doc = Document(
                        page_content=r["content"],
                        metadata={
                            **r["metadata"],
                            "hybrid_score": r["hybrid_score"],
                            "_source": "graph_rag_prerequisite",
                            "_prereq_skill": wid,
                        },
                    )
                    prereq_docs.append(doc)

            if weak_skills:
                logger.info(
                    "📊 GraphRAG: skill=%s → %d weak prereqs found: %s",
                    skill_id,
                    len(weak_skills),
                    [w["skill_id"] for w in weak_skills],
                )

        # ── Step 4: Deduplicate by content ───────────────────────────
        seen_contents: set[str] = set()
        deduped_main: list[Document] = []
        for doc in main_docs:
            fingerprint = doc.page_content[:200]
            if fingerprint not in seen_contents:
                seen_contents.add(fingerprint)
                deduped_main.append(doc)

        deduped_prereq: list[Document] = []
        for doc in prereq_docs:
            fingerprint = doc.page_content[:200]
            if fingerprint not in seen_contents:
                seen_contents.add(fingerprint)
                deduped_prereq.append(doc)

        return GraphRAGResult(
            main_docs=deduped_main,
            prereq_docs=deduped_prereq,
            weak_skills=weak_skills,
        )


class GraphRAGResult:
    """Container for GraphRAG retrieval output."""

    def __init__(
        self,
        main_docs: list[Document],
        prereq_docs: list[Document],
        weak_skills: list[dict],
    ):
        self.main_docs = main_docs
        self.prereq_docs = prereq_docs
        self.weak_skills = weak_skills

    @property
    def all_docs(self) -> list[Document]:
        """All documents (main + prerequisite) combined."""
        return self.main_docs + self.prereq_docs

    def build_context_text(self) -> str:
        """Build a formatted context string for prompt injection.

        Separates main context from prerequisite context with clear
        labels so the LLM knows WHY extra material is included.
        """
        parts: list[str] = []

        # Main context
        if self.main_docs:
            main_text = "\n\n".join(d.page_content for d in self.main_docs)
            parts.append(f"[TÀI LIỆU CHÍNH]\n{main_text}")

        # Prerequisite context (with explanation)
        if self.prereq_docs:
            prereq_text = "\n\n".join(d.page_content for d in self.prereq_docs)
            skills_str = ", ".join(
                f"{w['skill_name']} (mastery={w['mastery']:.0%})"
                for w in self.weak_skills
            )
            parts.append(
                f"[KIẾN THỨC NỀN TẢNG — Học sinh đang yếu: {skills_str}]\n"
                f"Hãy nhắc lại kiến thức này trước khi giảng bài mới.\n\n"
                f"{prereq_text}"
            )

        return "\n\n---\n\n".join(parts) if parts else "Không tìm thấy tài liệu liên quan."

    def build_gap_warning(self) -> str:
        """Build a structured warning string about prerequisite gaps."""
        if not self.weak_skills:
            return ""

        lines = []
        for w in self.weak_skills:
            lines.append(
                f"  • {w['skill_name']} — mastery chỉ {w['mastery']:.0%} (cần ≥50%)"
            )
        return (
            "⚠️ CẢNH BÁO: Học sinh HỔ NG kiến thức nền tảng sau:\n"
            + "\n".join(lines)
            + "\n→ Hãy nhắc lại các khái niệm này trước khi dạy bài mới."
        )
