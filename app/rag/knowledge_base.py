"""RAG Knowledge Base with ChromaDB and Hybrid Search.

Two-collection architecture:
  - math_theory  : lý thuyết Toán 12 (định nghĩa, công thức, tính chất)
  - math_exams   : câu hỏi từ đề thi thật (THPT QG, ĐGNL, …) kèm đáp án
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from rank_bm25 import BM25Okapi
from typing import Optional
import json
import os
import re

from app.config import get_settings

settings = get_settings()


class OpenAIV1EmbeddingFunction(EmbeddingFunction):
    """
    Custom ChromaDB EmbeddingFunction compatible with openai>=1.0.0.
    ChromaDB's built-in OpenAIEmbeddingFunction still uses the old API (openai.Embedding).
    """

    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model  = model_name

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        response = self._client.embeddings.create(
            input=input,
            model=self._model,
        )
        return [item.embedding for item in response.data]

# ── Valid data types ──────────────────────────────────────
VALID_THEORY_TYPES = {"theory"}
VALID_EXAM_TYPES   = {"exam_mcq", "exam_true_false", "exam_short_answer"}
ALL_VALID_TYPES    = VALID_THEORY_TYPES | VALID_EXAM_TYPES


class KnowledgeBase:
    """
    Manages two separate ChromaDB collections + BM25 indexes:
      • math_theory  — lý thuyết SGK
      • math_exams   — câu hỏi đề thi thật
    """

    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._embedding_fn = OpenAIV1EmbeddingFunction(
            api_key=settings.OPENAI_API_KEY,
            model_name=settings.EMBEDDING_MODEL,
        )

        # ── Collection 1: Lý thuyết ──────────────────────
        self.theory_collection = self.chroma_client.get_or_create_collection(
            name="math_theory",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_fn,
        )

        # ── Collection 2: Đề thi ─────────────────────────
        self.exam_collection = self.chroma_client.get_or_create_collection(
            name="math_exams",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_fn,
        )

        # BM25 indexes (rebuilt on each ingest call)
        self._theory_bm25: Optional[BM25Okapi] = None
        self._theory_store: list[dict] = []

        self._exam_bm25: Optional[BM25Okapi] = None
        self._exam_store: list[dict] = []

    # ── Ingest ───────────────────────────────────────────

    def ingest_theory(self, documents: list[dict]):
        """
        Ingest lý thuyết vào collection math_theory.
        Each doc: {"id": str, "content": str, "metadata": {"type": "theory", ...}}
        """
        self._ingest(documents, self.theory_collection, "_theory")

    def ingest_exams(self, documents: list[dict]):
        """
        Ingest câu hỏi đề thi vào collection math_exams.
        Each doc: {"id": str, "content": str, "metadata": {"type": "exam_mcq"|..., ...}}
        """
        self._ingest(documents, self.exam_collection, "_exam")

    def _ingest(self, documents: list[dict], collection, store_attr_prefix: str):
        if not documents:
            return

        ids       = [doc["id"]                    for doc in documents]
        contents  = [doc["content"]               for doc in documents]
        metadatas = [doc.get("metadata", {})      for doc in documents]

        # Upsert vào ChromaDB
        collection.upsert(ids=ids, documents=contents, metadatas=metadatas)

        # Build BM25
        tokenized = [self._tokenize(c) for c in contents]
        bm25      = BM25Okapi(tokenized)

        setattr(self, f"{store_attr_prefix}_store", documents)
        setattr(self, f"{store_attr_prefix}_bm25",  bm25)

    # ── Search ───────────────────────────────────────────

    def search_theory(self, query: str, k: int = 5, alpha: float = 0.6) -> list[dict]:
        """Hybrid search trong collection lý thuyết."""
        return self._hybrid_search(
            query, k, alpha,
            collection=self.theory_collection,
            bm25=self._theory_bm25,
            doc_store=self._theory_store,
        )

    def search_exams(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        year: Optional[int] = None,
        exam_source: Optional[str] = None,
        difficulty_part: Optional[int] = None,
        skill_id: Optional[str] = None,
        question_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Hybrid search trong collection đề thi với filter tùy chọn.

        Filters (ChromaDB where clause):
          year            — lọc theo năm (vd: 2024)
          exam_source     — lọc theo nguồn (vd: "THPT Quốc gia")
          difficulty_part — lọc theo phần (1=MCQ, 2=TF, 3=SA)
          skill_id        — lọc theo kỹ năng
          question_type   — lọc theo loại câu ("exam_mcq", "exam_true_false", …)
        """
        where: dict = {}
        if year            is not None: where["year"]            = year
        if exam_source     is not None: where["exam_source"]     = exam_source
        if difficulty_part is not None: where["difficulty_part"] = difficulty_part
        if skill_id        is not None: where["skill_id"]        = skill_id
        if question_type   is not None: where["type"]            = question_type

        return self._hybrid_search(
            query, k, alpha,
            collection=self.exam_collection,
            bm25=self._exam_bm25,
            doc_store=self._exam_store,
            where=where or None,
        )

    def search_all(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.55,
        boost_skill_id: Optional[str] = None,
        boost_chapter:  Optional[str] = None,
        skill_boost:    float = 0.15,
        chapter_boost:  float = 0.10,
    ) -> list[dict]:
        """
        Tìm kiếm gộp cả lý thuyết lẫn đề thi, sau đó re-rank qua Cross-Encoder.

        Pipeline:
          1. Hybrid search pool (k * CANDIDATE_MULTIPLIER docs)
          2. Soft Boost theo skill_id / chapter
          3. Cross-Encoder re-ranking → trả về top-k

        Falls back to hybrid_score ordering nếu reranker không khả dụng.
        """
        from app.rag.reranker import get_reranker  # lazy import tránh circular
        reranker = get_reranker()

        # Pool size: lấy nhiều hơn để reranker có đủ ứng viên
        pool_k = settings.RERANKER_CANDIDATE_K if settings.RERANKER_ENABLED else k * 2

        theory_results = self.search_theory(query, k=pool_k, alpha=alpha)
        exam_results   = self.search_exams(query,  k=pool_k, alpha=alpha)

        combined = theory_results + exam_results

        # ── Soft Boost ────────────────────────────────────────────────────────────
        if boost_skill_id or boost_chapter:
            boost_skill_id_lower   = (boost_skill_id  or "").lower().strip()
            boost_chapter_lower    = (boost_chapter   or "").lower().strip()

            for r in combined:
                meta = r.get("metadata", {})
                doc_skill   = str(meta.get("skill_id",  "")).lower().strip()
                doc_chapter = str(meta.get("chapter",   "")).lower().strip()

                bonus = 0.0
                if boost_skill_id_lower and doc_skill == boost_skill_id_lower:
                    bonus += skill_boost
                if boost_chapter_lower and doc_chapter == boost_chapter_lower:
                    bonus += chapter_boost

                if bonus > 0:
                    r["hybrid_score"] = min(r["hybrid_score"] + bonus, 1.0)
                    r["_boosted"] = True   # debug flag

        # ── Re-ranking (Cross-Encoder) ──────────────────────────────────────────
        return reranker.rerank(query, combined, k=k)

    def _hybrid_search(
        self,
        query: str,
        k: int,
        alpha: float,
        collection,
        bm25: Optional[BM25Okapi],
        doc_store: list[dict],
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Kết hợp vector search (ChromaDB) + keyword search (BM25).
        alpha=1 → thuần vector | alpha=0 → thuần BM25
        """
        results: list[dict] = []

        # ── Vector search ──────────────────────────────
        try:
            query_kwargs: dict = {
                "query_texts": [query],
                "n_results": min(k * 2, collection.count() or 1),
            }
            if where:
                query_kwargs["where"] = where

            vr = collection.query(**query_kwargs)
            for i, doc_id in enumerate(vr["ids"][0]):
                distance = vr["distances"][0][i] if vr.get("distances") else 0.0
                score    = max(1.0 - distance, 0.0)
                results.append({
                    "id":           doc_id,
                    "content":      vr["documents"][0][i],
                    "metadata":     vr["metadatas"][0][i] if vr.get("metadatas") else {},
                    "vector_score": score,
                    "bm25_score":   0.0,
                })
        except Exception:
            pass

        # ── BM25 keyword search ──────────────────────
        if bm25 and doc_store:
            tokens     = self._tokenize(query)
            bm25_scores = bm25.get_scores(tokens)
            max_bm25   = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

            for i, score in enumerate(bm25_scores):
                normalized = score / max_bm25
                doc        = doc_store[i]
                if where:
                    meta = doc.get("metadata", {})
                    if not all(meta.get(k) == v for k, v in where.items()):
                        continue
                existing = next((r for r in results if r["id"] == doc["id"]), None)
                if existing:
                    existing["bm25_score"] = normalized
                else:
                    results.append({
                        "id":           doc["id"],
                        "content":      doc["content"],
                        "metadata":     doc.get("metadata", {}),
                        "vector_score": 0.0,
                        "bm25_score":   normalized,
                    })

        # ── Combine ───────────────────────────────────
        for r in results:
            r["hybrid_score"] = alpha * r["vector_score"] + (1 - alpha) * r["bm25_score"]

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:k]

    # ── Utilities ────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """Vietnamese-aware tokenization."""
        text = text.lower()
        text = re.sub(
            r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]",
            " ", text,
        )
        return text.split()

    def get_theory_count(self) -> int:
        return self.theory_collection.count()

    def get_exam_count(self) -> int:
        return self.exam_collection.count()

    def get_stats(self) -> dict:
        return {
            "theory_docs": self.get_theory_count(),
            "exam_docs":   self.get_exam_count(),
            "total":       self.get_theory_count() + self.get_exam_count(),
        }


# ── Singleton ─────────────────────────────────────────────
_kb: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb
