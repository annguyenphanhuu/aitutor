"""Theory-only RAG knowledge base backed by ChromaDB and BM25.

The vector store intentionally contains only curriculum theory. Exam questions
remain available to quiz/evaluation services as JSON files, but they are not
embedded or retrieved for tutoring prompts.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from app.config import get_settings

settings = get_settings()


class OpenAIV1EmbeddingFunction(EmbeddingFunction):
    """ChromaDB embedding function compatible with openai>=1.0.0."""

    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model_name

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        response = self._client.embeddings.create(
            input=input,
            model=self._model,
        )
        return [item.embedding for item in response.data]


VALID_THEORY_TYPES = {"theory"}


class KnowledgeBase:
    """Manages the single theory collection and its in-memory BM25 index."""

    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._embedding_fn = OpenAIV1EmbeddingFunction(
            api_key=settings.OPENAI_API_KEY,
            model_name=settings.EMBEDDING_MODEL,
        )

        self.theory_collection = self.chroma_client.get_or_create_collection(
            name="math_theory",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_fn,
        )

        self._theory_bm25: Optional[BM25Okapi] = None
        self._theory_store: list[dict] = []

    def ingest_theory(self, documents: list[dict]) -> None:
        """Ingest theory documents into the math_theory collection."""
        theory_docs = [
            doc for doc in documents
            if doc.get("metadata", {}).get("type", "theory") in VALID_THEORY_TYPES
        ]
        self._ingest(theory_docs, self.theory_collection, "_theory")

    def _ingest(self, documents: list[dict], collection, store_attr_prefix: str) -> None:
        if not documents:
            return

        normalized_docs = []
        for doc in documents:
            normalized_docs.append({
                **doc,
                "metadata": self._sanitize_metadata(doc.get("metadata", {})),
            })

        ids = [doc["id"] for doc in normalized_docs]
        contents = [doc["content"] for doc in normalized_docs]
        metadatas = [doc.get("metadata", {}) for doc in normalized_docs]

        collection.upsert(ids=ids, documents=contents, metadatas=metadatas)

        tokenized = [self._tokenize(content) for content in contents]
        bm25 = BM25Okapi(tokenized)

        setattr(self, f"{store_attr_prefix}_store", normalized_docs)
        setattr(self, f"{store_attr_prefix}_bm25", bm25)

    def search_theory(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.6,
        skill_ids: Optional[list[str]] = None,
        chapter: Optional[str] = None,
    ) -> list[dict]:
        """Hybrid search over theory only.

        ``skill_ids`` and ``chapter`` are optional metadata filters. They are
        applied uniformly to vector and BM25 candidates.
        """
        where = self._build_simple_where(skill_ids=skill_ids, chapter=chapter)
        results = self._hybrid_search(
            query=query,
            k=k,
            alpha=alpha,
            collection=self.theory_collection,
            bm25=self._theory_bm25,
            doc_store=self._theory_store,
            where=where,
        )

        if skill_ids or chapter:
            results = [
                r for r in results
                if self._matches_metadata_filter(r.get("metadata", {}), skill_ids, chapter)
            ]
        return results[:k]

    def search_all(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.55,
        boost_skill_id: Optional[str] = None,
        boost_chapter: Optional[str] = None,
        skill_boost: float = 0.15,
        chapter_boost: float = 0.10,
    ) -> list[dict]:
        """Compatibility wrapper for the old unified search API.

        The returned pool now contains theory documents only. Existing callers
        can keep using ``search_all`` while the RAG layer avoids exam leakage.
        """
        from app.rag.reranker import get_reranker

        reranker = get_reranker()
        pool_k = settings.RERANKER_CANDIDATE_K if settings.RERANKER_ENABLED else k * 2

        candidates = self.search_theory(query, k=pool_k, alpha=alpha)
        self._apply_soft_boost(
            candidates,
            boost_skill_id=boost_skill_id,
            boost_chapter=boost_chapter,
            skill_boost=skill_boost,
            chapter_boost=chapter_boost,
        )
        return reranker.rerank(query, candidates, k=k)

    def get_formulas_by_ids(self, formula_ids: list[str]) -> list[dict]:
        """Return formula registry entries by id."""
        from app.rag.formula_registry import get_formulas_by_ids

        return get_formulas_by_ids(formula_ids)

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
        results: list[dict] = []

        try:
            query_kwargs: dict = {
                "query_texts": [query],
                "n_results": min(k * 2, collection.count() or 1),
            }
            if where:
                query_kwargs["where"] = where

            vector_results = collection.query(**query_kwargs)
            for i, doc_id in enumerate(vector_results["ids"][0]):
                distance = (
                    vector_results["distances"][0][i]
                    if vector_results.get("distances")
                    else 0.0
                )
                score = max(1.0 - distance, 0.0)
                results.append({
                    "id": doc_id,
                    "content": vector_results["documents"][0][i],
                    "metadata": (
                        vector_results["metadatas"][0][i]
                        if vector_results.get("metadatas")
                        else {}
                    ),
                    "vector_score": score,
                    "bm25_score": 0.0,
                })
        except Exception:
            pass

        if bm25 and doc_store:
            tokens = self._tokenize(query)
            bm25_scores = bm25.get_scores(tokens)
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

            for i, score in enumerate(bm25_scores):
                normalized = score / max_bm25
                doc = doc_store[i]
                if where and not self._matches_simple_where(doc.get("metadata", {}), where):
                    continue

                existing = next((r for r in results if r["id"] == doc["id"]), None)
                if existing:
                    existing["bm25_score"] = normalized
                else:
                    results.append({
                        "id": doc["id"],
                        "content": doc["content"],
                        "metadata": doc.get("metadata", {}),
                        "vector_score": 0.0,
                        "bm25_score": normalized,
                    })

        for result in results:
            result["hybrid_score"] = (
                alpha * result["vector_score"] + (1 - alpha) * result["bm25_score"]
            )

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:k]

    def _apply_soft_boost(
        self,
        results: list[dict],
        boost_skill_id: Optional[str],
        boost_chapter: Optional[str],
        skill_boost: float,
        chapter_boost: float,
    ) -> None:
        boost_skill = (boost_skill_id or "").lower().strip()
        boost_chapter_norm = (boost_chapter or "").lower().strip()

        if not boost_skill and not boost_chapter_norm:
            return

        for result in results:
            meta = result.get("metadata", {})
            bonus = 0.0

            if boost_skill and boost_skill in self._metadata_skill_ids(meta):
                bonus += skill_boost

            doc_chapter = str(meta.get("chapter", "")).lower().strip()
            if boost_chapter_norm and doc_chapter == boost_chapter_norm:
                bonus += chapter_boost

            if bonus > 0:
                result["hybrid_score"] = min(result.get("hybrid_score", 0.0) + bonus, 1.0)
                result["_boosted"] = True

    def _build_simple_where(
        self,
        skill_ids: Optional[list[str]],
        chapter: Optional[str],
    ) -> Optional[dict]:
        where: dict = {}
        if skill_ids and len(skill_ids) == 1:
            where["skill_id"] = skill_ids[0]
        if chapter:
            where["chapter"] = chapter
        return where or None

    def _matches_simple_where(self, metadata: dict, where: dict) -> bool:
        return all(metadata.get(key) == value for key, value in where.items())

    def _matches_metadata_filter(
        self,
        metadata: dict,
        skill_ids: Optional[list[str]],
        chapter: Optional[str],
    ) -> bool:
        if chapter and str(metadata.get("chapter", "")).strip() != chapter:
            return False
        if skill_ids:
            wanted = {skill_id.lower().strip() for skill_id in skill_ids if skill_id}
            if wanted and not (wanted & self._metadata_skill_ids(metadata)):
                return False
        return True

    def _metadata_skill_ids(self, metadata: dict) -> set[str]:
        raw_values = []
        if metadata.get("skill_id"):
            raw_values.append(metadata["skill_id"])
        if metadata.get("skill_ids"):
            raw_values.append(metadata["skill_ids"])

        skill_ids: set[str] = set()
        for raw in raw_values:
            if isinstance(raw, list):
                values = raw
            elif isinstance(raw, str):
                try:
                    decoded = json.loads(raw)
                    values = decoded if isinstance(decoded, list) else [raw]
                except json.JSONDecodeError:
                    values = [part.strip() for part in raw.split(",")]
            else:
                values = [raw]

            for value in values:
                value_str = str(value).lower().strip()
                if value_str:
                    skill_ids.add(value_str)
        return skill_ids

    def _sanitize_metadata(self, metadata: dict) -> dict:
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, bool):
                sanitized[key] = int(value)
            elif value is None:
                sanitized[key] = ""
            elif isinstance(value, (list, dict)):
                sanitized[key] = json.dumps(value, ensure_ascii=False)
            else:
                sanitized[key] = value
        return sanitized

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        return text.split()

    def get_theory_count(self) -> int:
        return self.theory_collection.count()

    def get_exam_count(self) -> int:
        """Compatibility metric: exams are no longer stored in vector DB."""
        return 0

    def get_stats(self) -> dict:
        theory_count = self.get_theory_count()
        return {
            "theory_docs": theory_count,
            "exam_docs": 0,
            "total": theory_count,
        }


_kb: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb
