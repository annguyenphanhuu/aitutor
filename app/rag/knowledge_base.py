"""Theory-only RAG knowledge base backed by ChromaDB and BM25.

The vector store contains only curriculum theory and formulas.
Exam questions remain available as JSON files for evaluation,
but are never embedded or retrieved for tutoring prompts.

RAG Flow:
  1. QueryExpander (LLM) phân tích câu hỏi → chapters, skills, formulas
  2. search_theory() → Hybrid Search (Vector + BM25, no metadata filter)
  3. Soft boost nhẹ dựa trên QueryExpander output (0.06/0.04)
  4. Reranker (optional) → top-k chunks
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from app.config import get_settings

logger = logging.getLogger(__name__)
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


VALID_THEORY_TYPES = {"theory", "formula"}


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

        # Auto-load in-memory BM25 from persisted ChromaDB
        self._init_bm25_from_chroma()

    # ── Ingest ────────────────────────────────────────────────────────────────

    def ingest_theory(self, documents: list[dict]) -> None:
        """Ingest theory/formula documents into the math_theory collection."""
        valid_docs = [
            doc for doc in documents
            if doc.get("metadata", {}).get("type", "theory") in VALID_THEORY_TYPES
        ]
        self._ingest(valid_docs, self.theory_collection, "_theory")

    def _init_bm25_from_chroma(self) -> None:
        """Fetch all documents from ChromaDB and build the in-memory BM25 index.

        This ensures BM25 works even when running scripts that don't call
        ingest_theory() first (e.g. run_eval_monitor.py).
        """
        try:
            count = self.theory_collection.count()
            if count == 0:
                return

            result = self.theory_collection.get(include=["documents", "metadatas"])
            if not result or not result.get("ids"):
                return

            docs = []
            for i, doc_id in enumerate(result["ids"]):
                docs.append({
                    "id": doc_id,
                    "content": result["documents"][i],
                    "metadata": result["metadatas"][i] if result.get("metadatas") else {}
                })

            tokenized = [self._tokenize(doc["content"]) for doc in docs]
            self._theory_bm25 = BM25Okapi(tokenized)
            self._theory_store = docs
            logger.debug("BM25 index loaded from ChromaDB: %d documents", len(docs))
        except Exception as e:
            logger.warning("Failed to init BM25 from ChromaDB: %s", e)

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

        # Rebuild BM25 from ALL documents in the collection (not just new ones)
        self._init_bm25_from_chroma()

    # ── Search ────────────────────────────────────────────────────────────────

    def search_theory(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.6,
    ) -> list[dict]:
        """Hybrid search over theory — no metadata filtering.

        Metadata filtering is deliberately removed. The QueryExpander + soft
        boost in the calling layer handles topic routing instead.
        """
        return self._hybrid_search(
            query=query,
            k=k,
            alpha=alpha,
            collection=self.theory_collection,
            bm25=self._theory_bm25,
            doc_store=self._theory_store,
        )

    def search_all(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.6,
    ) -> list[dict]:
        """Backward-compatible alias for the now theory-only knowledge base."""
        return self.search_theory(query=query, k=k, alpha=alpha)

    def get_formulas_by_ids(self, formula_ids: list[str]) -> list[dict]:
        """Return formula registry entries by id."""
        from app.rag.formula_registry import get_formulas_by_ids

        return get_formulas_by_ids(formula_ids)

    # ── Hybrid Search Core ────────────────────────────────────────────────────

    def _hybrid_search(
        self,
        query: str,
        k: int,
        alpha: float,
        collection,
        bm25: Optional[BM25Okapi],
        doc_store: list[dict],
    ) -> list[dict]:
        results: list[dict] = []

        # Vector search
        try:
            n_results = min(k * 3, collection.count() or 1)
            vector_results = collection.query(
                query_texts=[query],
                n_results=n_results,
            )
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

        # BM25 search
        if bm25 and doc_store:
            tokens = self._tokenize(query)
            bm25_scores = bm25.get_scores(tokens)
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

            for i, score in enumerate(bm25_scores):
                normalized = score / max_bm25
                doc = doc_store[i]

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

        # Compute hybrid score
        for result in results:
            result["hybrid_score"] = (
                alpha * result["vector_score"] + (1 - alpha) * result["bm25_score"]
            )

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:k]

    # ── Utilities ─────────────────────────────────────────────────────────────

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

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_theory_count(self) -> int:
        return self.theory_collection.count()

    def get_stats(self) -> dict:
        theory_count = self.get_theory_count()
        return {
            "theory_docs": theory_count,
            "total": theory_count,
        }


_kb: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb
