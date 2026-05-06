"""Cross-Encoder Re-ranker — nâng chất lượng RAG retrieval.

Pipeline:
    Hybrid Search (k=20 candidates)
        ↓
    CrossEncoder.predict([(query, doc), ...])
        ↓
    Sort by cross-encoder score → top-k=5

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Offline, ~68MB
    - Hỗ trợ tốt tiếng Việt (multilingual BERT core)
    - Latency: ~20ms/batch trên CPU

Fallback: nếu sentence-transformers chưa cài hoặc RERANKER_ENABLED=False
    → trả lại candidates theo hybrid_score gốc (không thay đổi behavior).
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class Reranker:
    """Local Cross-Encoder re-ranker wrapper.

    Usage:
        reranker = get_reranker()
        top_docs = reranker.rerank(query, candidates, k=5)
    """

    def __init__(self):
        self._model = None
        self._model_loaded = False
        self._load_attempted = False

    def _load_model(self) -> bool:
        """Lazy-load the cross-encoder model on first use."""
        if self._load_attempted:
            return self._model_loaded

        self._load_attempted = True

        if not settings.RERANKER_ENABLED:
            logger.info("⚡ Reranker disabled via config (RERANKER_ENABLED=False)")
            return False

        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            logger.info("📦 Loading CrossEncoder model: %s", model_name)
            self._model = CrossEncoder(model_name, max_length=512)
            self._model_loaded = True
            logger.info("✅ CrossEncoder model loaded successfully")
        except ImportError:
            logger.warning(
                "⚠️  sentence-transformers chưa được cài. "
                "Reranker bị tắt (fallback: hybrid score). "
                "Cài bằng: pip install sentence-transformers"
            )
        except Exception as e:
            logger.warning("⚠️  Không tải được CrossEncoder model: %s. Fallback to hybrid score.", e)

        return self._model_loaded

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        k: int | None = None,
    ) -> list[dict]:
        """Re-rank candidates using cross-encoder scores.

        Parameters
        ----------
        query : str
            The student's search query.
        candidates : list[dict]
            Each dict must have 'content' and 'hybrid_score' keys.
            (Output cua knowledge_base.search_all / search_theory)
        k : int | None
            Number of top results to return.
            Defaults to settings.RERANKER_TOP_K.

        Returns
        -------
        list[dict]
            Re-ranked candidates, each enriched with 'rerank_score'.
            Falls back to hybrid_score ordering if model unavailable.
        """
        top_k = k or settings.RERANKER_TOP_K

        if not candidates:
            return candidates

        if not self._load_model() or self._model is None:
            # Graceful fallback: return by hybrid_score as before
            return sorted(candidates, key=lambda x: x.get("hybrid_score", 0), reverse=True)[:top_k]

        try:
            # Build (query, passage) pairs
            pairs = [(query, c["content"]) for c in candidates]

            # Predict cross-encoder scores (batch)
            scores = self._model.predict(pairs, show_progress_bar=False)

            # Attach scores & sort
            for cand, score in zip(candidates, scores):
                cand["rerank_score"] = float(score)

            ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
            top = ranked[:top_k]

            logger.debug(
                "🔍 Reranker: %d candidates → top-%d (score range: %.3f–%.3f)",
                len(candidates),
                top_k,
                top[-1]["rerank_score"] if top else 0,
                top[0]["rerank_score"] if top else 0,
            )
            return top

        except Exception as e:
            logger.warning("⚠️  CrossEncoder predict failed: %s. Fallback to hybrid score.", e)
            return sorted(candidates, key=lambda x: x.get("hybrid_score", 0), reverse=True)[:top_k]

    @property
    def is_available(self) -> bool:
        """Returns True if the reranker model is loaded and ready."""
        return self._load_model()


# ── Singleton ─────────────────────────────────────────────────────────────────
_reranker: Optional[Reranker] = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
