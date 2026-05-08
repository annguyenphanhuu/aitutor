# -*- coding: utf-8 -*-
"""Debug reranker scores for probability question."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from app.rag.knowledge_base import get_knowledge_base
from app.rag.query_expander import expand_question
from app.rag.reranker import get_reranker
from app.config import get_settings

settings = get_settings()
kb = get_knowledge_base()
reranker = get_reranker()

question = """Cau 15. Mot kho hang co 85% san pham loai I va 15% san pham loai II, trong do co 1% san pham loai I bi hong, 4% san pham loai II bi hong. Xet cac bien co: A: Khach hang chon duoc san pham loai I, B: Khach hang chon duoc san pham khong bi hong."""

k = 5
pool_k = max(settings.RERANKER_CANDIDATE_K, k * 4)
expansion = expand_question(question)
candidates = kb.search_theory(question, k=pool_k, alpha=0.55)

def _norm_ch(s):
    return s.replace('\u2013', '-').replace('\u2014', '-').lower().strip()

SKILL_BOOST = 0.12
CHAPTER_BOOST = 0.07
expanded_skills = {s.lower() for s in expansion.skill_ids}
expanded_chapters = {_norm_ch(c) for c in expansion.chapters}

for r in candidates:
    bonus = 0.0
    meta = r.get("metadata", {})
    chunk_skill = str(meta.get("skill_id", "")).lower().strip()
    chunk_chapter = _norm_ch(str(meta.get("chapter", "")))
    if chunk_skill and chunk_skill in expanded_skills:
        bonus += SKILL_BOOST
    if chunk_chapter and any(chunk_chapter in exp_ch or exp_ch in chunk_chapter for exp_ch in expanded_chapters):
        bonus += CHAPTER_BOOST
    if bonus > 0:
        r["hybrid_score"] = min(r.get("hybrid_score", 0) + bonus, 1.0)
        r["_boosted"] = True

# Use reranker directly to see scores
from sentence_transformers import CrossEncoder
ce = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", max_length=512)
pairs = [(question, c["content"]) for c in candidates]
scores = ce.predict(pairs, show_progress_bar=False)
for cand, score in zip(candidates, scores):
    cand["rerank_score"] = float(score)

# Sort by rerank
ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
print("=== RERANK SCORES (all candidates) ===")
for i, r in enumerate(ranked[:20]):
    boosted = " BOOSTED" if r.get("_boosted") else ""
    top5 = " <<< TOP-5" if i < 5 else ""
    print(f"  [{i+1:2d}] id={r['id']:<25} hybrid={r['hybrid_score']:.3f}  rerank={r['rerank_score']:.3f}{boosted}{top5}")
    print(f"          chapter={r['metadata'].get('chapter','?')}")
