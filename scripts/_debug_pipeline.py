# -*- coding: utf-8 -*-
"""Debug: full retrieve pipeline trace for probability question."""
import sys
import io
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
raw_pool_k = settings.RERANKER_CANDIDATE_K if settings.RERANKER_ENABLED else k * 2
pool_k = max(raw_pool_k, k * 4)

print(f"=== CONFIG ===")
print(f"  RERANKER_ENABLED: {settings.RERANKER_ENABLED}")
print(f"  RERANKER_CANDIDATE_K: {settings.RERANKER_CANDIDATE_K}")
print(f"  raw_pool_k: {raw_pool_k}")
print(f"  pool_k (after fix): {pool_k}")

expansion = expand_question(question)
print(f"\n=== EXPANSION ===")
print(f"  chapters: {expansion.chapters}")
print(f"  skill_ids: {expansion.skill_ids}")

candidates = kb.search_theory(question, k=pool_k, alpha=0.55)
print(f"\n=== CANDIDATES (before boost, k={pool_k}) ===")
for i, r in enumerate(candidates):
    print(f"  [{i+1:2d}] id={r['id']:<25} hybrid={r['hybrid_score']:.3f}  vec={r['vector_score']:.3f}  bm25={r['bm25_score']:.3f}")
    print(f"          chapter={r['metadata'].get('chapter','?')} skill={r['metadata'].get('skill_id','?')}")

# Apply boosts
def _norm_ch(s):
    return s.replace('\u2013', '-').replace('\u2014', '-').lower().strip()

SKILL_BOOST = 0.12
CHAPTER_BOOST = 0.07
expanded_skills = {s.lower() for s in expansion.skill_ids}
expanded_chapters = {_norm_ch(c) for c in expansion.chapters}

print(f"\n=== BOOST PARAMS ===")
print(f"  expanded_skills:   {expanded_skills}")
print(f"  expanded_chapters: {expanded_chapters}")

for r in candidates:
    bonus = 0.0
    meta = r.get("metadata", {})
    chunk_skill   = str(meta.get("skill_id", "")).lower().strip()
    chunk_chapter = _norm_ch(str(meta.get("chapter", "")))
    
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

print(f"\n=== CANDIDATES (after boost) ===")
candidates_sorted = sorted(candidates, key=lambda x: x["hybrid_score"], reverse=True)
for i, r in enumerate(candidates_sorted[:15]):
    boosted = " BOOSTED" if r.get("_boosted") else ""
    print(f"  [{i+1:2d}] id={r['id']:<25} hybrid={r['hybrid_score']:.3f}{boosted}")
    print(f"          chapter={r['metadata'].get('chapter','?')} skill={r['metadata'].get('skill_id','?')}")

print(f"\n=== AFTER RERANKER (top-{k}) ===")
ranked = reranker.rerank(question, candidates, k=k)
for i, r in enumerate(ranked):
    rs = r.get("rerank_score", "N/A")
    boosted = " BOOSTED" if r.get("_boosted") else ""
    print(f"  [{i+1}] id={r['id']:<25} hybrid={r['hybrid_score']:.3f}  rerank={rs:.3f if isinstance(rs, float) else rs}{boosted}")
    print(f"        chapter={r['metadata'].get('chapter','?')} skill={r['metadata'].get('skill_id','?')}")
