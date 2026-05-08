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

question = """Câu 15. Một kho hàng có 85% sản phẩm loại I và 15% sản phẩm loại II, trong đó có 1% sản phẩm loại I bị hỏng, 4% sản phẩm loại II bị hỏng. Các sản phẩm có kích thước và hình dạng như nhau. Một khách hàng chọn ngẫu nhiên 1 sản phẩm. Xét các biến cố:
A: “Khách hàng chọn được sản phẩm loại I”
B: “Khách hàng chọn được sản phẩm không bị hỏng”."""

k = 5
pool_k = max(settings.RERANKER_CANDIDATE_K, k * 4)

expansion = expand_question(question)
candidates = kb.search_theory(question, k=pool_k, alpha=0.55)

def _norm_ch(s): return s.replace('–', '-').replace('—', '-').lower().strip()

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

# Use reranker logic to print out all scores
from sentence_transformers import CrossEncoder
ce = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", max_length=512)
pairs = [(question, c["content"]) for c in candidates]
scores = ce.predict(pairs, show_progress_bar=False)

for cand, score in zip(candidates, scores):
    cand["rerank_score"] = float(score)

rs_values = [c["rerank_score"] for c in candidates]
rs_min, rs_max = min(rs_values), max(rs_values)
rs_range = rs_max - rs_min if rs_max != rs_min else 1.0

RERANK_WEIGHT = 0.6
HYBRID_WEIGHT = 0.4

print(f"rs_min={rs_min:.3f}, rs_max={rs_max:.3f}, rs_range={rs_range:.3f}")
for cand in candidates:
    rs_norm = (cand["rerank_score"] - rs_min) / rs_range
    cand["rs_norm"] = rs_norm
    cand["final_score"] = (
        RERANK_WEIGHT * rs_norm + 
        HYBRID_WEIGHT * cand.get("hybrid_score", 0.0)
    )

ranked = sorted(candidates, key=lambda x: x["final_score"], reverse=True)

for i, r in enumerate(ranked[:10]):
    print(f"[{i+1}] id={r['id']:<20} final={r['final_score']:<5.3f} (hybrid={r['hybrid_score']:<5.3f} * 0.4 = {r['hybrid_score']*0.4:<5.3f}) + (rerank_norm={r['rs_norm']:<5.3f} * 0.6 = {r['rs_norm']*0.6:<5.3f})")
    print(f"    raw_rerank={r['rerank_score']:.3f} chapter={r['metadata'].get('chapter')}")
