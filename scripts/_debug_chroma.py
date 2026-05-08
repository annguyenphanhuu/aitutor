# -*- coding: utf-8 -*-
"""Debug: inspect ChromaDB theory collection metadata."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from app.rag.knowledge_base import get_knowledge_base
from app.rag.query_expander import expand_question

kb = get_knowledge_base()
result = kb.theory_collection.get(include=["documents", "metadatas"])

print("=== ALL THEORY CHUNKS IN CHROMADB ===")
prob_ids = []
for doc_id, meta, doc in zip(result["ids"], result["metadatas"], result["documents"]):
    chapter = meta.get("chapter", "?")
    skill   = meta.get("skill_id", "?")
    line = f"  {doc_id}: chapter='{chapter}' | skill='{skill}'"
    print(line)
    if any(kw in chapter.lower() for kw in ["xac", "suat", "hop"]):
        prob_ids.append(doc_id)

print(f"\n=== PROBABILITY-RELATED CHUNKS: {prob_ids} ===")
for doc_id, meta, doc in zip(result["ids"], result["metadatas"], result["documents"]):
    if doc_id in prob_ids:
        print(f"\n  ID: {doc_id}")
        print(f"  chapter: {meta.get('chapter', 'MISSING')}")
        print(f"  skill_id: {meta.get('skill_id', 'MISSING!')}")
        print(f"  content[:200]: {doc[:200]}")

# Test hybrid search for the probability question
print("\n=== HYBRID SEARCH TEST ===")
results = kb.search_theory("Xac suat co dieu kien Bayes cong thuc xac suat toan phan", k=5, alpha=0.55)
for i, r in enumerate(results):
    print(f"  [{i+1}] id={r['id']} | hybrid={r['hybrid_score']:.3f} | vec={r['vector_score']:.3f} | bm25={r['bm25_score']:.3f}")
    print(f"       chapter={r['metadata'].get('chapter','?')} | skill={r['metadata'].get('skill_id','?')}")
    print(f"       content: {r['content'][:100]}")

# Test QueryExpander
print("\n=== QUERY EXPANDER TEST ===")
question = """Cau 15. Mot kho hang co 85% san pham loai I va 15% san pham loai II, trong do co 1% san pham loai I bi hong, 4% san pham loai II bi hong. Xet cac bien co: A: Khach hang chon duoc san pham loai I, B: Khach hang chon duoc san pham khong bi hong."""
exp = expand_question(question)
print(f"  chapters   : {exp.chapters}")
print(f"  skill_ids  : {exp.skill_ids}")
print(f"  formula_ids: {exp.formula_ids}")
print(f"  reasoning  : {exp.reasoning}")
print(f"  error      : {exp.error}")
