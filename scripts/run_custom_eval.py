import asyncio
import json
from pathlib import Path
import sys

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.evaluation.dataset_builder import build_ragas_samples
from app.evaluation.ragas_evaluator import RAGASEvaluator, _print_report
from app.evaluation.math_judge import MathJudge

async def run_custom_eval():
    exam_files = [
        "data/exams/de7plus_de01.json",
        "data/exams/de7plus_de03.json",
        "data/exams/de7plus_de04.json",
        "data/exams/de7plus_de05.json",
    ]
    
    questions = []
    for f in exam_files:
        items = json.loads(Path(f).read_text(encoding="utf-8"))
        valid_items = []
        for item in items:
            meta = item.get("metadata", {})
            if meta.get("type") not in ["exam_mcq", "exam_short_answer", "exam_true_false"]:
                continue
            if meta.get("has_image") and not item.get("answer"):
                continue
            valid_items.append({
                "id":            item.get("id", ""),
                "question":      item.get("content", ""),
                "ground_truth":  item.get("answer", ""),
                "skill_id":      meta.get("skill_id", ""),
                "chapter":       meta.get("chapter", ""),
                "type":          meta.get("type", ""),
                "correct_answer": meta.get("correct_answer", ""),
                "has_image":     meta.get("has_image", False),
                "image_path":    meta.get("image_path", ""),
            })
        questions.extend(valid_items[:12]) # Take first 12
        
    print(f"Tổng số câu hỏi được load: {len(questions)}")
    
    # Re-ingest theory exclusively to ensure clean RAG logic as per improve.md
    print("Xóa bỏ data cũ trong VectorDB và Ingesting theory.json...")
    from app.rag.knowledge_base import get_knowledge_base
    kb = get_knowledge_base()
    
    # Xóa collection exams để thanh lọc data (Bước 1 trong improve.md)
    try:
        kb.client.delete_collection("math_exams")
        print("Đã xóa collection 'math_exams'.")
    except Exception:
        pass

    theory_items = json.loads(Path("data/theory.json").read_text(encoding="utf-8"))
    valid_theory = []
    for doc in theory_items:
        safe_meta = {}
        for k, v in doc.get("metadata", {}).items():
            safe_meta[k] = int(v) if isinstance(v, bool) else (v if v is not None else "")
        doc["metadata"] = safe_meta
        valid_theory.append(doc)
    
    # Ingest lý thuyết
    kb.ingest_theory(valid_theory)
    print(f"Đã ingest {len(valid_theory)} documents lý thuyết vào ChromaDB.")
    
    print(f"Building RAGAS samples for {len(questions)} questions...")
    samples = await build_ragas_samples(questions, k=5)
    
    print("Running RAGAS evaluation...")
    evaluator = RAGASEvaluator(model="gpt-4o-mini")
    ragas_report = evaluator.run_and_report(samples, output_path="data/ragas_report_4_exams.json")
    
    print("Running MathJudge evaluation...")
    judge = MathJudge(model="gpt-4o-mini", skip_explanation=True)
    math_report = judge.run_and_report(samples, output_path="data/math_report_4_exams.json")
    
    print("Generating Combined Report...")
    from app.evaluation.combined_report import merge_reports, print_combined_report
    combined = merge_reports(ragas_report, math_report, output_path="data/combined_report_4_exams.json")
    print_combined_report(combined)

if __name__ == "__main__":
    asyncio.run(run_custom_eval())
