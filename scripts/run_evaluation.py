"""
Master evaluation script — build dataset + chạy RAGAS + in report.

Chạy:
    # Evaluate toàn bộ (build mới + evaluate):
    python scripts/run_evaluation.py --build

    # Chỉ chạy trên dataset đã có:
    python scripts/run_evaluation.py

    # Chỉ chạy 10 câu, một số metric:
    python scripts/run_evaluation.py --build --limit 10 --metrics cp,cr,f

    # Chỉ xem hallucination report (không cần OpenAI call):
    python scripts/run_evaluation.py --hallucination-only
"""

from __future__ import annotations

import asyncio
import sys
import json
import logging
import argparse
from pathlib import Path

# Đảm bảo import được app package
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║        AITutor — RAGAS Evaluation Pipeline           ║
║  context_precision | context_recall | faithfulness   ║
║  answer_relevancy  | answer_correctness              ║
╚══════════════════════════════════════════════════════╝
""")


def print_quick_stats(samples: list[dict]) -> None:
    """In thông tin cơ bản về dataset."""
    types   = {}
    skills  = {}
    for s in samples:
        meta = s.get("_meta", {})
        t = meta.get("type",     "unknown")
        k = meta.get("skill_id", "unknown")
        types[t]  = types.get(t, 0) + 1
        skills[k] = skills.get(k, 0) + 1

    print(f"  📊 Dataset: {len(samples)} samples")
    print(f"  📋 By type:  {json.dumps(types, ensure_ascii=False)}")
    print(f"  📚 By skill: {len(skills)} unique skills")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main(args):
    print_banner()

    # ── Mode 1: Hallucination-only (không cần build/evaluate) ──────────────
    if args.hallucination_only:
        print("🔍 Hallucination Tracker Report (from live runtime):\n")
        from app.evaluation.hallucination_tracker import ReflectionMetrics
        ReflectionMetrics.print_report()
        return

    # ── Mode 2: Build dataset nếu cần ──────────────────────────────────────
    dataset_path = args.dataset
    if args.build or not Path(dataset_path).exists():
        print(f"\n🔨 Building RAGAS dataset (limit={args.limit})...")
        from app.evaluation.dataset_builder import (
            load_exam_questions,
            build_ragas_samples,
            save_samples,
        )

        questions = load_exam_questions(
            exam_dir=args.exam_dir,
            limit=args.limit,
        )
        if not questions:
            print("❌ Không có câu hỏi nào được load. Kiểm tra đường dẫn exam_dir.")
            return

        print(f"  Loaded {len(questions)} questions.")
        samples = await build_ragas_samples(questions, k=args.k)
        save_samples(samples, dataset_path)
        print(f"  ✅ Dataset saved → {dataset_path}")
    else:
        from app.evaluation.dataset_builder import load_samples
        samples = load_samples(dataset_path)
        print(f"\n📂 Loaded existing dataset: {dataset_path}")

    print_quick_stats(samples)

    # ── Mode 3: Run RAGAS evaluation ──────────────────────────────────────
    print(f"\n🚀 Running RAGAS evaluation...")

    metric_map = {
        "cp":  "context_precision",
        "cr":  "context_recall",
        "f":   "faithfulness",
        "ar":  "answer_relevancy",
        "ac":  "answer_correctness",
    }
    if args.metrics:
        requested = [metric_map.get(m.strip(), m.strip()) for m in args.metrics.split(",")]
    else:
        requested = None  # all 5

    from app.evaluation.ragas_evaluator import RAGASEvaluator, _print_report

    evaluator = RAGASEvaluator(
        metrics=requested,
        model=args.judge_model,
    )

    report = evaluator.run_and_report(
        samples=samples,
        output_path=args.output,
    )

    _print_report(report)

    # ── Bonus: Hallucination tracker summary ────────────────────────────────
    print("\n🔍 Hallucination Tracker (from ReflectionEngine):")
    from app.evaluation.hallucination_tracker import ReflectionMetrics
    ReflectionMetrics.print_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AITutor RAGAS Evaluation Pipeline")

    parser.add_argument("--dataset",     default="data/ragas_dataset.json",  help="Dataset JSON path")
    parser.add_argument("--output",      default="data/ragas_report.json",   help="Output report JSON path")
    parser.add_argument("--exam-dir",    default="data/exams",               help="Exam JSON directory")
    parser.add_argument("--build",       action="store_true",                help="Rebuild dataset từ đầu")
    parser.add_argument("--limit",       type=int, default=20,               help="Số câu khi build dataset")
    parser.add_argument("--k",           type=int, default=5,                help="Số chunks RAG retrieve")
    parser.add_argument(
        "--metrics",
        default=None,
        help=(
            "Comma-separated: cp,cr,f,ar,ac "
            "(context_precision,context_recall,faithfulness,answer_relevancy,answer_correctness)"
        ),
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="LLM judge model (mặc định gpt-4o-mini để tiết kiệm chi phí)",
    )
    parser.add_argument(
        "--hallucination-only",
        action="store_true",
        help="Chỉ in hallucination report (không cần OpenAI call)",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
