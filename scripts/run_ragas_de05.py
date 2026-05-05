"""
RAGAS Evaluation: Ingest theory + de01 + de03 + de04 → evaluate de05.

Pipeline:
  1. Ingest theory.json vào ChromaDB (math_theory collection)
  2. Ingest de7plus_de01.json + de7plus_de03.json + de7plus_de04.json vào ChromaDB (math_exams)
  3. Load câu hỏi từ de7plus_de05.json làm test set
  4. Build RAGAS samples (retrieve context + generate answer)
  5. Run RAGAS evaluation (5 metrics)
  6. In báo cáo + lưu JSON

Chạy:
    python scripts/run_ragas_de05.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import argparse
import sys
import os
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── File paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
EXAMS_DIR    = PROJECT_ROOT / "data" / "exams"
THEORY_FILE  = PROJECT_ROOT / "data" / "theory.json"
INGEST_FILES = [
    EXAMS_DIR / "de7plus_de01.json",
    EXAMS_DIR / "de7plus_de03.json",
    EXAMS_DIR / "de7plus_de04.json",
]
TEST_FILE    = EXAMS_DIR / "de7plus_de05.json"
DATASET_PATH          = str(PROJECT_ROOT / "data" / "ragas_de05_dataset.json")
REPORT_PATH           = str(PROJECT_ROOT / "data" / "ragas_de05_report.json")
MATH_JUDGE_REPORT_PATH = str(PROJECT_ROOT / "data" / "math_judge_de05_report.json")
COMBINED_REPORT_PATH   = str(PROJECT_ROOT / "data" / "combined_de05_report.json")


# ── Banner ─────────────────────────────────────────────────────────────────────

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║        AITutor — RAGAS Evaluation: Đề 7+ (De05)             ║
║  KB: theory + de01 + de03 + de04  →  Test: de05             ║
║  context_precision | context_recall | faithfulness           ║
║  answer_relevancy  | answer_correctness                      ║
╚══════════════════════════════════════════════════════════════╝
""")


# ── Step 1a: Ingest Theory ────────────────────────────────────────────────────

def ingest_theory_kb(theory_file: Path) -> None:
    """Ingest theory.json into ChromaDB math_theory collection."""
    from app.rag.knowledge_base import get_knowledge_base

    print("\n📚 STEP 1a: INGEST THEORY")
    print("=" * 55)

    if not theory_file.exists():
        print(f"  ⚠️  theory.json không tồn tại: {theory_file}")
        return

    try:
        items = json.loads(theory_file.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ❌ Lỗi đọc theory.json: {e}")
        return

    valid = []
    for doc in items:
        if not doc.get("id") or not doc.get("content"):
            continue
        meta = doc.get("metadata", {})
        safe_meta = {}
        for k, v in meta.items():
            if isinstance(v, bool):
                safe_meta[k] = int(v)
            elif v is None:
                safe_meta[k] = ""
            else:
                safe_meta[k] = v
        doc["metadata"] = safe_meta
        valid.append(doc)

    print(f"  📄 theory.json                      {len(valid)} entries hợp lệ")

    if not valid:
        print("  ⚠️  Không có theory entry nào để ingest.")
        return

    kb = get_knowledge_base()
    kb.ingest_theory(valid)

    stats = kb.get_stats()
    print(f"  ✅ math_theory : {stats['theory_docs']} docs")
    print(f"  ✅ math_exams  : {stats['exam_docs']} docs")


# ── Step 1b: Ingest Exams ──────────────────────────────────────────────────────

def ingest_kb(files: list[Path]) -> None:
    """Ingest specified exam JSON files into ChromaDB knowledge base."""
    from app.rag.knowledge_base import get_knowledge_base

    VALID_EXAM_TYPES = {"exam_mcq", "exam_true_false", "exam_short_answer"}

    print("\n📥 STEP 1: INGEST KNOWLEDGE BASE")
    print("=" * 55)

    all_docs = []
    for fpath in files:
        if not fpath.exists():
            print(f"  ❌ File không tồn tại: {fpath}")               
            continue

        try:
            items = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ❌ Lỗi đọc {fpath.name}: {e}")
            continue

        valid = []
        for doc in items:
            meta  = doc.get("metadata", {})
            dtype = meta.get("type", "")
            if not doc.get("id"):
                continue
            if not doc.get("content"):
                continue
            # Flatten metadata booleans/int for ChromaDB compatibility
            safe_meta = {}
            for k, v in meta.items():
                if isinstance(v, bool):
                    safe_meta[k] = int(v)  # ChromaDB không chấp nhận bool
                elif v is None:
                    safe_meta[k] = ""
                else:
                    safe_meta[k] = v
            doc["metadata"] = safe_meta
            if dtype in VALID_EXAM_TYPES:
                valid.append(doc)

        print(f"  📄 {fpath.name:<35} {len(valid)} câu hợp lệ")
        all_docs.extend(valid)

    if not all_docs:
        print("  ❌ Không có document nào để ingest. Kiểm tra lại file paths.")
        sys.exit(1)

    print(f"\n  🔄 Ingesting {len(all_docs)} documents vào math_exams...")
    kb = get_knowledge_base()
    kb.ingest_exams(all_docs)

    stats = kb.get_stats()
    print(f"  ✅ math_theory : {stats['theory_docs']} docs")
    print(f"  ✅ math_exams  : {stats['exam_docs']} docs")
    print(f"  ✅ Total       : {stats['total']} docs")


# ── Step 2: Load test questions ────────────────────────────────────────────────

def load_test_questions(
    test_file: Path,
    limit: int | None = None,
    offset: int = 0,
    question_types: list[str] | None = None,
) -> list[dict]:
    """Load questions from de05 as test set."""
    from app.evaluation.dataset_builder import load_exam_questions

    print(f"\n📋 STEP 2: LOAD TEST QUESTIONS")
    print("=" * 55)
    print(f"  📂 Test file: {test_file.name}")

    questions = load_exam_questions(
        exam_dir=str(test_file),     # load single file
        question_types=question_types,
        limit=None,                  # load all, then slice
    )

    # Apply offset + limit
    if offset:
        questions = questions[offset:]
    if limit is not None:
        questions = questions[:limit]

    if not questions:
        print("  ❌ Không load được câu hỏi nào từ de05. Kiểm tra file.")
        sys.exit(1)

    # Print breakdown
    by_type = {}
    for q in questions:
        t = q.get("type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    print(f"  ✅ Loaded {len(questions)} questions")
    for t, cnt in by_type.items():
        print(f"     • {t:<28} {cnt} câu")

    return questions


# ── Step 3: Build RAGAS samples ────────────────────────────────────────────────

async def build_samples(
    questions: list[dict],
    k: int = 5,
    dataset_path: str = DATASET_PATH,
) -> list[dict]:
    """Build RAGAS samples (retrieve + generate)."""
    from app.evaluation.dataset_builder import build_ragas_samples, save_samples

    print(f"\n🔨 STEP 3: BUILD RAGAS SAMPLES")
    print("=" * 55)
    print(f"  📊 {len(questions)} questions × top-{k} RAG chunks")

    samples = await build_ragas_samples(questions, k=k)
    save_samples(samples, dataset_path)

    print(f"  ✅ Built {len(samples)} samples → {dataset_path}")
    return samples


# ── Step 4: Run RAGAS evaluation ───────────────────────────────────────────────

def run_ragas(
    samples: list[dict],
    requested_metrics: list[str] | None,
    judge_model: str,
    report_path: str = REPORT_PATH,
) -> dict:
    """Run RAGAS evaluation and return report."""
    from app.evaluation.ragas_evaluator import RAGASEvaluator, _print_report

    print(f"\n🚀 STEP 4: RUN RAGAS EVALUATION")
    print("=" * 55)
    print(f"  🤖 Judge model: {judge_model}")
    print(f"  📐 Metrics: {requested_metrics or 'all (5)'}")

    evaluator = RAGASEvaluator(
        metrics=requested_metrics,
        model=judge_model,
    )

    report = evaluator.run_and_report(
        samples=samples,
        output_path=report_path,
    )

    _print_report(report)
    print(f"\n  💾 Report saved → {report_path}")
    return report


# ── Main ───────────────────────────────────────────────────────────────────────

async def main(args):
    print_banner()

    # Step 1: Ingest
    if not args.skip_ingest:
        ingest_theory_kb(THEORY_FILE)
        ingest_kb(INGEST_FILES)
    else:
        print("\n⏭️  STEP 1: Bỏ qua ingest (--skip-ingest)")

    # Step 2: Load test questions
    questions = load_test_questions(
        test_file=TEST_FILE,
        limit=args.limit,
        offset=args.offset,
    )

    # Step 3: Build or load dataset
    if args.load_dataset and Path(DATASET_PATH).exists():
        from app.evaluation.dataset_builder import load_samples
        samples = load_samples(DATASET_PATH)
        print(f"\n📂 STEP 3: Loaded existing dataset: {len(samples)} samples")
    else:
        samples = await build_samples(questions, k=args.k, dataset_path=DATASET_PATH)

    # Step 4: RAGAS evaluation
    METRIC_MAP = {
        "cp": "context_precision",
        "cr": "context_recall",
        "f":  "faithfulness",
        "ar": "answer_relevancy",
        "ac": "answer_correctness",
    }
    requested = None
    if args.metrics:
        requested = [METRIC_MAP.get(m.strip(), m.strip()) for m in args.metrics.split(",")]

    ragas_report = run_ragas(
        samples=samples,
        requested_metrics=requested,
        judge_model=args.judge_model,
        report_path=REPORT_PATH,
    )

    # Step 5: MathJudge evaluation
    if not args.skip_math_judge:
        from app.evaluation.math_judge import MathJudge, print_math_report

        print(f"\n🧮 STEP 5: MATH JUDGE EVALUATION")
        print("=" * 55)
        print(f"  📐 Mode: {'accuracy only (offline)' if args.skip_explanation else 'accuracy + explanation quality'}")

        math_judge = MathJudge(
            model=args.judge_model,
            skip_explanation=args.skip_explanation,
        )
        math_report = math_judge.run_and_report(
            samples=samples,
            output_path=MATH_JUDGE_REPORT_PATH,
        )
        print_math_report(math_report)
        print(f"  💾 Math report saved → {MATH_JUDGE_REPORT_PATH}")

        # Step 6: Combined report
        from app.evaluation.combined_report import merge_reports, print_combined_report

        print(f"\n📊 STEP 6: COMBINED REPORT")
        print("=" * 55)
        combined = merge_reports(
            ragas_report=ragas_report,
            math_report=math_report,
            output_path=COMBINED_REPORT_PATH,
        )
        print_combined_report(combined)
        print(f"  💾 Combined report saved → {COMBINED_REPORT_PATH}")
    else:
        print("\n⏭️  STEP 5-6: Bỏ qua MathJudge (--skip-math-judge)")

    print("\n✅ Đánh giá hoàn tất!")
    print(f"   Dataset         : {DATASET_PATH}")
    print(f"   RAGAS Report    : {REPORT_PATH}")
    if not args.skip_math_judge:
        print(f"   MathJudge Report: {MATH_JUDGE_REPORT_PATH}")
        print(f"   Combined Report : {COMBINED_REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAGAS Evaluation: Ingest de01+de03+de04 → Evaluate de05",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Bỏ qua bước ingest (khi KB đã được build sẵn)",
    )
    parser.add_argument(
        "--load-dataset",
        action="store_true",
        help=f"Load dataset đã có từ thay vì build lại",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số câu test (mặc định: tất cả câu de05)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Bỏ qua N câu đầu tiên (mặc định: 0). Ví dụ: --offset 1 --limit 1 = chỉ câu 2",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Số chunks RAG retrieve cho mỗi câu (mặc định: 5)",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help=(
            "Comma-separated metrics. Shortcuts: cp,cr,f,ar,ac. "
        ),
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="OpenAI model làm judge (mặc định: gpt-4o-mini)",
    )
    parser.add_argument(
        "--skip-math-judge",
        action="store_true",
        help="Bỏ qua bước MathJudge và Combined Report",
    )
    parser.add_argument(
        "--skip-explanation",
        action="store_true",
        help="Chỉ chạy accuracy judge (không gọi LLM cho explanation_quality — offline, miễn phí)",
    )

    args = parser.parse_args()
    asyncio.run(main(args))

