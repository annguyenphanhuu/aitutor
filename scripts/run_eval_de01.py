"""
Evaluate AITutor pipeline on Đề 01 (de7plus_de01).

KB:   theory.json  →  math_theory ChromaDB collection
              +  formulas.json  (exact-match formula_registry)
Test: de7plus_de01.json  (all question types)

Pipeline:
  1. Ingest theory vào ChromaDB (nếu chưa có / --force-ingest)
  2. Load câu hỏi từ de7plus_de01.json
  3. Retrieve context (theory hybrid search) + generate answer
  4. Run RAGAS evaluation (5 metrics)
  5. Run MathJudge evaluation (accuracy + explanation quality)
  6. In báo cáo chi tiết từng câu + combined report

Chạy:
    cd d:\\ANNGUYEN\\Project\\AITutor
    python scripts/run_eval_de01.py
    python scripts/run_eval_de01.py --skip-ingest --load-dataset
    python scripts/run_eval_de01.py --limit 12 --types exam_mcq,exam_short_answer
"""

from __future__ import annotations

import asyncio
import json
import logging
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
THEORY_FILE   = DATA_DIR / "theory.json"
TEST_FILE     = DATA_DIR / "exams" / "de7plus_de01.json"
EVAL_DIR      = DATA_DIR / "evaluation"

DATASET_PATH       = str(EVAL_DIR / "de01_samples.json")
RAGAS_REPORT_PATH  = str(EVAL_DIR / "de01_ragas_report.json")
MATH_REPORT_PATH   = str(EVAL_DIR / "de01_math_judge_report.json")
COMBINED_PATH      = str(EVAL_DIR / "de01_combined_report.json")
DETAIL_PATH        = str(EVAL_DIR / "de01_detailed_report.json")


# ── Banner ─────────────────────────────────────────────────────────────────────

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║        AITutor — Evaluation: Đề 7+ (Đề số 1)                ║
║  KB: theory (ChromaDB) + formulas (exact-match)              ║
║  Test: de7plus_de01.json                                     ║
║  Metrics: RAGAS (5) + MathJudge (accuracy + explanation)     ║
╚══════════════════════════════════════════════════════════════╝
""")


# ── Step 1: Ingest theory ──────────────────────────────────────────────────────

def ingest_theory(theory_file: Path, force: bool = False) -> None:
    from app.rag.knowledge_base import get_knowledge_base

    print("\n📚 STEP 1: INGEST THEORY")
    print("=" * 60)

    kb = get_knowledge_base()
    current_count = kb.get_theory_count()

    if current_count > 0 and not force:
        print(f"  ✅ math_theory đã có {current_count} docs — bỏ qua ingest.")
        print(f"     (dùng --force-ingest để ingest lại)")
        return

    if not theory_file.exists():
        print(f"  ⚠️  Không tìm thấy: {theory_file}")
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
        safe = {}
        for k, v in meta.items():
            safe[k] = int(v) if isinstance(v, bool) else ("" if v is None else v)
        doc["metadata"] = safe
        valid.append(doc)

    print(f"  📄 {theory_file.name}: {len(valid)} entries hợp lệ")

    if valid:
        kb.ingest_theory(valid)
        print(f"  ✅ math_theory: {kb.get_theory_count()} docs")
    else:
        print("  ⚠️  Không có entry hợp lệ nào.")


# ── Step 2: Load questions ─────────────────────────────────────────────────────

def load_questions(
    test_file: Path,
    types: list[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict]:
    from app.evaluation.dataset_builder import load_exam_questions

    print(f"\n📋 STEP 2: LOAD TEST QUESTIONS")
    print("=" * 60)

    allowed_types = types or ["exam_mcq", "exam_true_false", "exam_short_answer"]
    questions = load_exam_questions(
        exam_dir=str(test_file),
        question_types=allowed_types,
        limit=None,
    )

    if offset:
        questions = questions[offset:]
    if limit is not None:
        questions = questions[:limit]

    if not questions:
        print(f"  ❌ Không load được câu hỏi nào từ {test_file.name}")
        sys.exit(1)

    by_type: dict[str, int] = {}
    for q in questions:
        t = q.get("type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    print(f"  ✅ Loaded {len(questions)} câu từ {test_file.name}")
    for t, cnt in sorted(by_type.items()):
        label = {
            "exam_mcq":          "Trắc nghiệm",
            "exam_true_false":   "Đúng / Sai",
            "exam_short_answer": "Trả lời ngắn",
        }.get(t, t)
        print(f"     • {label:<25} {cnt} câu")

    return questions


# ── Step 3: Build / load samples ──────────────────────────────────────────────

async def build_samples(
    questions: list[dict],
    k: int = 5,
    dataset_path: str = DATASET_PATH,
) -> list[dict]:
    from app.evaluation.dataset_builder import build_ragas_samples, save_samples

    print(f"\n🔨 STEP 3: BUILD RAGAS SAMPLES")
    print("=" * 60)
    print(f"  📊 {len(questions)} câu × top-{k} theory chunks")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    samples = await build_ragas_samples(questions, k=k)
    save_samples(samples, dataset_path)

    print(f"  ✅ {len(samples)} samples → {dataset_path}")
    return samples


# ── Step 4: RAGAS evaluation ───────────────────────────────────────────────────

def run_ragas(
    samples: list[dict],
    requested_metrics: list[str] | None,
    judge_model: str,
    report_path: str = RAGAS_REPORT_PATH,
) -> dict:
    from app.evaluation.ragas_evaluator import RAGASEvaluator, _print_report

    print(f"\n🚀 STEP 4: RAGAS EVALUATION")
    print("=" * 60)
    print(f"  🤖 Judge model : {judge_model}")
    print(f"  📐 Metrics     : {requested_metrics or 'all (5)'}")

    evaluator = RAGASEvaluator(metrics=requested_metrics, model=judge_model)
    report = evaluator.run_and_report(samples=samples, output_path=report_path)

    _print_report(report)
    print(f"\n  💾 RAGAS report → {report_path}")
    return report


# ── Step 5: MathJudge evaluation ───────────────────────────────────────────────

def run_math_judge(
    samples: list[dict],
    judge_model: str,
    skip_explanation: bool = False,
    report_path: str = MATH_REPORT_PATH,
) -> dict:
    from app.evaluation.math_judge import MathJudge, print_math_report

    print(f"\n🧮 STEP 5: MATH JUDGE EVALUATION")
    print("=" * 60)
    mode = "accuracy only (offline)" if skip_explanation else "accuracy + explanation quality"
    print(f"  📐 Mode: {mode}")

    judge = MathJudge(model=judge_model, skip_explanation=skip_explanation)
    report = judge.run_and_report(samples=samples, output_path=report_path)

    print_math_report(report)
    print(f"  💾 MathJudge report → {report_path}")
    return report


# ── Step 6: Combined + detailed per-question report ───────────────────────────

def run_combined(
    ragas_report: dict,
    math_report: dict,
    samples: list[dict],
    combined_path: str = COMBINED_PATH,
    detail_path: str = DETAIL_PATH,
) -> dict:
    from app.evaluation.combined_report import merge_reports, print_combined_report

    print(f"\n📊 STEP 6: COMBINED + DETAILED REPORT")
    print("=" * 60)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    combined = merge_reports(
        ragas_report=ragas_report,
        math_report=math_report,
        output_path=combined_path,
    )
    print_combined_report(combined)
    print(f"  💾 Combined report → {combined_path}")

    # ── Build per-question detail ─────────────────────────────────────────────
    per_q = _build_per_question_detail(samples, ragas_report, math_report)

    Path(detail_path).write_text(
        json.dumps(per_q, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _print_detailed_report(per_q)
    print(f"\n  💾 Detailed report → {detail_path}")

    return combined


def _build_per_question_detail(
    samples: list[dict],
    ragas_report: dict,
    math_report: dict,
) -> list[dict]:
    """Build per-question row merging RAGAS + MathJudge scores."""
    # Index MathJudge results by question id
    mj_by_id: dict[str, dict] = {}
    for item in math_report.get("per_question", []):
        qid = item.get("id") or item.get("question_id", "")
        mj_by_id[qid] = item

    # Index RAGAS per-question scores (if available)
    ragas_by_id: dict[str, dict] = {}
    for item in ragas_report.get("per_question", []):
        qid = item.get("id") or item.get("question_id", "")
        ragas_by_id[qid] = item

    detail_rows = []
    for s in samples:
        meta = s.get("_meta", {})
        qid  = meta.get("id", "")
        qnum = qid.split("_q")[-1] if "_q" in qid else "?"

        mj   = mj_by_id.get(qid, {})
        rg   = ragas_by_id.get(qid, {})

        row = {
            "question_number": qnum,
            "id":              qid,
            "type":            meta.get("type", ""),
            "chapter":         meta.get("chapter", ""),
            "skill_id":        meta.get("skill_id", ""),
            "correct_answer":  meta.get("correct_answer", ""),
            "has_image":       meta.get("has_image", False),

            # RAGAS metrics (per-question if available, else aggregate)
            "context_precision":   rg.get("context_precision"),
            "context_recall":      rg.get("context_recall"),
            "faithfulness":        rg.get("faithfulness"),
            "answer_relevancy":    rg.get("answer_relevancy"),
            "answer_correctness":  rg.get("answer_correctness"),

            # MathJudge
            "math_accuracy":            mj.get("accuracy"),
            "explanation_quality":      mj.get("explanation_quality"),
            "math_judge_correct":       mj.get("correct"),
            "math_judge_predicted":     mj.get("predicted"),
            "math_judge_reason":        mj.get("reason", ""),

            # Context retrieved
            "n_contexts": len(s.get("retrieved_contexts", [])),
            "retrieved_contexts": s.get("retrieved_contexts", []),
        }
        detail_rows.append(row)

    return detail_rows


def _print_detailed_report(detail_rows: list[dict]) -> None:
    """Print per-question score table to stdout."""
    print("\n" + "═" * 90)
    print("  CHI TIẾT TỪNG CÂU — ĐỀ 01")
    print("═" * 90)

    TYPE_SHORT = {
        "exam_mcq":          "MCQ",
        "exam_true_false":   "T/F",
        "exam_short_answer": "SA ",
    }

    header = (
        f"  {'Q':>3}  {'T':3}  {'Acc':>5}  {'CP':>5}  {'CR':>5}  "
        f"{'Faith':>5}  {'AR':>5}  {'AC':>5}  {'EQ':>5}  Chapter / Skill"
    )
    print(header)
    print("  " + "-" * 87)

    def _fmt(v):
        if v is None:
            return "  —  "
        try:
            return f"{float(v):5.2f}"
        except (TypeError, ValueError):
            return f"{str(v):>5}"

    low_score_questions = []

    for row in detail_rows:
        qn      = row["question_number"]
        qtype   = TYPE_SHORT.get(row["type"], row["type"][:3])
        acc     = row.get("math_accuracy")
        cp      = row.get("context_precision")
        cr      = row.get("context_recall")
        faith   = row.get("faithfulness")
        ar      = row.get("answer_relevancy")
        ac      = row.get("answer_correctness")
        eq      = row.get("explanation_quality")
        chapter = (row.get("chapter") or "")[:28]

        # Mark low-performing rows
        is_low = (acc is not None and float(acc) < 0.6)
        flag = " ⚠️" if is_low else "   "

        print(
            f"  {qn:>3}  {qtype}  {_fmt(acc)}  {_fmt(cp)}  {_fmt(cr)}  "
            f"{_fmt(faith)}  {_fmt(ar)}  {_fmt(ac)}  {_fmt(eq)}  {chapter}{flag}"
        )

        if is_low:
            low_score_questions.append(row)

    print("═" * 90)
    print(
        "  Cột: Q=Câu, T=Type, Acc=Math Accuracy, CP=Context Precision, "
        "CR=Context Recall,\n"
        "       Faith=Faithfulness, AR=Answer Relevancy, AC=Answer Correctness, "
        "EQ=Explanation Quality"
    )

    # ── Low-score analysis ────────────────────────────────────────────────────
    if low_score_questions:
        print(f"\n⚠️  {len(low_score_questions)} CÂU CÓ ĐIỂM THẤP (Acc < 0.60)")
        print("─" * 70)
        for row in low_score_questions:
            print(f"\n  Câu {row['question_number']} [{row['type']}] — {row['chapter']}")
            print(f"  Skill: {row['skill_id']}")
            print(f"  Đáp án đúng: {row['correct_answer']}")
            print(f"  AI dự đoán : {row.get('math_judge_predicted', '?')}")

            # Diagnose common failure causes
            reasons = []
            cp_val = row.get("context_precision")
            cr_val = row.get("context_recall")
            faith_val = row.get("faithfulness")
            has_img = row.get("has_image", False)
            n_ctx  = row.get("n_contexts", 0)

            if has_img:
                reasons.append("📷 Câu có hình vẽ — model cần vision để đọc đúng dữ liệu")
            if n_ctx == 0:
                reasons.append("❌ Không retrieve được context nào từ theory KB")
            elif cp_val is not None and float(cp_val) < 0.4:
                reasons.append(f"📉 Context Precision thấp ({cp_val:.2f}) — theory KB thiếu nội dung khớp")
            if cr_val is not None and float(cr_val) < 0.4:
                reasons.append(f"📉 Context Recall thấp ({cr_val:.2f}) — retrieval bỏ sót kiến thức liên quan")
            if faith_val is not None and float(faith_val) < 0.5:
                reasons.append(f"⚠️  Faithfulness thấp ({faith_val:.2f}) — model có dấu hiệu hallucination")

            judge_reason = row.get("math_judge_reason", "").strip()
            if judge_reason:
                reasons.append(f"🔍 MathJudge: {judge_reason[:200]}")

            if not reasons:
                reasons.append("❓ Cần xem xét thủ công (không đủ thông tin tự chẩn đoán)")

            for r in reasons:
                print(f"    → {r}")

    print()


# ── Main ───────────────────────────────────────────────────────────────────────

async def main(args):
    print_banner()

    # Step 1 — Ingest theory
    if not args.skip_ingest:
        ingest_theory(THEORY_FILE, force=args.force_ingest)
    else:
        print("\n⏭️  STEP 1: Bỏ qua ingest (--skip-ingest)")

    # Step 2 — Load questions
    question_types = (
        [t.strip() for t in args.types.split(",")]
        if args.types else None
    )
    questions = load_questions(
        test_file=TEST_FILE,
        types=question_types,
        limit=args.limit,
        offset=args.offset,
    )

    # Step 3 — Build or load dataset
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if args.load_dataset and Path(DATASET_PATH).exists():
        from app.evaluation.dataset_builder import load_samples
        samples = load_samples(DATASET_PATH)
        print(f"\n📂 STEP 3: Loaded existing dataset — {len(samples)} samples")
    else:
        samples = await build_samples(questions, k=args.k, dataset_path=DATASET_PATH)

    # Step 4 — RAGAS
    METRIC_MAP = {
        "cp": "context_precision",
        "cr": "context_recall",
        "f":  "faithfulness",
        "ar": "answer_relevancy",
        "ac": "answer_correctness",
    }
    requested_metrics = None
    if args.metrics:
        requested_metrics = [
            METRIC_MAP.get(m.strip(), m.strip())
            for m in args.metrics.split(",")
        ]

    ragas_report = run_ragas(
        samples=samples,
        requested_metrics=requested_metrics,
        judge_model=args.judge_model,
        report_path=RAGAS_REPORT_PATH,
    )

    # Step 5 — MathJudge
    if not args.skip_math_judge:
        math_report = run_math_judge(
            samples=samples,
            judge_model=args.judge_model,
            skip_explanation=args.skip_explanation,
            report_path=MATH_REPORT_PATH,
        )

        # Step 6 — Combined + detailed
        run_combined(
            ragas_report=ragas_report,
            math_report=math_report,
            samples=samples,
            combined_path=COMBINED_PATH,
            detail_path=DETAIL_PATH,
        )
    else:
        print("\n⏭️  STEP 5-6: Bỏ qua MathJudge (--skip-math-judge)")

    print("\n✅ Đánh giá Đề 01 hoàn tất!")
    print(f"   Dataset          : {DATASET_PATH}")
    print(f"   RAGAS Report     : {RAGAS_REPORT_PATH}")
    if not args.skip_math_judge:
        print(f"   MathJudge Report : {MATH_REPORT_PATH}")
        print(f"   Combined Report  : {COMBINED_PATH}")
        print(f"   Detailed Report  : {DETAIL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate AITutor on Đề 01 (theory+formulas KB only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Bỏ qua bước ingest theory (nếu ChromaDB đã có data)",
    )
    parser.add_argument(
        "--force-ingest", action="store_true",
        help="Ingest lại theory kể cả khi ChromaDB đã có data",
    )
    parser.add_argument(
        "--load-dataset", action="store_true",
        help="Load dataset đã build sẵn thay vì generate lại",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Giới hạn số câu test (mặc định: tất cả)",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Bỏ qua N câu đầu (mặc định: 0)",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Số theory chunks retrieve mỗi câu (mặc định: 5)",
    )
    parser.add_argument(
        "--types", default=None,
        help=(
            "Lọc loại câu (comma-separated). "
            "Ví dụ: exam_mcq,exam_short_answer. "
            "Mặc định: tất cả 3 loại."
        ),
    )
    parser.add_argument(
        "--metrics", default=None,
        help="Comma-separated RAGAS metrics. Shortcuts: cp,cr,f,ar,ac.",
    )
    parser.add_argument(
        "--judge-model", default="gpt-4o-mini",
        help="OpenAI model làm judge (mặc định: gpt-4o-mini)",
    )
    parser.add_argument(
        "--skip-math-judge", action="store_true",
        help="Bỏ qua MathJudge và Combined Report",
    )
    parser.add_argument(
        "--skip-explanation", action="store_true",
        help="Chỉ chạy accuracy judge (không gọi LLM cho explanation_quality)",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
