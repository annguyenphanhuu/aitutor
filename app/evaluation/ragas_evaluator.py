"""
RAGAS Evaluator — chỉ chạy RAG RETRIEVAL metrics.

⚠️  Lưu ý thiết kế:
  RAGAS được dùng THUẦN TÚY để monitor chất lượng retrieval.
  Các metrics generation (faithfulness, answer_relevancy, answer_correctness)
  đã bị loại bỏ vì không phù hợp với:
    - Câu hỏi có hình ảnh/đồ thị (Vision) → faithfulness luôn = 0
    - MCQ format tiếng Việt → answer_relevancy bị nhiễu bởi các lựa chọn A/B/C/D
  Thay thế bằng LLMExaminer và judge_visual_reasoning trong math_judge.py.

Metrics được đo (2):
  • context_precision    — (RAGAS) Ngữ cảnh retrieve có thực sự cần thiết không?
  • math_context_coverage — (Custom) Ngữ cảnh có cung cấp đủ PHƯƠNG PHÁP/CÔNG CỤ
                            để giải bài không? (thay thế context_recall gốc vốn
                            chỉ đo fact-matching, không phù hợp bài Toán)

Docs: https://docs.ragas.io/en/stable/concepts/metrics/

Usage:
    python -m app.evaluation.ragas_evaluator --dataset data/ragas_dataset.json
    python -m app.evaluation.ragas_evaluator --build --limit 20
"""

from __future__ import annotations

import asyncio
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ── RAGAS imports (lazy để tránh lỗi nếu chưa cài) ─────────────────────────

def _import_ragas():
    """Import RAGAS components — raises ImportError với hướng dẫn nếu chưa cài."""
    try:
        from ragas import evaluate, EvaluationDataset, SingleTurnSample
        from ragas.metrics import LLMContextPrecisionWithReference
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        return {
            "evaluate":          evaluate,
            "EvaluationDataset": EvaluationDataset,
            "SingleTurnSample":  SingleTurnSample,
            "metrics": {
                # context_recall bị loại bỏ — thay bằng math_context_coverage
                # (xem app/evaluation/math_context_coverage.py)
                "context_precision": LLMContextPrecisionWithReference(),
            },
            "ChatOpenAI":       ChatOpenAI,
            "OpenAIEmbeddings": OpenAIEmbeddings,
            "LangchainLLMWrapper":        LangchainLLMWrapper,
            "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
        }
    except ImportError:
        raise ImportError(
            "RAGAS chưa được cài. Chạy: pip install ragas datasets"
        )


# ── Core Evaluator ─────────────────────────────────────────────────────────

class RAGASEvaluator:
    """
    Đánh giá RAG pipeline của AITutor bằng RAGAS.

    Attributes
    ----------
    metrics : list[str]
        Tên các metrics muốn đo. Mặc định là tất cả 5 metrics.
    model : str
        OpenAI model dùng làm LLM judge.
    embedding_model : str
        OpenAI embedding model dùng cho answer_relevancy.
    """

    def __init__(
        self,
        metrics: Optional[list[str]] = None,
        model:           str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ):
        from app.config import get_settings
        self.settings = get_settings()

        # Default metrics:
        #   context_precision    — RAGAS built-in
        #   math_context_coverage — Custom (thay thế context_recall)
        self.requested_metrics = metrics or [
            "context_precision",
            "math_context_coverage",
        ]
        self.judge_model      = model
        self.embedding_model  = embedding_model
        self._ragas = None  # lazy init

    def _setup_ragas(self):
        """Khởi tạo RAGAS với OpenAI LLM + Embedding judges."""
        if self._ragas:
            return self._ragas

        r = _import_ragas()

        # Wrap LangChain models cho RAGAS
        judge_llm = r["LangchainLLMWrapper"](
            r["ChatOpenAI"](
                model=self.judge_model,
                api_key=self.settings.OPENAI_API_KEY,
                temperature=0.0,
            )
        )
        judge_emb = r["LangchainEmbeddingsWrapper"](
            r["OpenAIEmbeddings"](
                model=self.embedding_model,
                api_key=self.settings.OPENAI_API_KEY,
            )
        )

        # Set judge LLM/embedding cho từng RAGAS built-in metric
        # math_context_coverage là custom — không xử lý ở đây
        _RAGAS_BUILTIN = {"context_precision"}
        selected_metrics = []
        for name in self.requested_metrics:
            if name not in _RAGAS_BUILTIN:
                continue  # custom metrics handled separately
            metric = r["metrics"].get(name)
            if metric is None:
                logger.warning("Unknown RAGAS metric: %s — skipped", name)
                continue
            metric.llm = judge_llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = judge_emb
            selected_metrics.append(metric)

        self._ragas = {
            "evaluate":          r["evaluate"],
            "EvaluationDataset": r["EvaluationDataset"],
            "SingleTurnSample":  r["SingleTurnSample"],
            "selected_metrics":  selected_metrics,
            "_judge_llm":        judge_llm,  # expose để custom metrics dùng
        }
        return self._ragas

    def build_dataset(self, samples: list[dict]):
        """
        Convert samples list → RAGAS EvaluationDataset.

        samples format (vd. từ dataset_builder.py):
            [{
                "user_input":         "câu hỏi",
                "retrieved_contexts": ["chunk1", "chunk2", ...],
                "response":           "câu trả lời của AI",
                "reference":          "đáp án chuẩn",
            }, ...]
        """
        r = self._setup_ragas()
        ragas_samples = []

        for s in samples:
            ragas_samples.append(
                r["SingleTurnSample"](
                    user_input         = s["user_input"],
                    retrieved_contexts = s["retrieved_contexts"],
                    response           = s["response"],
                    reference          = s["reference"],
                )
            )

        return r["EvaluationDataset"](samples=ragas_samples)

    def _run_math_coverage(self, samples: list[dict]) -> list[dict]:
        """
        Chạy MathContextCoverage (custom metric) trên toàn bộ samples.
        Trả về list of {math_context_coverage: float, math_context_coverage_reason: str}.
        """
        from app.evaluation.math_context_coverage import MathContextCoverage
        r = self._setup_ragas()

        # Reuse cùng judge LLM đã được setup
        judge_llm = r.get("_judge_llm")
        metric = MathContextCoverage(llm=judge_llm)
        return metric.score_batch(samples)

    def run(self, samples: list[dict]) -> dict:
        """
        Run RAGAS evaluation on samples.
        Merge kết quả RAGAS built-in + custom MathContextCoverage.

        Returns
        -------
        dict
            RAGAS result object (có thêm attribute _custom_scores).
        """
        r = self._setup_ragas()

        dataset  = self.build_dataset(samples)
        metrics  = r["selected_metrics"]

        logger.info(
            "Running RAGAS evaluation: %d samples × %d metrics",
            len(samples), len(metrics),
        )

        result = r["evaluate"](
            dataset=dataset,
            metrics=metrics,
        )

        # Chạy custom metric nếu được yêu cầu
        if "math_context_coverage" in self.requested_metrics:
            custom_scores = self._run_math_coverage(samples)
            result._custom_scores = custom_scores  # attach để run_and_report dùng
        else:
            result._custom_scores = []

        return result

    def run_and_report(
        self,
        samples: list[dict],
        output_path: Optional[str] = None,
    ) -> dict:
        """
        Evaluate và tạo báo cáo đầy đủ.

        Parameters
        ----------
        samples : list[dict]
            Danh sách samples từ dataset_builder.
        output_path : str, optional
            Nếu có, lưu báo cáo JSON ra file.

        Returns
        -------
        dict
            Báo cáo đầy đủ.
        """
        result = self.run(samples)

        # ── RAGAS built-in scores ──────────────────────────────────────────────
        summary_scores = {}
        try:
            df = result.to_pandas()
            for col in df.columns:
                if col not in ("user_input", "retrieved_contexts", "response", "reference"):
                    summary_scores[col] = round(float(df[col].mean(skipna=True)), 4)
        except Exception:
            summary_scores = dict(result)

        # ── Custom MathContextCoverage scores ─────────────────────────────────
        custom_list = getattr(result, "_custom_scores", [])
        if custom_list:
            import statistics
            cov_scores = [r.get("math_context_coverage") for r in custom_list
                          if r.get("math_context_coverage") is not None]
            if cov_scores:
                summary_scores["math_context_coverage"] = round(
                    statistics.mean(cov_scores), 4
                )
            # Lưu per-sample reasons để debug
            coverage_reasons = [
                r.get("math_context_coverage_reason", "") for r in custom_list
            ]
        else:
            coverage_reasons = []

        # ── Per-skill breakdown ────────────────────────────────────────────────
        skill_breakdown = {}
        try:
            df = result.to_pandas()
            metas = [s.get("_meta", {}) for s in samples]
            df["skill_id"] = [m.get("skill_id", "unknown") for m in metas]
            df["chapter"]  = [m.get("chapter",  "unknown") for m in metas]

            ragas_metric_cols = [c for c in df.columns
                                  if c in self.requested_metrics
                                  and c != "math_context_coverage"]
            for skill, grp in df.groupby("skill_id"):
                skill_breakdown[skill] = {
                    col: round(float(grp[col].mean(skipna=True)), 4)
                    for col in ragas_metric_cols
                    if col in grp.columns
                }
        except Exception as e:
            logger.debug("Skill breakdown failed: %s", e)

        thresholds = {
            "context_precision":     0.70,
            "math_context_coverage": 0.65,
        }

        report = {
            "timestamp":        datetime.now().isoformat(),
            "n_samples":        len(samples),
            "judge_model":      self.judge_model,
            "metrics_run":      self.requested_metrics,
            "summary_scores":   summary_scores,
            "coverage_reasons": coverage_reasons,
            "skill_breakdown":  skill_breakdown,
            "thresholds":       thresholds,
            "pass_fail": {
                name: (
                    "✅ PASS"
                    if summary_scores.get(name, 0) >= thresholds.get(name, 0.70)
                    else "❌ FAIL"
                )
                for name in self.requested_metrics
                if name in summary_scores
            },
            "note": (
                "RAGAS đo context_precision (built-in). "
                "math_context_coverage (custom) đo mức độ context hỗ trợ phương pháp giải. "
                "Generation metrics được đo bằng LLMExaminer trong math_judge.py."
            ),
        }

        # Save
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Report saved → %s", output_path)

        return report


# ── CLI ────────────────────────────────────────────────────────────────────────

def _print_report(report: dict) -> None:
    """In báo cáo ra console."""
    print("\n" + "=" * 60)
    print("📊  RAGAS EVALUATION REPORT — AITutor")
    print("=" * 60)
    print(f"⏰  Timestamp : {report['timestamp']}")
    print(f"📝  Samples   : {report['n_samples']}")
    print(f"🤖  Judge LLM : {report['judge_model']}")
    print()
    print("── SUMMARY SCORES ──────────────────────────────────────")

    thresholds = report.get("thresholds", {})
    for name, score in report["summary_scores"].items():
        import math
        if score is None or (isinstance(score, float) and math.isnan(score)):
            print(f"  ⚠️  {name:<25} N/A     (RAGAS parse error)")
            continue
        thresh  = thresholds.get(name, 0.70)
        status  = "✅" if score >= thresh else "❌"
        bar_len = int(score * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {status} {name:<25} {score:.4f}  [{bar}]  (threshold: {thresh})")

    if report.get("skill_breakdown"):
        print()
        print("── SKILL BREAKDOWN ─────────────────────────────────────")
        for skill, scores in report["skill_breakdown"].items():
            score_str = "  ".join(f"{k}: {v:.3f}" for k, v in scores.items())
            print(f"  📚 {skill:<30} {score_str}")

    print()
    print("── PASS/FAIL ────────────────────────────────────────────")
    for name, result in report["pass_fail"].items():
        print(f"  {result}  {name}")
    print("=" * 60 + "\n")


async def _main(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Step 1: Build or load dataset
    if args.build or not Path(args.dataset).exists():
        print(f"🔨 Building dataset (limit={args.limit})...")
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
            print("❌ No questions found.")
            return
        samples = await build_ragas_samples(questions, k=5)
        save_samples(samples, args.dataset)
    else:
        from app.evaluation.dataset_builder import load_samples
        samples = load_samples(args.dataset)
        print(f"📂 Loaded {len(samples)} samples from {args.dataset}")

    # Step 2: Parse requested metrics
    metric_map = {
        "cp":  "context_precision",
        "cr":  "context_recall",
        "f":   "faithfulness",
        "ar":  "answer_relevancy",
        "ac":  "answer_correctness",
    }
    if args.metrics:
        requested = [metric_map.get(m, m) for m in args.metrics.split(",")]
    else:
        requested = None  # all

    # Step 3: Run RAGAS
    evaluator = RAGASEvaluator(
        metrics=requested,
        model=args.judge_model,
    )
    report = evaluator.run_and_report(
        samples=samples,
        output_path=args.output,
    )

    _print_report(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS evaluation for AITutor")
    parser.add_argument("--dataset",    default="data/ragas_dataset.json",  help="Path to samples JSON")
    parser.add_argument("--output",     default="data/ragas_report.json",   help="Path to save report JSON")
    parser.add_argument("--build",      action="store_true",               help="Rebuild dataset before evaluating")
    parser.add_argument("--exam-dir",   default="data/exams",              help="Source exam directory")
    parser.add_argument("--limit",      type=int, default=20,              help="Max questions when building")
    parser.add_argument(
        "--metrics",
        default=None,
        help=(
            "Comma-separated metrics. Shortcuts: cp,cr,f,ar,ac. "
            "Full names: context_precision,context_recall,"
            "faithfulness,answer_relevancy,answer_correctness"
        ),
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="OpenAI model to use as judge (default: gpt-4o-mini to save cost)",
    )
    args = parser.parse_args()
    asyncio.run(_main(args))
