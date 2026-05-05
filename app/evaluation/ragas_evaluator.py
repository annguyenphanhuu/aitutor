"""
RAGAS Evaluator — chạy các metrics RAGAS trên dataset đã build.

Metrics được đo:
  • context_precision    — Ngữ cảnh retrieve có thực sự cần thiết không?
  • context_recall       — Ngữ cảnh có bao phủ đủ ground truth không?
  • faithfulness         — Câu trả lời có dựa trên context không?
  • answer_relevancy     — Câu trả lời có liên quan đến câu hỏi không?
  • answer_correctness   — Câu trả lời có đúng so với ground truth không? (semantic + factual)

Docs: https://docs.ragas.io/en/stable/concepts/metrics/

Usage:
    # Quick: chỉ chạy trên dataset đã có
    python -m app.evaluation.ragas_evaluator --dataset data/ragas_dataset.json

    # Build dataset mới rồi evaluate
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
        from ragas.metrics import (
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            Faithfulness,
            AnswerRelevancy,
            AnswerCorrectness,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        return {
            "evaluate":          evaluate,
            "EvaluationDataset": EvaluationDataset,
            "SingleTurnSample":  SingleTurnSample,
            "metrics": {
                "context_precision": LLMContextPrecisionWithReference(),
                "context_recall":    LLMContextRecall(),
                "faithfulness":      Faithfulness(),
                "answer_relevancy":  AnswerRelevancy(),
                "answer_correctness": AnswerCorrectness(),
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
        model:           str = "gpt-4o-mini",   # Dùng mini để tiết kiệm chi phí judge
        embedding_model: str = "text-embedding-3-small",
    ):
        from app.config import get_settings
        self.settings = get_settings()

        self.requested_metrics = metrics or [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
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

        # Set judge LLM/embedding cho từng metric
        selected_metrics = []
        for name in self.requested_metrics:
            metric = r["metrics"].get(name)
            if metric is None:
                logger.warning("Unknown metric: %s — skipped", name)
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

    def run(self, samples: list[dict]) -> dict:
        """
        Run RAGAS evaluation on samples.

        Returns
        -------
        dict
            Summary của tất cả metrics + per-sample scores.
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

        # Lấy score trung bình
        summary_scores = {}
        try:
            df = result.to_pandas()
            for col in df.columns:
                if col not in ("user_input", "retrieved_contexts", "response", "reference"):
                    summary_scores[col] = round(float(df[col].mean(skipna=True)), 4)
        except Exception:
            # Fallback nếu result không có to_pandas
            summary_scores = dict(result)

        # Tạo per-skill breakdown nếu có metadata
        skill_breakdown = {}
        try:
            df = result.to_pandas()
            metas = [s.get("_meta", {}) for s in samples]
            df["skill_id"] = [m.get("skill_id", "unknown") for m in metas]
            df["chapter"]  = [m.get("chapter",  "unknown") for m in metas]

            metric_cols = [c for c in df.columns if c in self.requested_metrics]
            for skill, grp in df.groupby("skill_id"):
                skill_breakdown[skill] = {
                    col: round(float(grp[col].mean(skipna=True)), 4)
                    for col in metric_cols
                    if col in grp.columns
                }
        except Exception as e:
            logger.debug("Skill breakdown failed: %s", e)

        report = {
            "timestamp":       datetime.now().isoformat(),
            "n_samples":       len(samples),
            "judge_model":     self.judge_model,
            "metrics_run":     self.requested_metrics,
            "summary_scores":  summary_scores,
            "skill_breakdown": skill_breakdown,
            "thresholds": {
                "context_precision":  0.70,
                "context_recall":     0.65,
                "faithfulness":       0.80,
                "answer_relevancy":   0.75,
                "answer_correctness": 0.70,
            },
            "pass_fail": {
                name: (
                    "✅ PASS"
                    if summary_scores.get(name, 0) >= 0.70
                    else "❌ FAIL"
                )
                for name in self.requested_metrics
            },
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
