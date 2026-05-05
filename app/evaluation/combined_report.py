"""
Combined Report — Merge RAGAS + MathJudge reports thành một dashboard.

Usage:
    from app.evaluation.combined_report import merge_reports, print_combined_report

    report = merge_reports(ragas_report, math_judge_report)
    print_combined_report(report)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


def merge_reports(
    ragas_report: dict,
    math_report: dict,
    output_path: Optional[str] = None,
) -> dict:
    """
    Merge RAGAS report + MathJudge report thành một báo cáo thống nhất.

    Strategy:
    - Retrieval metrics (context_precision, context_recall) → từ RAGAS (đáng tin cậy)
    - Generation metrics (accuracy, explanation_quality) → từ MathJudge (phù hợp domain)
    - answer_correctness RAGAS → giữ lại để so sánh
    """

    # ── Retrieval từ RAGAS ──
    ragas_scores = ragas_report.get("summary_scores", {})
    retrieval_scores = {
        k: v for k, v in ragas_scores.items()
        if k in {"llm_context_precision_with_reference", "context_recall"}
    }

    # ── Generation từ MathJudge ──
    math_scores = math_report.get("summary_scores", {})

    # ── Combine ──
    combined_scores = {
        # Retrieval (RAGAS)
        "context_precision":     retrieval_scores.get("llm_context_precision_with_reference", None),
        "context_recall":        retrieval_scores.get("context_recall", None),
        # Accuracy per type (MathJudge, no LLM)
        "mcq_accuracy":          math_scores.get("mcq_accuracy", None),
        "true_false_accuracy":   math_scores.get("true_false_accuracy", None),
        "short_answer_accuracy": math_scores.get("short_answer_accuracy", None),
        "overall_accuracy":      math_scores.get("overall_accuracy", None),
        # Pedagogical quality (LLM, optional)
        "step_clarity":          math_scores.get("step_clarity", None),
        # Reference: RAGAS answer_correctness (semantic similarity)
        "ragas_answer_correctness": ragas_scores.get("answer_correctness", None),
    }
    # Remove None values
    combined_scores = {k: v for k, v in combined_scores.items() if v is not None}

    thresholds = {
        "context_precision":     0.70,
        "context_recall":        0.65,
        "mcq_accuracy":          0.85,
        "true_false_accuracy":   0.75,
        "short_answer_accuracy": 0.70,
        "overall_accuracy":      0.75,
        "step_clarity":          0.70,
        "ragas_answer_correctness": 0.70,
    }


    import math
    pass_fail = {
        name: (
            "N/A (parse error)"
            if score is None or (isinstance(score, float) and math.isnan(score))
            else ("\u2705 PASS" if score >= thresholds.get(name, 0.70) else "\u274c FAIL")
        )
        for name, score in combined_scores.items()
    }

    # ── Per-skill (MathJudge accuracy, cleaner data) ──
    skill_accuracy = math_report.get("skill_breakdown", {})
    skill_ragas    = ragas_report.get("skill_breakdown", {})

    skill_combined: dict[str, dict] = {}
    all_skills = set(skill_accuracy) | set(skill_ragas)
    for skill in sorted(all_skills):
        skill_combined[skill] = {}
        if skill in skill_accuracy:
            skill_combined[skill]["accuracy"] = skill_accuracy[skill]
        if skill in skill_ragas:
            sk = skill_ragas[skill]
            if "context_recall" in sk:
                skill_combined[skill]["context_recall"] = sk["context_recall"]

    report = {
        "timestamp":        datetime.now().isoformat(),
        "n_samples":        ragas_report.get("n_samples", 0),
        "ragas_judge_model": ragas_report.get("judge_model", ""),
        "math_judge_model":  math_report.get("judge_model", ""),
        "combined_scores":   combined_scores,
        "type_breakdown":    math_report.get("type_breakdown", {}),
        "skill_breakdown":   skill_combined,
        "thresholds":        thresholds,
        "pass_fail":         pass_fail,
        "note": (
            "Retrieval metrics (context_precision, context_recall) từ RAGAS. "
            "Generation metrics (answer_accuracy, explanation_quality) từ MathJudge."
        ),
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return report


def print_combined_report(report: dict) -> None:
    """In combined report ra console."""
    print("\n" + "=" * 65)
    print("📊  COMBINED EVALUATION REPORT — AITutor")
    print("=" * 65)
    print(f"⏰  Timestamp    : {report['timestamp']}")
    print(f"📝  Samples      : {report['n_samples']}")
    print(f"🤖  RAGAS judge  : {report.get('ragas_judge_model', 'N/A')}")
    print(f"🧮  Math judge   : {report.get('math_judge_model', 'N/A')}")
    print()

    print("── COMBINED SCORES ───────────────────────────────────────────")
    section_labels = {
        "context_precision":        "📡 [RETRIEVAL] context_precision",
        "context_recall":           "📡 [RETRIEVAL] context_recall",
        "answer_accuracy":          "✏️  [GENERATION] answer_accuracy",
        "explanation_quality":      "📚 [GENERATION] explanation_quality",
        "ragas_answer_correctness": "🔍 [REFERENCE]  ragas_answer_correctness",
    }

    import math
    thresholds = report.get("thresholds", {})
    for name, score in report.get("combined_scores", {}).items():
        label = section_labels.get(name, name)
        if score is None or (isinstance(score, float) and math.isnan(score)):
            print(f"  ⚠️  {label:<45} N/A  (RAGAS parse error)")
            continue
        thresh  = thresholds.get(name, 0.70)
        status  = "✅" if score >= thresh else "❌"
        bar_len = int(score * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {status} {label:<45} {score:.4f}  [{bar}]")

    print()
    print("── BY QUESTION TYPE (accuracy) ───────────────────────────────")
    for qtype, score in report.get("type_breakdown", {}).items():
        status = "✅" if score >= 0.70 else "❌"
        print(f"  {status} {qtype:<35} {score:.4f}")

    print()
    print("── SKILL BREAKDOWN ───────────────────────────────────────────")
    for skill, scores in report.get("skill_breakdown", {}).items():
        details = "  ".join(f"{k}: {v:.3f}" for k, v in scores.items())
        print(f"  📚 {skill:<35} {details}")

    print()
    print("── PASS/FAIL ─────────────────────────────────────────────────")
    for name, verdict in report["pass_fail"].items():
        print(f"  {verdict}  {name}")

    print()
    print(f"  ℹ️  {report.get('note', '')}")
    print("=" * 65 + "\n")
