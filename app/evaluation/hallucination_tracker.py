"""
Reflection Metrics Tracker — theo dõi hallucination rate từ ReflectionEngine.

Tích hợp vào app/agents/reflection.py để log mỗi khi Reflection chạy.

Usage (trong reflection.py):
    from app.evaluation.hallucination_tracker import ReflectionMetrics
    ReflectionMetrics.record(result)          # sau khi chạy reflect()
    print(ReflectionMetrics.report())         # xem tổng kết

API endpoint /api/evaluation/hallucination-report cũng expose dữ liệu này.
"""

from __future__ import annotations

import threading
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ReflectionMetrics:
    """
    Thread-safe singleton counter cho hallucination stats.

    Metrics tracked:
        total_reflections   — tổng số lần ReflectionEngine chạy
        total_corrections   — số lần LLM cần sửa (was_corrected = True)
        total_verifications — tổng số phép tính được SymPy verify
        total_sympy_errors  — số phép tính mà SymPy thất bại
        skill_corrections   — corrections theo skill_id
    """

    _lock                = threading.Lock()
    _total_reflections   = 0
    _total_corrections   = 0
    _total_verifications = 0
    _total_sympy_errors  = 0
    _skill_corrections: dict[str, dict] = {}  # skill_id → {total, corrections}
    _history: list[dict] = []                 # raw records (last 1000)

    @classmethod
    def record(
        cls,
        result,                            # ReflectionResult instance
        skill_id: Optional[str] = None,
    ) -> None:
        """Record a ReflectionResult. Call this after every reflect() call."""
        with cls._lock:
            cls._total_reflections   += 1
            verifications             = result.verifications or []
            sympy_errors              = sum(
                1 for v in verifications
                if not v.get("result", {}).get("success", True)
            )

            cls._total_verifications += len(verifications)
            cls._total_sympy_errors  += sympy_errors

            if result.was_corrected:
                cls._total_corrections += 1

            # Per-skill tracking
            if skill_id:
                if skill_id not in cls._skill_corrections:
                    cls._skill_corrections[skill_id] = {"total": 0, "corrections": 0}
                cls._skill_corrections[skill_id]["total"] += 1
                if result.was_corrected:
                    cls._skill_corrections[skill_id]["corrections"] += 1

            # History (keep last 1000)
            cls._history.append({
                "ts":            datetime.now().isoformat(),
                "skill_id":      skill_id,
                "was_corrected": result.was_corrected,
                "n_verifications": len(verifications),
                "n_sympy_errors":  sympy_errors,
                "thinking_steps":  len(result.thinking_log),
            })
            if len(cls._history) > 1000:
                cls._history = cls._history[-1000:]

    @classmethod
    def report(cls) -> dict:
        """Return current hallucination metrics summary."""
        with cls._lock:
            n_ref  = max(cls._total_reflections, 1)
            n_ver  = max(cls._total_verifications, 1)

            correction_rate  = cls._total_corrections   / n_ref
            sympy_error_rate = cls._total_sympy_errors  / n_ver
            coverage_rate    = (
                cls._total_verifications / (n_ref * 5)  # assume ~5 expr per response
            )

            per_skill = {}
            for skill, counts in cls._skill_corrections.items():
                per_skill[skill] = {
                    "total":       counts["total"],
                    "corrections": counts["corrections"],
                    "rate": round(counts["corrections"] / max(counts["total"], 1), 4),
                }

            return {
                "timestamp":           datetime.now().isoformat(),
                "total_reflections":   cls._total_reflections,
                "total_corrections":   cls._total_corrections,
                "total_verifications": cls._total_verifications,
                "total_sympy_errors":  cls._total_sympy_errors,

                # Key metrics
                "correction_rate":     round(correction_rate, 4),
                "sympy_error_rate":    round(sympy_error_rate, 4),
                "reflection_coverage": round(min(coverage_rate, 1.0), 4),

                # Thresholds
                "thresholds": {
                    "correction_rate":  0.10,  # target: < 10%
                    "sympy_error_rate": 0.05,  # target: < 5%
                },

                # Pass/fail
                "pass_fail": {
                    "correction_rate":  "✅ PASS" if correction_rate < 0.10  else "❌ FAIL",
                    "sympy_error_rate": "✅ PASS" if sympy_error_rate < 0.05 else "❌ FAIL",
                },

                "per_skill": per_skill,
            }

    @classmethod
    def print_report(cls) -> None:
        """Print formatted report to console."""
        r = cls.report()
        print("\n── HALLUCINATION TRACKER ──────────────────────────────")
        print(f"  Reflections run   : {r['total_reflections']}")
        print(f"  Corrections made  : {r['total_corrections']}")
        print(f"  SymPy verifications: {r['total_verifications']}")
        print(f"  SymPy errors      : {r['total_sympy_errors']}")
        print()
        print(f"  Correction rate   : {r['correction_rate']:.2%}  {r['pass_fail']['correction_rate']}")
        print(f"  SymPy error rate  : {r['sympy_error_rate']:.2%}  {r['pass_fail']['sympy_error_rate']}")
        print(f"  Reflection coverage: {r['reflection_coverage']:.2%}")
        print("─" * 55)

    @classmethod
    def reset(cls) -> None:
        """Reset all counters (dùng cho tests)."""
        with cls._lock:
            cls._total_reflections   = 0
            cls._total_corrections   = 0
            cls._total_verifications = 0
            cls._total_sympy_errors  = 0
            cls._skill_corrections   = {}
            cls._history             = []

    @classmethod
    def save_history(cls, path: str = "data/reflection_history.json") -> None:
        """Save history to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with cls._lock:
            Path(path).write_text(
                json.dumps(cls._history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        logger.info("Reflection history saved → %s", path)
