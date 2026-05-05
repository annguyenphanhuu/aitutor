"""
MathJudge — Custom LLM-as-a-Judge cho AITutor.

Đánh giá 4 chiều:
  1. accuracy          — câu trả lời đúng/sai (KHÔNG cần LLM, deterministic)
  2. explanation_quality — chất lượng lời giải (LLM judge)

Hỗ trợ 3 dạng câu hỏi:
  • exam_mcq          → extract A/B/C/D từ response
  • exam_true_false   → parse dạng "a-T,b-F,c-T,d-F"
  • exam_short_answer → so sánh số / chuỗi chính xác

Usage:
    from app.evaluation.math_judge import MathJudge

    judge = MathJudge(model="gpt-4o-mini")
    report = judge.run_and_report(samples, output_path="data/math_judge_report.json")
"""

from __future__ import annotations

import json
import logging
import re
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Prompts ────────────────────────────────────────────────────────────────────

EXPLANATION_QUALITY_PROMPT = """\
Bạn là chuyên gia đánh giá chất lượng giảng dạy Toán 12.

Hãy đánh giá lời giải của gia sư AI theo thang điểm từ 0.0 đến 1.0:
- 1.0: Lập luận đúng, rõ ràng, đầy đủ các bước, học sinh lớp 12 dễ hiểu
- 0.75: Đúng kết quả, có bước giải thích nhưng bỏ một vài bước trung gian
- 0.5: Đúng kết quả nhưng giải thích ngắn hoặc thiếu rõ ràng
- 0.25: Có cố gắng giải nhưng lập luận có lỗ hổng logic
- 0.0: Sai kết quả hoặc không liên quan đến câu hỏi

Câu hỏi:
{question}

Đáp án tham chiếu (ground truth):
{reference}

Lời giải của AI:
{response}

Chỉ trả lời bằng một số thực duy nhất trong khoảng [0.0, 1.0], không giải thích thêm.
Ví dụ hợp lệ: 0.75
"""

STEP_CLARITY_PROMPT = """\
Bạn là giáo viên Toán 12 đang chấm lời giải của học sinh.

Đánh giá lời giải dưới đây theo tiêu chí RÕ RÀNG VÀ ĐẦY ĐỦ BƯỚC GIẢI:
- 1.0: Có đủ các bước trung gian, lập luận mạch lạc, học sinh có thể theo dõi được
- 0.75: Có bước giải nhưng bỏ 1-2 bước trung gian không quan trọng
- 0.5: Chỉ có kết quả + giải thích ngắn, thiếu bước thực hiện
- 0.25: Lời giải rời rạc, khó hiểu
- 0.0: Không có lời giải hoặc không liên quan

Câu hỏi: {question}
Lời giải AI: {response}

Chỉ trả lời đúng 1 số trong [0.0, 1.0]. Ví dụ: 0.75
"""



# ── Low-level judge functions (no LLM) ────────────────────────────────────────

def judge_mcq_accuracy(response: str, correct_answer: str) -> float:
    """
    Extract lựa chọn A/B/C/D từ response và so sánh với đáp án đúng.

    Xử lý các format phổ biến:
      - "Chọn A", "Đáp án B", "**C**", "chọn c", "A."
      - "Vậy đáp án là C"
      - Bold: "**B. ...**"
    Returns 1.0 nếu đúng, 0.0 nếu sai hoặc không tìm thấy lựa chọn.
    """
    if not correct_answer or not response:
        return 0.0

    correct = correct_answer.strip().upper()
    if correct not in {"A", "B", "C", "D"}:
        return 0.0

    # Scan response for the answer choice
    patterns = [
        r"\b(?:chọn|đáp án|answer|chọn đáp án)\s*[:.]?\s*\**([ABCD])\**",
        r"\*\*([ABCD])[.)]",
        r"^([ABCD])[.)\s]",
        r"\b([ABCD])\b",
    ]

    text = response.upper()
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            found = m.group(1).upper()
            return 1.0 if found == correct else 0.0

    return 0.0


def judge_true_false_accuracy(response: str, correct_answer: str) -> float:
    """
    Đánh giá câu Đúng/Sai dạng "a-T,b-F,c-T,d-F".

    Tìm và so sánh từng mệnh đề a/b/c/d.
    Returns tỷ lệ mệnh đề đúng (0.0–1.0).
    """
    if not correct_answer or not response:
        return 0.0

    def parse_tf(text: str) -> dict[str, bool]:
        """Parse 'a-T,b-F,...' hoặc '(a) Đúng (b) Sai ...' thành dict."""
        result: dict[str, bool] = {}

        # Format chuẩn: a-T, b-F, c-ĐÚNG, d-SAI (case insensitive)
        for m in re.finditer(
            r"\b([abcd])\s*[-:]\s*(T|F|Đ|S|True|False|Đúng|Sai|ĐÚNG|SAI)\b",
            text,
            re.IGNORECASE,
        ):
            key = m.group(1).lower()
            val_raw = m.group(2).upper()
            result[key] = val_raw in {"T", "TRUE", "Đ", "ĐÚNG"}

        # Fallback: "(a) Đúng" or "(a): Đúng"
        if not result:
            for m in re.finditer(
                r"\(([abcd])\)\s*:?\s*(Đúng|Sai|True|False)",
                text,
                re.IGNORECASE,
            ):
                key = m.group(1).lower()
                val_raw = m.group(2).upper()
                result[key] = val_raw in {"ĐÚNG", "TRUE"}

        return result

    ref_parsed  = parse_tf(correct_answer)
    resp_parsed = parse_tf(response)

    if not ref_parsed:
        return 0.0

    correct_count = sum(
        1 for k, v in ref_parsed.items()
        if resp_parsed.get(k) == v
    )
    return round(correct_count / len(ref_parsed), 4)


def judge_short_answer_accuracy(response: str, correct_answer: str) -> float:
    """
    So sánh câu trả lời ngắn (số hoặc chuỗi đơn giản).

    Cách thức:
    1. Thử numeric comparison (làm tròn 4 chữ số)
    2. Normalize string → so sánh exact
    Returns 1.0 nếu đúng, 0.0 nếu sai.
    """
    if not correct_answer or not response:
        return 0.0

    def extract_number(text: str) -> Optional[float]:
        """Tìm số cuối cùng (hoặc duy nhất) trong text."""
        # Thay dấu phẩy châu Âu (4,5) → dấu chấm (4.5)
        text_normalized = re.sub(r"(\d),(\d)", r"\1.\2", text)
        nums = re.findall(r"-?\d+\.?\d*", text_normalized)
        if nums:
            try:
                return float(nums[-1])
            except ValueError:
                return None
        return None

    def normalize_str(text: str) -> str:
        """Lowercase, strip, remove spaces."""
        return re.sub(r"\s+", "", text.strip().lower())

    # Try numeric
    ref_num  = extract_number(correct_answer)
    resp_num = extract_number(response)
    if ref_num is not None and resp_num is not None:
        return 1.0 if abs(ref_num - resp_num) < 0.01 else 0.0

    # Fallback: string exact match
    return 1.0 if normalize_str(response) == normalize_str(correct_answer) else 0.0


async def judge_step_clarity(
    question: str,
    response: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> float:
    """
    Dùng LLM đánh giá sự rõ ràng và đầy đủ bước giải (0.0–1.0).
    Không cần reference — đo chất lượng trình bày thuần túy.
    Returns 0.5 khi lỗi.
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage
        from app.config import get_settings

        settings = get_settings()
        llm = ChatOpenAI(
            model=model,
            api_key=api_key or settings.OPENAI_API_KEY,
            temperature=0.0,
        )

        prompt = STEP_CLARITY_PROMPT.format(
            question=question[:1000],
            response=response[:800],
        )

        result = await llm.ainvoke([HumanMessage(content=prompt)])
        text = result.content.strip()
        m = re.search(r"\d+\.?\d*", text)
        if m:
            score = float(m.group())
            return max(0.0, min(1.0, score))

    except Exception as e:
        logger.warning("Step clarity judge failed: %s", e)

    return 0.5



# ── MathJudge coordinator ──────────────────────────────────────────────────────

class MathJudge:
    """
    Đánh giá toàn bộ một danh sách RAGAS samples bằng custom metrics.

    Parameters
    ----------
    model : str
        OpenAI model dùng cho explanation_quality judge.
    skip_explanation : bool
        Nếu True, bỏ qua LLM call (chỉ chạy accuracy — offline, free).
    """

    THRESHOLDS = {
        "mcq_accuracy":          0.85,
        "true_false_accuracy":   0.75,
        "short_answer_accuracy": 0.70,
        "step_clarity":          0.70,
    }

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        skip_explanation: bool = False,
    ):
        self.model = model
        self.skip_explanation = skip_explanation  # also skips step_clarity

    async def judge_sample(self, sample: dict) -> dict:
        """Judge một sample duy nhất, trả về scores + metadata."""
        meta          = sample.get("_meta", {})
        question_type = meta.get("type", "")
        correct_ans   = meta.get("correct_answer", "")
        response      = sample.get("response", "")
        reference     = sample.get("reference", "")
        question      = sample.get("user_input", "")

        # ── Accuracy (no LLM) ──
        if question_type == "exam_mcq":
            accuracy = judge_mcq_accuracy(response, correct_ans)
        elif question_type == "exam_true_false":
            accuracy = judge_true_false_accuracy(response, correct_ans)
        elif question_type == "exam_short_answer":
            accuracy = judge_short_answer_accuracy(response, correct_ans)
        else:
            accuracy = 0.0
            logger.warning("Unknown question type: %s", question_type)

        # ── Explanation / Step Clarity (LLM, optional) ──
        if self.skip_explanation:
            step_clarity = None
        else:
            step_clarity = await judge_step_clarity(
                question=question,
                response=response,
                model=self.model,
            )

        return {
            "id":                  meta.get("id", ""),
            "skill_id":            meta.get("skill_id", ""),
            "chapter":             meta.get("chapter", ""),
            "question_type":       question_type,
            "correct_answer":      correct_ans,
            "accuracy":            accuracy,
            "step_clarity":        step_clarity,
        }

    async def run(self, samples: list[dict]) -> list[dict]:
        """Judge tất cả samples (concurrent)."""
        tasks = [self.judge_sample(s) for s in samples]
        return await asyncio.gather(*tasks)

    def run_and_report(
        self,
        samples: list[dict],
        output_path: Optional[str] = None,
    ) -> dict:
        """
        Đồng bộ wrapper: judge + tổng hợp báo cáo + lưu JSON.

        Returns
        -------
        dict
            Báo cáo đầy đủ với summary + per-skill breakdown + pass/fail.
        """
        results = asyncio.run(self.run(samples))
        return self._build_report(results, samples, output_path)

    def _build_report(
        self,
        results: list[dict],
        samples: list[dict],
        output_path: Optional[str],
    ) -> dict:
        """Tổng hợp kết quả thành báo cáo."""

        # ── Per-type accuracy (split into 3 separate metrics) ──
        mcq_accs  = [r["accuracy"] for r in results if r["question_type"] == "exam_mcq"]
        tf_accs   = [r["accuracy"] for r in results if r["question_type"] == "exam_true_false"]
        sa_accs   = [r["accuracy"] for r in results if r["question_type"] == "exam_short_answer"]

        summary: dict[str, float] = {}
        if mcq_accs:
            summary["mcq_accuracy"]          = round(sum(mcq_accs) / len(mcq_accs), 4)
        if tf_accs:
            summary["true_false_accuracy"]   = round(sum(tf_accs) / len(tf_accs), 4)
        if sa_accs:
            summary["short_answer_accuracy"] = round(sum(sa_accs) / len(sa_accs), 4)

        # Combined overall (for reference)
        all_accs = mcq_accs + tf_accs + sa_accs
        if all_accs:
            summary["overall_accuracy"] = round(sum(all_accs) / len(all_accs), 4)

        # ── Step clarity (LLM) ──
        clarities = [
            r["step_clarity"]
            for r in results
            if r.get("step_clarity") is not None
        ]
        if clarities:
            summary["step_clarity"] = round(sum(clarities) / len(clarities), 4)

        # ── Per type breakdown ──
        by_type: dict[str, list] = {}
        for r in results:
            t = r["question_type"]
            by_type.setdefault(t, []).append(r["accuracy"])

        type_breakdown = {
            t: round(sum(accs) / len(accs), 4)
            for t, accs in by_type.items()
        }

        # ── Per skill breakdown ──
        by_skill: dict[str, list] = {}
        for r in results:
            s = r["skill_id"] or "unknown"
            by_skill.setdefault(s, []).append(r["accuracy"])

        skill_breakdown = {
            skill: round(sum(accs) / len(accs), 4)
            for skill, accs in by_skill.items()
        }

        # ── Pass/fail ──
        pass_fail = {
            name: "✅ PASS" if score >= self.THRESHOLDS.get(name, 0.70) else "❌ FAIL"
            for name, score in summary.items()
        }

        report = {
            "timestamp":       datetime.now().isoformat(),
            "n_samples":       len(samples),
            "judge_model":     self.model,
            "skip_explanation": self.skip_explanation,
            "summary_scores":  summary,
            "type_breakdown":  type_breakdown,
            "skill_breakdown": skill_breakdown,
            "thresholds":      self.THRESHOLDS,
            "pass_fail":       pass_fail,
            "per_sample":      results,
        }

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("MathJudge report saved → %s", output_path)

        return report


def print_math_report(report: dict) -> None:
    """In báo cáo MathJudge ra console."""
    print("\n" + "=" * 60)
    print("🧮  MATH JUDGE REPORT — AITutor")
    print("=" * 60)
    print(f"⏰  Timestamp   : {report['timestamp']}")
    print(f"📝  Samples     : {report['n_samples']}")
    print(f"🤖  Judge model : {report['judge_model']}")
    print()

    print("── SUMMARY SCORES ────────────────────────────────────────")
    thresholds = report.get("thresholds", {})
    for name, score in report["summary_scores"].items():
        thresh  = thresholds.get(name, 0.70)
        status  = "✅" if score >= thresh else "❌"
        bar_len = int(score * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {status} {name:<25} {score:.4f}  [{bar}]  (threshold: {thresh})")

    print()
    print("── BY QUESTION TYPE ──────────────────────────────────────")
    for qtype, score in report.get("type_breakdown", {}).items():
        status = "✅" if score >= 0.70 else "❌"
        print(f"  {status} {qtype:<30} accuracy: {score:.4f}")

    print()
    print("── SKILL BREAKDOWN (accuracy) ────────────────────────────")
    for skill, score in report.get("skill_breakdown", {}).items():
        status = "✅" if score >= 0.70 else "❌"
        print(f"  {status} {skill:<35} {score:.4f}")

    print()
    print("── PASS/FAIL ─────────────────────────────────────────────")
    for name, verdict in report["pass_fail"].items():
        print(f"  {verdict}  {name}")
    print("=" * 60 + "\n")
