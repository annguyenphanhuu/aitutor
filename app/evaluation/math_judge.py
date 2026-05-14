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

# ── LLM-as-Examiner prompt (4-criterion rubric) ─────────────────────────────

# ── LLM-as-Examiner: 3-criterion prompt (accuracy is deterministic, not LLM) ──
EXAMINER_PROMPT = """\
Bạn là giáo viên Toán 12 Việt Nam đang chấm phương pháp giải của gia sư AI.
Lưu ý: Tiêu chí ACCURACY đã được chấm điểm tự động bởi hệ thống, bạn chỉ cần đánh giá 3 tiêu chí dưới đây:

1. METHOD (Phương pháp): Cách giải có đúng với lý thuyết Toán 12 không?
   1.0=phương pháp hoàn toàn đúng | 0.5=đúng hướng nhưng có sai sót nhỏ | 0.0=sai phương pháp

2. STEPS (Trình bày): Các bước giải có đầy đủ, logic, dễ theo không?
   1.0=rõ ràng đầy đủ | 0.75=thiếu 1-2 bước nhỏ | 0.5=quá tóm tắt | 0.25=rời rạc | 0.0=không có

3. KNOWLEDGE_USE (Vận dụng lý thuyết): Có nêu/áp dụng đúng định lý/công thức liên quan không?
   1.0=trích dẫn và áp dụng đúng | 0.5=áp dụng nhưng không nêu rõ | 0.0=không vận dụng

Câu hỏi:
{question}

Đáp án chuẩn (tham khảo cho METHOD/STEPS):
{ground_truth}

Câu trả lời AI:
{response}

Trả về JSON hợp lệ duy nhất (không thêm markdown, không giải thích):
{{"method": <0-1>, "steps": <0-1>, "knowledge_use": <0-1>, "comment": "<nhận xét ngắn về phương pháp và trình bày>"}}
"""

# ── Visual reasoning judge prompt ────────────────────────────────────────────

VISUAL_REASONING_PROMPT = """\
Bạn là giáo viên Toán 12 đang đánh giá khả năng đọc đồ thị/hình vẽ của gia sư AI.

Câu hỏi yêu cầu đọc đồ thị/hình vẽ:
{question}

Đáp án chuẩn: {ground_truth}
Câu trả lời AI: {response}

Đánh giá xem AI có:
1. Đọc đúng tọa độ/giá trị từ hình không?
2. Xác định đúng đặc điểm cần tìm (cực trị, tiệm cận, giao điểm...) không?
3. Lý luận dựa trên hình ảnh có nhất quán không?

Chỉ trả lời đúng 1 số trong [0.0, 1.0]:
1.0=đọc hình chính xác và lý luận đúng | 0.5=đọc hình có sai sót nhỏ | 0.0=không đọc được hình
"""



# ── Low-level judge functions (no LLM) ────────────────────────────────────────

def judge_mcq_accuracy(response: str, correct_answer: str) -> float:
    """
    Extract lựa chọn A/B/C/D từ response và so sánh với đáp án đúng.
    Ưu tiên 1: Lấy đáp án trong thẻ <answer>X</answer>
    Dự phòng: So sánh regex (có bỏ dấu tiếng Việt và chỉ quét đoạn đầu)
    Returns 1.0 nếu đúng, 0.0 nếu sai hoặc không tìm thấy lựa chọn.
    """
    if not correct_answer or not response:
        return 0.0

    correct = correct_answer.strip().upper()
    if correct not in {"A", "B", "C", "D"}:
        return 0.0

    # Ưu tiên số 1: Tìm đúng thẻ <answer>X</answer>
    m = re.search(r"<answer>\s*([ABCD])\s*</answer>", response, re.IGNORECASE)
    if m:
        return 1.0 if m.group(1).upper() == correct else 0.0

    # Fallback cho định dạng cũ (giới hạn quét ở đoạn đầu để tránh bắt nhầm phần giải thích)
    import unicodedata
    def remove_vietnamese_accents(s: str) -> str:
        s = unicodedata.normalize('NFD', s)
        s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
        return s.replace('đ', 'd').replace('Đ', 'D')
    
    head_text = remove_vietnamese_accents(response).upper()[:80]
    
    patterns = [
        r"\b(?:CHON|DAP AN|ANSWER)\s*[:.]?\s*\**([ABCD])\**",
        r"\*\*([ABCD])[.)]",
        r"^([ABCD])[.)\s]",
    ]

    for pat in patterns:
        m_fallback = re.search(pat, head_text, re.MULTILINE)
        if m_fallback:
            return 1.0 if m_fallback.group(1) == correct else 0.0

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

    def extract_number(text: str, is_response: bool = False) -> Optional[float]:
        """Tìm số cuối cùng (hoặc duy nhất) trong text."""
        # Ưu tiên lấy trong thẻ <answer> nếu là response
        if is_response:
            m = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
            if m:
                text = m.group(1)
                
        # Thay dấu phẩy châu Âu (4,5) → dấu chấm (4.5)
        text_normalized = re.sub(r"(\d),(\d)", r"\1.\2", text)
        
        # Thử parse latex fraction \dfrac{a}{b} hoặc \frac{a}{b}
        frac_m = re.findall(r"\\(?:d)?frac\s*\{(-?\d+\.?\d*)\}\s*\{(-?\d+\.?\d*)\}", text_normalized)
        if frac_m:
            try:
                a, b = float(frac_m[-1][0]), float(frac_m[-1][1])
                if b != 0:
                    return a / b
            except ValueError:
                pass
                
        # Thử parse phân số thường a/b
        frac_simple = re.findall(r"(-?\d+\.?\d*)\s*/\s*(-?\d+\.?\d*)", text_normalized)
        if frac_simple:
            try:
                a, b = float(frac_simple[-1][0]), float(frac_simple[-1][1])
                if b != 0:
                    return a / b
            except ValueError:
                pass
                
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
    resp_num = extract_number(response, is_response=True)
    if ref_num is not None and resp_num is not None:
        if abs(ref_num - resp_num) < 0.01:
            return 1.0
        # Kiểm tra tương đương % ↔ decimal:
        # VD: đáp án "0,56" (decimal) nhưng model trả "56" (phần trăm) → vẫn đúng.
        # Áp dụng khi ref nằm trong [0,1] và resp = ref × 100 (hoặc ngược lại).
        if 0.0 <= ref_num <= 1.0 and abs(ref_num * 100 - resp_num) < 0.1:
            return 1.0
        if 0.0 <= resp_num <= 1.0 and abs(resp_num * 100 - ref_num) < 0.1:
            return 1.0
        return 0.0

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


# ── LLM-as-Examiner ─────────────────────────────────────────────────────────

def _build_examiner_ground_truth(
    correct_answer: str,
    ground_truth: str,
    question_type: str,
) -> str:
    """
    Đặt đáp án chính xác (correct_answer) lên đầu để LLM examiner
    không bị nhầm lẫn bởi phong cách trình bày khi chấm ACCURACY.

    Với câu T/F: thêm chú thích rõ ràng về format a-T/F.
    """
    if not correct_answer:
        return ground_truth[:500]

    if question_type == "exam_true_false":
        header = (
            f"ĐÁP ÁN CHÍNH XÁC (dùng để chấm ACCURACY): {correct_answer}\n"
            f"(Format: a/b/c/d — T=Đúng, F=Sai. Đúng hoàn toàn khi khớp cả 4 mệnh đề.)\n\n"
            f"Lời giải tham khảo (chỉ dùng cho METHOD/STEPS/KNOWLEDGE_USE):\n"
        )
    elif question_type == "exam_mcq":
        header = (
            f"ĐÁP ÁN CHÍNH XÁC (dùng để chấm ACCURACY): {correct_answer}\n\n"
            f"Lời giải tham khảo:\n"
        )
    else:
        header = f"ĐÁP ÁN CHÍNH XÁC: {correct_answer}\n\nLời giải:\n"

    return header + ground_truth[:400]


def _safe_response_for_examiner(response: str) -> str:
    """
    Truncate response nhưng LUÔN giữ lại phần cuối (chứa KET QUA).
    Tránh cắt mất dòng đáp án cuối cùng.
    """
    MAX = 2200
    if len(response) <= MAX:
        return response
    head = response[:MAX - 300]
    tail = response[-300:]   # Phần cuối chứa KET QUA / đáp án
    return head + "\n...[nội dung giữa đã rút gọn]...\n" + tail


async def judge_examiner(
    question: str,
    response: str,
    ground_truth: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    correct_answer: str = "",
    question_type: str = "",
    deterministic_accuracy: Optional[float] = None,
) -> dict:
    """
    Đánh giá câu trả lời theo rubric 4 chiều của giáo viên Toán 12.

    Kiến trúc mới:
    - accuracy     → được inject từ MathJudge (deterministic), KHÔNG dùng LLM
    - method       → LLM judge
    - steps        → LLM judge
    - knowledge_use → LLM judge

    Trẍng số: accuracy=0.40, method=0.30, steps=0.20, knowledge_use=0.10
    """
    WEIGHTS = {"accuracy": 0.40, "method": 0.30, "steps": 0.20, "knowledge_use": 0.10}
    default = {k: 0.5 for k in WEIGHTS}
    default.update({"comment": "", "weighted_score": 0.5, "error": None})

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

        examiner_gt = _build_examiner_ground_truth(
            correct_answer=correct_answer,
            ground_truth=ground_truth,
            question_type=question_type,
        )

        # Nếu có deterministic_accuracy, thêm thông tin vào đầu prompt
        # giúp LLM examiner không đánh giá thấp METHOD khi kết quả đã được xác nhận đúng.
        acc_note = ""
        if deterministic_accuracy is not None:
            if deterministic_accuracy >= 1.0:
                acc_note = "[HỆ THỐNG XÁC NHẬN: Đáp án cuối cùng ĐÚNG HOÀN TOÀN — chấm METHOD/STEPS/KNOWLEDGE_USE dựa trên lập luận, không nghi ngờ kết quả.]\n\n"
            elif deterministic_accuracy >= 0.5:
                acc_note = f"[HỆ THỐNG XÁC NHẬN: Đáp án đúng một phần ({deterministic_accuracy:.0%}) — đánh giá METHOD/STEPS/KNOWLEDGE_USE dựa trên quy trình.]\n\n"
            else:
                acc_note = "[HỆ THỐNG XÁC NHẬN: Đáp án sai — tập trung đánh giá METHOD/STEPS/KNOWLEDGE_USE dựa trên quy trình suy luận.]\n\n"

        prompt = acc_note + EXAMINER_PROMPT.format(
            question=question[:1200],
            ground_truth=examiner_gt,
            response=_safe_response_for_examiner(response),
        )

        result = await llm.ainvoke([HumanMessage(content=prompt)])
        text = result.content.strip()

        # Tách JSON khỏi markdown nếu có
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            raise ValueError(f"Không tìm thấy JSON trong response: {text[:200]}")

        parsed = json.loads(json_match.group())

        # LLM chỉ chấm 3 tiêu chí METHOD / STEPS / KNOWLEDGE_USE
        llm_criteria = ["method", "steps", "knowledge_use"]
        scores = {
            k: max(0.0, min(1.0, float(parsed.get(k, 0.5))))
            for k in llm_criteria
        }

        # ACCURACY: dùng deterministic score nếu có, fallback sang LLM
        if deterministic_accuracy is not None:
            scores["accuracy"] = float(deterministic_accuracy)
        else:
            raw_acc = parsed.get("accuracy")
            scores["accuracy"] = max(0.0, min(1.0, float(raw_acc))) if raw_acc is not None else 0.5

        weighted = sum(scores[k] * w for k, w in WEIGHTS.items())

        return {
            **scores,
            "comment":        parsed.get("comment", ""),
            "weighted_score": round(weighted, 4),
            "error":          None,
        }

    except Exception as e:
        logger.warning("LLMExaminer failed: %s", e)
        default["error"] = str(e)
        return default


async def judge_visual_reasoning(
    question: str,
    response: str,
    ground_truth: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> float:
    """
    Đánh giá khả năng đọc đồ thị/hình vẽ của AI (0.0–1.0).
    Chỉ gọi khi has_image=True. Thay thế RAGAS faithfulness cho Vision questions.
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

        prompt = VISUAL_REASONING_PROMPT.format(
            question=question[:1000],
            ground_truth=ground_truth[:300],
            response=response[:800],
        )

        result = await llm.ainvoke([HumanMessage(content=prompt)])
        text = result.content.strip()
        m = re.search(r"\d+\.?\d*", text)
        if m:
            return max(0.0, min(1.0, float(m.group())))

    except Exception as e:
        logger.warning("Visual reasoning judge failed: %s", e)

    return 0.5


async def judge_retrieval_quality(
    question: str,
    contexts: list[str],
    ground_truth: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> dict:
    """
    Đánh giá chất lượng RAG retrieval mà không cần RAGAS.
    Dùng LLM để đánh giá:
      - relevance: context có liên quan đến câu hỏi không?
      - coverage: context có đủ thông tin để trả lời không?

    Returns {"relevance": 0-1, "coverage": 0-1, "error": None|str}
    """
    RETRIEVAL_PROMPT = """\
Bạn là chuyên gia đánh giá hệ thống RAG cho Toán 12.

Câu hỏi: {question}
Đáp án chuẩn: {ground_truth}

Các đoạn tài liệu được truy xuất:
{contexts}

Đánh giá 2 tiêu chí (trả về JSON, không giải thích thêm):
1. relevance (0-1): Tài liệu có liên quan đến câu hỏi không?
   1.0=rất liên quan | 0.5=có liên quan một phần | 0.0=không liên quan
2. coverage (0-1): Tài liệu có đủ thông tin để trả lời đúng không?
   1.0=đủ | 0.5=một phần | 0.0=thiếu hoàn toàn

{{"relevance": <0-1>, "coverage": <0-1>}}"""

    default = {"relevance": 0.5, "coverage": 0.5, "error": None}
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

        ctx_text = "\n---\n".join(
            f"[{i+1}] {c[:400]}" for i, c in enumerate(contexts[:5])
        )
        prompt = RETRIEVAL_PROMPT.format(
            question=question[:800],
            ground_truth=ground_truth[:300],
            contexts=ctx_text,
        )

        result = await llm.ainvoke([HumanMessage(content=prompt)])
        text = result.content.strip()
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return {
                "relevance": max(0.0, min(1.0, float(parsed.get("relevance", 0.5)))),
                "coverage":  max(0.0, min(1.0, float(parsed.get("coverage",  0.5)))),
                "error":     None,
            }
    except Exception as e:
        logger.warning("Retrieval quality judge failed: %s", e)
        default["error"] = str(e)

    return default
