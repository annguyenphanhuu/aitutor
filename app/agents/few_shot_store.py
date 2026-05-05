"""Dynamic Few-Shot Store — BKT-driven pedagogical example selection.

This module maintains a curated library of *exemplary teaching
conversations* categorised by mastery level.  When the Teacher Agent
is about to respond, it queries this store to find a few-shot example
that matches the student's current BKT mastery level.

How it works
------------
1. We pre-seed a set of ``PedagogicalExample`` objects — each is a
   short (question, exemplary_response) pair tagged with a mastery
   tier (beginner / developing / proficient / mastered).

2. At inference time, the engine looks up the student's float mastery
   score, maps it to a tier, and retrieves the best-matching example.

3. The example is injected into the System Prompt as a *few-shot
   demonstration*, so the LLM naturally mimics the teaching tone,
   depth, and pacing of that example.

Effect on the LLM
------------------
  • **Beginner** (~0.1–0.3): slow pace, encouraging tone, break into
    tiny steps, use analogies, avoid jargon.
  • **Developing** (~0.3–0.6): moderate pace, introduce formal
    notation gradually, ask guiding questions.
  • **Proficient** (~0.6–0.85): concise, focus on edge cases and
    exam tricks, reference related skills.
  • **Mastered** (~0.85–1.0): terse, give the fastest solution path,
    suggest challenging extensions.

Why this impresses recruiters
-----------------------------
Demonstrates the "Data Engine" mindset: the AI system *adapts its
behaviour* dynamically based on user data, not just hard-coded rules.
This is the pattern used by Duolingo, Khan Academy, and OpenAI's
internal RLHF pipelines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class PedagogicalExample:
    """A single teaching demonstration for few-shot injection."""
    tier: str                # "beginner" | "developing" | "proficient" | "mastered"
    chapter: str             # e.g. "Đạo hàm", "Tích phân", "general"
    student_question: str    # What the student asked
    teacher_response: str    # Exemplary teacher answer
    style_notes: str         # Meta-description of the teaching style


# ── Pre-seeded Examples ──────────────────────────────────────────────────────
# In a production system these would live in a vector DB.  For now we
# hard-code a carefully curated set (much easier to control quality).

_EXAMPLES: list[PedagogicalExample] = [
    # ─────────── BEGINNER ────────────
    PedagogicalExample(
        tier="beginner",
        chapter="general",
        student_question="Tính đạo hàm của f(x) = x³ + 2x",
        teacher_response=(
            "Chào em! 😊 Mình cùng nhau tìm hiểu từng bước nhé.\n\n"
            "**Bước 1:** Nhớ lại công thức đạo hàm cơ bản:\n"
            "$$\\frac{d}{dx}(x^n) = n \\cdot x^{n-1}$$\n"
            "Em có thể nhớ quy tắc này bằng cách: *\"Hạ số mũ xuống làm hệ số, rồi trừ mũ đi 1\"*.\n\n"
            "**Bước 2:** Áp dụng cho từng phần:\n"
            "- Với $x^3$: mũ là 3, nên đạo hàm = $3 \\cdot x^{3-1} = 3x^2$\n"
            "- Với $2x$: nghĩa là $2 \\cdot x^1$, nên đạo hàm = $2 \\cdot 1 \\cdot x^0 = 2$\n\n"
            "**Kết quả:** $f'(x) = 3x^2 + 2$\n\n"
            "💡 Em thử tự tính đạo hàm của $g(x) = x^4 + 5x$ xem sao nhé?"
        ),
        style_notes="Chậm rãi, chia nhỏ từng bước, dùng emoji khuyến khích, kết thúc bằng bài tập nhỏ",
    ),
    PedagogicalExample(
        tier="beginner",
        chapter="Nguyên hàm và Tích phân",
        student_question="Tích phân là gì?",
        teacher_response=(
            "Câu hỏi rất hay! 🌟 Mình giải thích đơn giản nhé.\n\n"
            "Em hình dung **đạo hàm** như việc cắt một cái bánh thành nhiều lát mỏng.\n"
            "Thì **tích phân** chính là việc ghép các lát mỏng lại thành cái bánh hoàn chỉnh!\n\n"
            "Nói cách khác:\n"
            "- Đạo hàm: cho biết tốc độ thay đổi → *\"Xe chạy nhanh bao nhiêu?\"*\n"
            "- Tích phân: tính tổng tích lũy → *\"Xe đã chạy được bao xa?\"*\n\n"
            "Về mặt toán học:\n"
            "$$\\int f(x) \\, dx = F(x) + C$$\n"
            "trong đó $F'(x) = f(x)$ (tức là $F$ là nguyên hàm của $f$).\n\n"
            "Em đã hiểu ý tưởng chưa? Nếu rồi, mình sẽ làm ví dụ cụ thể nhé! 😊"
        ),
        style_notes="Dùng phép tương tự (analogy) đời thường, không vội vàng, kiểm tra hiểu biết",
    ),

    # ─────────── DEVELOPING ────────────
    PedagogicalExample(
        tier="developing",
        chapter="general",
        student_question="Tính đạo hàm của f(x) = sin(2x) · eˣ",
        teacher_response=(
            "Đây là bài áp dụng **quy tắc đạo hàm tích** — em còn nhớ công thức không?\n\n"
            "$$[u \\cdot v]' = u' \\cdot v + u \\cdot v'$$\n\n"
            "Hãy thử tự xác định:\n"
            "- $u = \\sin(2x)$ → $u' = ?$ (gợi ý: cần đạo hàm hàm hợp)\n"
            "- $v = e^x$ → $v' = ?$\n\n"
            "Em thử tính rồi gửi lại cho mình kiểm tra nhé!"
        ),
        style_notes="Gợi ý qua câu hỏi Socratic, nhắc công thức nhưng không giải hộ, mong đợi em tự làm",
    ),
    PedagogicalExample(
        tier="developing",
        chapter="Tổ hợp – Xác suất",
        student_question="Bốc 3 quả bóng từ hộp có 5 đỏ, 4 xanh. XS cả 3 đỏ?",
        teacher_response=(
            "Bài này dùng **xác suất cổ điển** kết hợp **tổ hợp**.\n\n"
            "Em hãy suy nghĩ theo 2 bước:\n"
            "1. **Tổng số cách** bốc 3 quả từ $5 + 4 = 9$ quả: $C_9^3 = ?$\n"
            "2. **Số cách thuận lợi** (cả 3 đỏ): $C_5^3 = ?$\n\n"
            "→ $P = \\frac{C_5^3}{C_9^3}$\n\n"
            "Em tính cụ thể ra rồi gửi kết quả nhé. Nhớ rút gọn phân số!"
        ),
        style_notes="Cấu trúc hóa bước giải, dùng câu hỏi gợi mở, yêu cầu tự tính",
    ),

    # ─────────── PROFICIENT ────────────
    PedagogicalExample(
        tier="proficient",
        chapter="general",
        student_question="Tính tích phân ∫₀¹ x·eˣ dx",
        teacher_response=(
            "Tích phân từng phần: $\\int u \\, dv = uv - \\int v \\, du$\n\n"
            "Đặt $u = x$, $dv = e^x dx$:\n"
            "$$\\int_0^1 xe^x dx = [xe^x]_0^1 - \\int_0^1 e^x dx = e - (e - 1) = 1$$\n\n"
            "💡 Khi thi, gặp dạng $\\int x^n e^x dx$, luôn đặt $u = x^n$ (hạ bậc dần)."
        ),
        style_notes="Ngắn gọn, đi thẳng vào cách giải, thêm mẹo thi",
    ),

    # ─────────── MASTERED ────────────
    PedagogicalExample(
        tier="mastered",
        chapter="general",
        student_question="Tìm GTLN của f(x) = x³ - 3x + 1 trên [0, 2]",
        teacher_response=(
            "$f'(x) = 3x^2 - 3 = 0 \\Rightarrow x = 1 \\in [0,2]$\n\n"
            "$f(0) = 1,\\ f(1) = -1,\\ f(2) = 3$\n\n"
            "**GTLN = 3** tại $x = 2$.\n\n"
            "🔥 Mở rộng: Nếu đề hỏi \"tham số $m$ để $f(x) = m$ có 3 nghiệm trên $\\mathbb{R}$\"? "
            "→ Cần $f_{CĐ} > m > f_{CT}$, tức $-1 < m < 3$."
        ),
        style_notes="Cực ngắn, chỉ kết quả + mở rộng nâng cao",
    ),
]


# ── Few-Shot Selector ────────────────────────────────────────────────────────

def get_mastery_tier(p_mastery: float) -> str:
    """Map a BKT mastery probability to a pedagogical tier."""
    if p_mastery < 0.3:
        return "beginner"
    elif p_mastery < 0.6:
        return "developing"
    elif p_mastery < 0.85:
        return "proficient"
    else:
        return "mastered"


def select_few_shot(
    p_mastery: float,
    chapter: Optional[str] = None,
) -> Optional[PedagogicalExample]:
    """Select the best few-shot example for the given mastery level.

    Tries to match both tier AND chapter.  Falls back to same-tier
    with chapter="general" if no chapter-specific example exists.
    """
    tier = get_mastery_tier(p_mastery)

    # Try chapter-specific first
    if chapter:
        for ex in _EXAMPLES:
            if ex.tier == tier and ex.chapter == chapter:
                return ex

    # Fallback: same tier, any chapter (prefer "general")
    general_match = None
    any_match = None
    for ex in _EXAMPLES:
        if ex.tier == tier:
            if ex.chapter == "general":
                general_match = ex
            any_match = ex

    return general_match or any_match


def build_few_shot_prompt(
    p_mastery: float,
    chapter: Optional[str] = None,
) -> str:
    """Build a few-shot instruction block for injection into System Prompt.

    Returns an empty string if no matching example is found.
    """
    example = select_few_shot(p_mastery, chapter)
    if not example:
        return ""

    tier_descriptions = {
        "beginner": "CỰC KỲ KIÊN NHẪN, chia nhỏ từng bước, dùng ví dụ đời thường, khuyến khích nhiều",
        "developing": "GỢI MỞ qua câu hỏi Socratic, nhắc công thức nhưng để em tự làm",
        "proficient": "NGẮN VÀ CHÍNH XÁC, đi thẳng vào vấn đề, chia sẻ mẹo thi",
        "mastered": "CỰC NGẮN, chỉ kết quả + gợi ý mở rộng nâng cao",
    }

    style_desc = tier_descriptions.get(example.tier, "")

    return (
        f"\n\n── PHONG CÁCH SƯ PHẠM (Dynamic Few-Shot) ──\n"
        f"Mức độ thành thạo hiện tại: {example.tier.upper()} ({p_mastery:.0%})\n"
        f"Phong cách yêu cầu: {style_desc}\n\n"
        f"📝 VÍ DỤ MẪU — Hãy bắt chước giọng điệu và cách tiếp cận sau:\n"
        f"[Học sinh hỏi]: {example.student_question}\n"
        f"[Gia sư trả lời]:\n{example.teacher_response}\n"
        f"── HẾT VÍ DỤ ──\n"
    )
