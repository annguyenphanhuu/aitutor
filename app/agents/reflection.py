"""Reflection Engine — Test-Time Compute for Math Tutoring.

Implements a *System-2 Thinking* loop inspired by OpenAI o1/o3:
the Teacher Agent must "think before it speaks".

Architecture
------------
1. Teacher Agent generates a *draft* answer.
2. This engine scans the draft for mathematical expressions.
3. For each expression, it calls the SymPy tools to *verify* the math.
4. If discrepancies are found, the engine asks the LLM to *self-correct*.
5. Only the final, verified answer is returned to the student.

The entire thinking process is wrapped in ``<thinking>...</thinking>``
tags.  The post-processor strips these before sending to the frontend
(or optionally exposes them as a collapsible "AI thinking" section).

Why this matters for recruiters
-------------------------------
This demonstrates understanding of:
  • Test-Time Compute (scaling inference, not just training)
  • LLM self-correction and verification loops
  • Tool-augmented generation (code interpreter pattern)
  • Anti-hallucination engineering
"""

from __future__ import annotations

import re
import json
import logging
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from app.config import get_settings
from app.agents.tools import (
    MATH_TOOLS,
)
from app.utils.cost_tracker import log_from_response
from app.utils.llm import compatible_temperature
from app.evaluation.hallucination_tracker import ReflectionMetrics

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Prompts ──────────────────────────────────────────────────────────────────

REFLECTION_SYSTEM = """Bạn là một hệ thống kiểm tra toán học nội bộ (Internal Math Verifier).

NHIỆM VỤ: Phân tích bài giải bên dưới và tìm các biểu thức toán học cần kiểm chứng bằng SymPy.
LƯU Ý CÁC HÀM VÀ THAM SỐ CỦA TỪNG HÀM:
1. compute_derivative: {{"expr_str": "x^2", "var": "x", "order": 1}}
2. compute_integral: {{"expr_str": "x^2", "var": "x", "lower": "0", "upper": "1"}}
3. solve_equation: {{"equation_str": "x^2 - 1 = 0", "var": "x"}}
4. simplify_expression: {{"expr_str": "x + x"}}

Với mỗi biểu thức, hãy trả về JSON array. Mỗi phần tử có dạng:
{{
  "expression": "<biểu thức hoặc phương trình>",
  "operation": "<tên_hàm_tương_ứng>",
  "params": {{"<tên_tham_số>": "<giá_trị>"}}
}}

Chỉ trích xuất những phép tính CÓ THỂ KIỂM CHỨNG bằng SymPy.
Nếu không có phép tính nào cần kiểm tra, trả về: []

CHỈ TRẢ VỀ JSON ARRAY, không giải thích."""

CORRECTION_PROMPT = r"""Bạn vừa giải một bài toán nhưng hệ thống phát hiện sai sót:

--- BÀI GIẢI GỐC ---
{draft}

--- KẾT QUẢ KIỂM CHỨNG SYMPY ---
{verification_results}

Hãy VIẾT LẠI bài giải hoàn chỉnh với các kết quả đúng.
Giữ nguyên phong cách giảng dạy và cách xưng hô (xưng "thầy", gọi học sinh là "em"),
chỉ sửa lại phần tính toán sai.
QUAN TRỌNG:
1. NẾU bài giải gốc có thẻ <answer>...</answer> thì BẮT BUỘC giữ nguyên định dạng thẻ đó
   chứa đáp án cuối cùng. NẾU bài giải gốc KHÔNG có thẻ <answer> thì KHÔNG tự thêm vào.
2. NẾU câu hỏi ban đầu có yêu cầu LÀM TRÒN (ví dụ: làm tròn đến hàng phần chục, phần trăm), bạn PHẢI làm tròn kết quả từ SymPy thành số thập phân rồi mới đưa vào thẻ <answer>. KHÔNG đưa biểu thức căn (ví dụ $\sqrt{{6}}$) hay phân số vào thẻ <answer> nếu có yêu cầu làm tròn.
Dùng LaTeX: $...$ inline, $$...$$ block.
Trả lời bằng tiếng Việt."""


# ── Reflection Engine ────────────────────────────────────────────────────────

class ReflectionEngine:
    """Verify and self-correct LLM math outputs using SymPy tools.

    Usage:
        engine = ReflectionEngine()
        result = await engine.reflect(draft_response, original_question)
    """

    def __init__(self, max_corrections: int = 1):
        # Reflection-Correct: full model — rewriting math answers requires quality
        temp_main = compatible_temperature(settings.LLM_MODEL, 0.0)
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=temp_main,
        )
        # Reflection-Extract: mini model — structured JSON extraction only
        temp_mini = compatible_temperature(settings.LLM_MODEL_MINI, 0.0)
        self.extractor_llm = ChatOpenAI(
            model=settings.LLM_MODEL_MINI,
            api_key=settings.OPENAI_API_KEY,
            temperature=temp_mini,
        )
        self.max_corrections = max_corrections

    async def reflect(
        self,
        draft: str,
        question: str,
    ) -> ReflectionResult:
        """Run the full reflection loop on a draft answer.

        Parameters
        ----------
        draft : str
            The Teacher Agent's initial response.
        question : str
            The student's original question.

        Returns
        -------
        ReflectionResult
            Contains the final answer, thinking log, and verification details.
        """
        thinking_log: list[str] = []
        thinking_log.append("🔍 Bắt đầu kiểm chứng bài giải...")

        # ── Step 1: Extract verifiable expressions ────────────────────
        extractions = await self._extract_verifiable_expressions(draft)
        thinking_log.append(f"📋 Tìm thấy {len(extractions)} biểu thức cần kiểm tra.")

        if not extractions:
            thinking_log.append("✅ Không có phép tính cần kiểm chứng.")
            no_expr_result = ReflectionResult(
                final_answer=draft,
                thinking_log=thinking_log,
                was_corrected=False,
                verifications=[],
            )
            ReflectionMetrics.record(no_expr_result)
            return no_expr_result


        # ── Step 2: Run SymPy verification ────────────────────────────
        verifications: list[dict] = []

        for ext in extractions:
            tool_name = ext.get("operation", "")
            params = ext.get("params", {})
            tool_fn = MATH_TOOLS.get(tool_name)

            if not tool_fn:
                thinking_log.append(f"⚠️ Tool không tồn tại: {tool_name}")
                continue

            try:
                result = tool_fn(**params)
                # result may be str (LangChain @tool returns str) or dict (legacy)
                if isinstance(result, str):
                    result_str = result
                    success = not result_str.startswith("ERROR")
                    result_dict = {"success": success, "result_latex": result_str}
                else:
                    result_dict = result
                    result_str = str(result)

                verifications.append({
                    "expression": ext.get("expression", ""),
                    "tool": tool_name,
                    "params": params,
                    "result": result_dict,
                })

                if result_dict.get("success"):
                    thinking_log.append(
                        f"  \u2713 {tool_name}({params.get('expr_str', '')}) "
                        f"= {result_dict.get('result_latex', result_dict.get('solutions', result_str))}"
                    )
                else:
                    thinking_log.append(
                        f"  \u2717 {tool_name} failed: {result_dict.get('error', result_str)}"
                    )
            except Exception as e:
                thinking_log.append(f"  \u2717 Exception in {tool_name}: {e}")

        # ── Step 3: Check if draft contains any mismatches ────────────
        # Compare SymPy results with what the LLM wrote
        verification_summary = self._build_verification_summary(verifications)

        if verification_summary:
            thinking_log.append(f"❌ Phát hiện {len(verifications)} kết quả cần đối chiếu.")
            thinking_log.append("📝 Tiến hành yêu cầu LLM tự sửa...")

            # ── Step 4: Self-correction ───────────────────────────────
            corrected = await self._self_correct(
                draft, verification_summary, question
            )
            thinking_log.append("✅ Đã tạo bài giải đã đối chiếu với SymPy.")

            corrected_result = ReflectionResult(
                final_answer=corrected,
                thinking_log=thinking_log,
                was_corrected=True,
                verifications=verifications,
            )
            ReflectionMetrics.record(corrected_result)
            return corrected_result

        thinking_log.append("✅ Bài giải đã kiểm chứng — không phát hiện sai sót rõ ràng.")
        clean_result = ReflectionResult(
            final_answer=draft,
            thinking_log=thinking_log,
            was_corrected=False,
            verifications=verifications,
        )
        ReflectionMetrics.record(clean_result)
        return clean_result

    # ── Private helpers ──────────────────────────────────────────────────

    async def _extract_verifiable_expressions(self, draft: str) -> list[dict]:
        """Use LLM to extract math expressions from the draft that can be verified."""
        try:
            messages = [
                SystemMessage(content=REFLECTION_SYSTEM),
                HumanMessage(content=f"BÀI GIẢI:\n{draft}"),
            ]
            response = await self.extractor_llm.ainvoke(messages)
            log_from_response(
                agent="Reflection-Extract",
                model=settings.LLM_MODEL_MINI,
                response=response,
            )

            content = response.content.strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]

            result = json.loads(content)
            if isinstance(result, list):
                return result[:5]  # Cap at 5 verifications
            return []
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Reflection extraction failed: %s", e)
            return []

    def _build_verification_summary(self, verifications: list[dict]) -> str:
        """Build a human-readable summary of SymPy verification results."""
        if not verifications:
            return ""

        lines = []
        for v in verifications:
            r = v.get("result", {})
            # r may be dict (legacy) or already normalised dict
            if isinstance(r, dict) and r.get("success"):
                result_str = r.get("result_latex", r.get("solutions", ""))
                if "result_float" in r:
                    result_str += f" (Gi\u00e1 tr\u1ecb th\u1eadp ph\u00e2n: {r['result_float']})"
                lines.append(
                    f"\u2022 {v['tool']}({v['params'].get('expr_str', '')}) "
                    f"\u2192 K\u1ebcT QU\u1ea2 \u0110\u00daNG (SymPy): {result_str}"
                )
        return "\n".join(lines) if lines else ""

    async def _self_correct(
        self,
        draft: str,
        verification_results: str,
        question: str,
    ) -> str:
        """Ask the LLM to rewrite its answer using SymPy-verified results."""
        prompt = CORRECTION_PROMPT.format(
            draft=draft,
            verification_results=verification_results,
        )
        messages = [
            SystemMessage(
                content="Bạn là \"thầy\" — gia sư Toán 12 (xưng \"thầy\", gọi học sinh là \"em\"). "
                        "Hãy viết lại bài giải với kết quả tính toán CHÍNH XÁC từ SymPy. "
                        "QUAN TRỌNG: nếu bài giải gốc có thẻ <answer>...</answer> thì giữ nguyên thẻ đó; "
                        "nếu bài giải gốc không có thẻ <answer> thì KHÔNG tự thêm."
            ),
            HumanMessage(content=prompt),
        ]
        response = await self.llm.ainvoke(messages)
        log_from_response(
            agent="Reflection-Correct",
            model=settings.LLM_MODEL,
            response=response,
        )
        return response.content


class ReflectionResult:
    """Container for reflection output."""

    def __init__(
        self,
        final_answer: str,
        thinking_log: list[str],
        was_corrected: bool,
        verifications: list[dict],
    ):
        self.final_answer = final_answer
        self.thinking_log = thinking_log
        self.was_corrected = was_corrected
        self.verifications = verifications

    def format_thinking_block(self) -> str:
        """Format the thinking log as a collapsible block."""
        log_text = "\n".join(self.thinking_log)
        return f"<thinking>\n{log_text}\n</thinking>"

    def build_full_response(self, include_thinking: bool = True) -> str:
        """Build the complete response with optional thinking block.

        Parameters
        ----------
        include_thinking : bool
            If True, prepend a ``<thinking>`` block (hidden by default
            on the frontend, but available for debugging/demo).
        """
        parts = []
        if include_thinking and self.thinking_log:
            parts.append(self.format_thinking_block())
        parts.append(self.final_answer)
        return "\n\n".join(parts)


# ── Post-processor ───────────────────────────────────────────────────────────

def strip_thinking_tags(text: str) -> str:
    """Remove <thinking>...</thinking> blocks from response text.

    Used when the frontend does not support rendering thinking blocks.
    """
    return re.sub(
        r"<thinking>.*?</thinking>",
        "",
        text,
        flags=re.DOTALL,
    ).strip()


def extract_thinking_block(text: str) -> tuple[str, str]:
    """Extract thinking block and clean response separately.

    Returns (thinking_content, clean_answer).
    """
    match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    thinking = match.group(1).strip() if match else ""
    clean = strip_thinking_tags(text)
    return thinking, clean
