"""
MathContextCoverage — Custom RAGAS-compatible metric

Thay thế LLMContextRecall (vốn đo fact-matching) bằng câu hỏi phù hợp với
bài Toán:

  "Các chunk lý thuyết RAG có cung cấp đủ CÔNG CỤ / PHƯƠNG PHÁP để một học
  sinh có thể tìm ra đáp án mẫu không?"

Tại sao cần custom:
  - context_recall gốc tách ground-truth thành các mệnh đề sự kiện (fact)
    rồi kiểm tra xem mỗi fact có xuất hiện trong context hay không.
  - Với bài Toán, ground-truth chứa toạ độ / kết quả cụ thể (vd. "x = 1")
    nhưng context là LÝ THUYẾT (định nghĩa, công thức) → không bao giờ match
    → context_recall luôn = 0 dù RAG truy xuất đúng.
  - Metric này hỏi đúng hơn: "Context có đủ PHƯƠNG PHÁP để giải không?"

Thang điểm:
  1.0 — Context cung cấp đủ phương pháp / công thức / định nghĩa cần thiết
  0.5 — Context cung cấp một phần (còn thiếu một bước quan trọng)
  0.0 — Context hoàn toàn không liên quan / không hỗ trợ được gì
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Prompt template ────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
Bạn là chuyên gia đánh giá hệ thống RAG cho môn Toán lớp 12.

NHIỆM VỤ:
Đánh giá xem các đoạn tài liệu lý thuyết (CONTEXT) có cung cấp đủ CÔNG CỤ
hoặc PHƯƠNG PHÁP để một học sinh có thể tìm ra đáp án chuẩn hay không.

LƯU Ý QUAN TRỌNG:
- KHÔNG kiểm tra xem context có chứa kết quả / con số / đáp án cụ thể hay không.
- CHỈ kiểm tra xem context có chứa ĐỊNH NGHĨA, CÔNG THỨC, QUY TẮC, PHƯƠNG PHÁP
  cần thiết để thực hiện từng bước giải hay không.
- Với bài có đồ thị / hình ảnh: context chỉ cần cung cấp lý thuyết để ĐỌC ĐỒ THỊ
  (vd. định nghĩa cực trị, tiệm cận), không cần chứa tọa độ cụ thể.

CÂU HỎI:
{question}

ĐÁP ÁN CHUẨN (GROUND TRUTH — chỉ để hiểu bài cần phương pháp gì):
{reference}

CONTEXT (các đoạn lý thuyết RAG truy xuất được):
{context}

THANG ĐIỂM:
- 1.0: Context cung cấp ĐẦY ĐỦ định nghĩa / công thức / phương pháp cần thiết
- 0.5: Context cung cấp MỘT PHẦN (còn thiếu 1 bước hoặc 1 công thức quan trọng)
- 0.0: Context KHÔNG liên quan hoặc thiếu hoàn toàn các công cụ cần thiết

Trả lời theo đúng định dạng sau (KHÔNG giải thích thêm gì):
SCORE: <0.0 hoặc 0.5 hoặc 1.0>
REASON: <1 câu giải thích ngắn gọn bằng tiếng Việt>
"""


# ── Metric class ───────────────────────────────────────────────────────────────

@dataclass
class MathContextCoverage:
    """
    Custom RAGAS-style metric đo mức độ context hỗ trợ phương pháp giải Toán.

    Tương thích với RAGAS EvaluationDataset (dùng các trường:
    user_input, retrieved_contexts, response, reference).

    Parameters
    ----------
    llm : LangchainLLMWrapper
        LLM dùng làm judge (set sau khi khởi tạo, như RAGAS built-in metrics).
    name : str
        Tên metric hiển thị trong kết quả.
    """

    name: str = "math_context_coverage"
    llm: Optional[object] = field(default=None, repr=False)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_prompt(self, user_input: str, retrieved_contexts: list[str], reference: str) -> str:
        context_str = "\n\n---\n\n".join(retrieved_contexts) if retrieved_contexts else "(Không có)"
        return _PROMPT_TEMPLATE.format(
            question=user_input,
            reference=reference,
            context=context_str,
        )

    def _parse_response(self, text: str) -> tuple[float, str]:
        """Parse 'SCORE: X.X\\nREASON: ...' → (score, reason)."""
        score = 0.0
        reason = ""
        for line in text.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("SCORE:"):
                try:
                    score = float(line.split(":", 1)[1].strip())
                    score = max(0.0, min(1.0, score))  # clamp
                except ValueError:
                    score = 0.0
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
        return score, reason

    # ── Score single sample (sync wrapper) ────────────────────────────────────

    def score_sample(
        self,
        user_input: str,
        retrieved_contexts: list[str],
        reference: str,
    ) -> tuple[float, str]:
        """
        Chấm điểm 1 sample. Trả về (score, reason).
        Gọi LLM judge đồng bộ qua .invoke().
        """
        if self.llm is None:
            raise RuntimeError("MathContextCoverage.llm chưa được set.")

        prompt = self._build_prompt(user_input, retrieved_contexts, reference)

        # RAGAS wraps LangChain LLM — gọi trực tiếp langchain_core interface
        try:
            # LangchainLLMWrapper expose .langchain_llm
            inner_llm = getattr(self.llm, "langchain_llm", self.llm)
            from langchain_core.messages import HumanMessage
            resp = inner_llm.invoke([HumanMessage(content=prompt)])
            text = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            logger.warning("MathContextCoverage LLM call failed: %s", e)
            return 0.5, f"(judge error: {e})"

        return self._parse_response(text)

    # ── Batch score (used by RAGASEvaluator) ──────────────────────────────────

    def score_batch(self, samples: list[dict]) -> list[dict]:
        """
        Chấm điểm danh sách samples.

        samples: list of dicts with keys:
          user_input, retrieved_contexts, reference
        Returns:
          list of {math_context_coverage: float, math_context_coverage_reason: str}
        """
        results = []
        for s in samples:
            score, reason = self.score_sample(
                user_input=s.get("user_input", ""),
                retrieved_contexts=s.get("retrieved_contexts", []),
                reference=s.get("reference", ""),
            )
            results.append({
                self.name:              score,
                f"{self.name}_reason":  reason,
            })
        return results
