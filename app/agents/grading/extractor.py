"""GradingExtractor — trích xuất đề bài + đáp án của học sinh thành dạng chuẩn.

Bước đầu của grading pipeline. LLM chỉ TRÍCH XUẤT (không chấm): xác định
tin nhắn có chứa bài làm của CHÍNH học sinh hay không, và nếu có thì đưa
đề + đáp án về dạng sympy-parseable để verifier kiểm chứng deterministic.

Fail-safe: mọi lỗi (API, parse, validation) → gradable=False (abstain) —
không chấm, không ghi BKT.
"""

from __future__ import annotations

import logging

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agents.contracts import GradingExtraction
from app.config import get_settings
from app.utils.cost_tracker import log_from_response
from app.utils.llm import compatible_temperature

logger = logging.getLogger(__name__)
settings = get_settings()

EXTRACTOR_SYSTEM_PROMPT = """Bạn là bộ trích xuất dữ liệu chấm bài Toán 12. Bạn KHÔNG chấm điểm — chỉ trích xuất.

Nhiệm vụ: từ tin nhắn của học sinh (và đề bài gốc nếu có), xác định:

0. XÁC ĐỊNH ĐỀ BÀI: nếu TIN NHẮN của học sinh chứa một đề bài mới (dấu hiệu:
   "em làm thêm bài", "còn bài này", "bài khác", hoặc đề bài + đáp án nằm cùng
   trong tin nhắn), thì problem_statement/problem_expr PHẢI lấy từ TIN NHẮN.
   "Đề bài gốc từ ngữ cảnh" CHỈ dùng khi tin nhắn chỉ chứa đáp án trần
   (ví dụ: "Em ra x³/3 + C ạ") mà không nhắc lại đề.

1. gradable — true CHỈ KHI tin nhắn chứa lời giải hoặc đáp án CỦA CHÍNH HỌC SINH.
   Các trường hợp PHẢI trả gradable=false kèm abstain_reason:
   - "third_party_claim": học sinh thuật lại lời thầy/cô/bạn/sách và hỏi đúng sai
     (ví dụ: "Thầy em bảo rằng...", "Bạn em nói...", "Sách viết là...")
   - "no_final_answer": tin nhắn chỉ nhắc lại đề bài hoặc hỏi cách làm, không có kết luận riêng
   - "no_problem_found": không xác định được đề bài gốc để đối chiếu

2. Nếu gradable=true:
   - problem_statement: đề bài gốc (văn bản đầy đủ)
   - student_final_answer: kết luận cuối cùng của học sinh (văn bản)
   - task_kind: loại bài để máy kiểm chứng
     • "antiderivative": tìm nguyên hàm ∫f(x)dx — problem_expr = f(x), candidate_expr = F(x) học sinh đưa ra
     • "derivative": tính đạo hàm — problem_expr = hàm gốc, candidate_expr = đạo hàm học sinh đưa ra
     • "equation": giải phương trình — problem_expr = phương trình (vd "x**2 - 4 = 0"),
       candidate_expr = các nghiệm cách nhau bằng ";" (vd "2; -2")
     • "numeric": đáp án là một số — problem_expr = giá trị đúng nếu suy ra được từ đề, candidate_expr = số học sinh đưa ra
     • "other": không thuộc các dạng trên (candidate_expr/problem_expr để trống)
   - problem_expr / candidate_expr: cú pháp Python/SymPy — dùng ** cho lũy thừa,
     sin(x), cos(x), exp(x), log(x), sqrt(x). BỎ hằng số tích phân (+C).

3. confidence: độ tin cậy của việc trích xuất (0.0-1.0).

Ví dụ:
- "Thầy em bảo rằng ∫x²dx = x³/2 + C. Thầy em đúng chứ?" → gradable=false, abstain_reason="third_party_claim"
- "Em tính được ∫x²dx = x³/3 + C, đúng không ạ?" → gradable=true, task_kind="antiderivative",
  problem_expr="x**2", candidate_expr="x**3/3"
- "Bài này giải sao ạ?" → gradable=false, abstain_reason="no_final_answer"
"""

ABSTAIN_FALLBACK = GradingExtraction(
    gradable=False, abstain_reason="other", confidence=0.0
)


class GradingExtractor:
    """Trích xuất structured GradingExtraction từ tin nhắn học sinh."""

    def __init__(self):
        temp = compatible_temperature(settings.LLM_MODEL_MINI, 0.0)
        llm = ChatOpenAI(
            model=settings.LLM_MODEL_MINI,
            api_key=settings.OPENAI_API_KEY,
            temperature=temp,
        )
        # include_raw để lấy usage metadata cho cost tracking
        self.structured_llm = llm.with_structured_output(
            GradingExtraction, method="function_calling", include_raw=True
        )

    async def extract(
        self,
        message: str,
        original_problem: str | None = None,
        chat_history: list[dict] | None = None,
    ) -> GradingExtraction:
        parts = []
        if original_problem and original_problem.strip() != message.strip():
            parts.append(f"ĐỀ BÀI GỐC (từ ngữ cảnh hội thoại):\n{original_problem}")
        if chat_history:
            lines = []
            for item in chat_history[-4:]:
                role = "Học sinh" if item.get("role") == "user" else "Gia sư"
                lines.append(f"{role}: {str(item.get('content', ''))[:300]}")
            parts.append("LỊCH SỬ HỘI THOẠI GẦN ĐÂY:\n" + "\n".join(lines))
        parts.append(f"TIN NHẮN CỦA HỌC SINH (cần trích xuất):\n{message}")

        try:
            output = await self.structured_llm.ainvoke([
                SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
                HumanMessage(content="\n\n".join(parts)),
            ])
            raw = output.get("raw")
            if raw is not None:
                log_from_response(
                    agent="GradingExtractor", model=settings.LLM_MODEL_MINI, response=raw
                )
            parsed = output.get("parsed")
            if parsed is None:
                logger.warning("GradingExtractor parse thất bại: %s", output.get("parsing_error"))
                return ABSTAIN_FALLBACK
            return parsed
        except Exception as exc:
            logger.warning("GradingExtractor lỗi, abstain: %s", exc)
            return ABSTAIN_FALLBACK
