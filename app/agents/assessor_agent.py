"""Assessor Agent — grades student answers and identifies errors."""

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from app.config import get_settings
from app.utils.cost_tracker import log_from_response
import json

settings = get_settings()


ASSESSOR_SYSTEM_PROMPT = """Bạn là giám khảo chấm bài Toán 12.
Nhiệm vụ: Đánh giá câu trả lời của học sinh.

QUY TẮC:
1. So sánh câu trả lời của học sinh với lời giải đúng.
2. Xác định loại lỗi (nếu có):
   - "calculation": Lỗi tính toán
   - "conceptual": Hiểu sai khái niệm / công thức
   - "procedural": Sai phương pháp giải
   - "none": Đáp án đúng
3. Đưa ra nhận xét cụ thể về lỗi.
4. Cung cấp lời giải đúng.

BẮT BUỘC trả về JSON theo format:
{{
    "is_correct": true/false,
    "score": 0.0-1.0,
    "error_type": "none" | "calculation" | "conceptual" | "procedural",
    "feedback": "Nhận xét chi tiết bằng tiếng Việt",
    "correct_solution": "Lời giải đúng đầy đủ"
}}

CHỈ TRẢ VỀ JSON, KHÔNG CÓ TEXT KHÁC.
"""


class AssessorAgent:
    """Agent that evaluates student answers."""

    def __init__(self):
        # MINI: structured JSON output, no deep math reasoning needed
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL_MINI,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.1,
        )

    async def assess(self, question: str, student_answer: str) -> dict:
        """
        Evaluate a student's answer.

        Returns: dict with is_correct, score, error_type, feedback, correct_solution
        """
        prompt = f"""CÂU HỎI:
{question}

CÂU TRẢ LỜI CỦA HỌC SINH:
{student_answer}

Hãy chấm điểm và phân tích."""

        messages = [
            SystemMessage(content=ASSESSOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)

        # ── cost log ──────────────────────────────────────────────────────────
        log_from_response(agent="Assessor", model=settings.LLM_MODEL_MINI, response=response)

        try:
            # Parse JSON response
            content = response.content.strip()
            # Remove markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]
            result = json.loads(content)
        except (json.JSONDecodeError, IndexError):
            result = {
                "is_correct": False,
                "score": 0.0,
                "error_type": "unknown",
                "feedback": response.content,
                "correct_solution": "Không thể phân tích lời giải.",
            }

        return result
