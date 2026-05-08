"""
Question Splitter — Tách raw OCR text thành danh sách câu hỏi riêng lẻ.

Sử dụng LLM (GPT-5.4-mini) để hiểu cấu trúc đề thi và tách chính xác,
bao gồm cả câu hỏi con (a, b, c, d trong Đúng/Sai).

Pipeline:
  Raw OCR text (multi-page) → LLM Splitter → [Question_1, ..., Question_n]

Tại sao LLM thay vì regex?
  - Đề thi VN có format rất đa dạng (Câu 1 / Bài 1 / I. / 1.)
  - Câu Đúng/Sai có 4 mệnh đề con → cần hiểu ngữ cảnh
  - Đề có phần chung dùng cho nhiều câu → cần gộp context
  - LLM xử lý lỗi OCR tốt hơn regex
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from openai import AsyncOpenAI
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SPLITTER_PROMPT = """\
Bạn là công cụ tách câu hỏi từ đề thi Toán 12.

INPUT: Nội dung đề thi đã OCR (có thể có lỗi OCR nhỏ).

NHIỆM VỤ: Tách thành danh sách các câu hỏi riêng lẻ.

QUY TẮC:
1. Mỗi câu hỏi PHẢI là một đơn vị HOÀN CHỈNH, tự chứa đủ thông tin để giải.
2. Giữ NGUYÊN LaTeX formatting ($...$, $$...$$, \\frac, \\sqrt, ...).
3. Giữ nguyên số thứ tự câu (Câu 1, Câu 2, ...).
4. Nếu câu hỏi Đúng/Sai có 4 mệnh đề a/b/c/d → giữ nguyên thành MỘT câu (không tách).
5. Nếu đề có phần chung (dữ kiện dùng cho nhiều câu) → gộp dữ kiện vào MỖI câu liên quan.
6. Bỏ qua phần header (tên trường, mã đề, thời gian, họ tên...).
7. Sửa lỗi OCR rõ ràng (VD: "ham so" → "hàm số") nhưng KHÔNG thay đổi nội dung toán học.
8. Nếu câu hỏi trắc nghiệm có 4 đáp án A/B/C/D → giữ nguyên thành MỘT câu.

OUTPUT: JSON array, mỗi phần tử là một object:
[
  {
    "question_number": "Câu 1",
    "question_type": "mcq",
    "content": "Nội dung đầy đủ của câu hỏi (bao gồm các phương án A/B/C/D nếu có)"
  },
  {
    "question_number": "Câu 2",
    "question_type": "true_false",
    "content": "Nội dung câu hỏi + 4 mệnh đề a/b/c/d"
  }
]

Các giá trị question_type hợp lệ:
- "mcq": Trắc nghiệm chọn 1 đáp án (A/B/C/D)
- "true_false": Đúng/Sai (có 4 mệnh đề a/b/c/d)
- "short_answer": Trả lời ngắn (điền số, biểu thức)
- "essay": Tự luận

CHỈ TRẢ VỀ JSON ARRAY. KHÔNG giải thích, KHÔNG thêm markdown."""


class QuestionSplitter:
    """Split raw OCR text into individual questions using LLM."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        # Use MINI model — this is a structured extraction task, not reasoning
        self.model = settings.LLM_MODEL_MINI

    async def split(self, raw_ocr_text: str) -> list[dict]:
        """Split OCR text into list of question dicts.

        Returns
        -------
        list[dict]
            Each dict has: question_number, question_type, content
        """
        if not raw_ocr_text or len(raw_ocr_text.strip()) < 20:
            logger.warning("OCR text too short to split: %d chars", len(raw_ocr_text))
            return []

        try:
            response = await self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": SPLITTER_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": f"Nội dung đề thi:\n\n{raw_ocr_text}",
                    },
                ],
            )

            content = response.output_text.strip()

            # Cost tracking
            from app.utils.cost_tracker import log_call
            input_tokens = getattr(response.usage, "input_tokens", 0) if response.usage else 0
            output_tokens = getattr(response.usage, "output_tokens", 0) if response.usage else 0
            log_call(
                agent="QuestionSplitter",
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            # Parse JSON — strip markdown fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]

            questions = json.loads(content)

            if not isinstance(questions, list):
                logger.error("Splitter returned non-list: %s", type(questions))
                return []

            # Validate structure
            validated = []
            for q in questions:
                if isinstance(q, dict) and "content" in q:
                    validated.append({
                        "question_number": q.get("question_number", f"Câu {len(validated) + 1}"),
                        "question_type": q.get("question_type", "mcq"),
                        "content": q["content"],
                    })

            logger.info(
                "📋 Question Splitter: %d questions extracted from %d chars OCR text",
                len(validated), len(raw_ocr_text),
            )
            return validated

        except json.JSONDecodeError as e:
            logger.error("Splitter JSON parse error: %s", e)
            return self._fallback_split(raw_ocr_text)
        except Exception as e:
            logger.error("Splitter error: %s", e)
            return self._fallback_split(raw_ocr_text)

    def _fallback_split(self, text: str) -> list[dict]:
        """Regex-based fallback if LLM splitting fails."""
        import re

        # Try common Vietnamese exam patterns
        pattern = r"(?:Câu|Bài)\s+(\d+)[.:\s]"
        splits = re.split(pattern, text, flags=re.IGNORECASE)

        questions = []
        i = 1
        while i < len(splits) - 1:
            num = splits[i]
            content = splits[i + 1].strip()
            if content:
                questions.append({
                    "question_number": f"Câu {num}",
                    "question_type": "mcq",
                    "content": content,
                })
            i += 2

        if not questions and text.strip():
            # Last resort: treat entire text as one question
            questions = [{
                "question_number": "Câu 1",
                "question_type": "essay",
                "content": text.strip(),
            }]

        logger.warning(
            "⚠️ Used fallback regex splitter: %d questions", len(questions)
        )
        return questions
