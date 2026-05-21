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
9. Nếu một câu hỏi nằm trên 2 trang (bị cắt giữa chừng), hãy NỐI lại thành 1 câu hoàn chỉnh.

PHÁT HIỆN HÌNH VẼ:
- Nếu câu hỏi THAM CHIẾU đến hình vẽ, đồ thị, bảng biến thiên, hình minh họa
  (ví dụ: "như hình vẽ", "đồ thị bên", "bảng biến thiên sau", "hình chóp S.ABCD")
  → đặt "has_figure": true và mô tả ngắn hình đó trong "figure_description".
- Nếu câu hỏi THUẦN TEXT/CÔNG THỨC (không cần hình) → "has_figure": false.
- XÁC ĐỊNH trang nào chứa hình vẽ đó (dựa vào [Trang X] markers trong OCR text)
  → ghi vào "figure_pages" (array số trang, 1-indexed).

OUTPUT: JSON array, mỗi phần tử là một object:
[
  {
    "question_number": "Câu 1",
    "question_type": "mcq",
    "has_figure": false,
    "figure_description": null,
    "figure_pages": [],
    "content": "Nội dung câu hỏi"
  },
  {
    "question_number": "Câu 9",
    "question_type": "mcq",
    "has_figure": true,
    "figure_description": "Đồ thị hàm số bậc 4 đi qua gốc tọa độ, có 1 cực đại và 1 cực tiểu",
    "figure_pages": [2],
    "content": "Hàm số nào dưới đây có đồ thị là đường cong như hình vẽ..."
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

    # Max chars sent to LLM in one call — stay well under context limit
    # gpt-5.4-mini has ~128k token context but long input causes truncation in output
    MAX_CHARS_PER_CHUNK = 12_000

    async def split(self, raw_ocr_text: str) -> list[dict]:
        """Split OCR text into list of question dicts.

        For long exams (> MAX_CHARS_PER_CHUNK), auto-splits by exam section
        (Phần 1/2/3) or by page boundaries, processes each chunk independently,
        then merges results in order.

        Returns
        -------
        list[dict]
            Each dict has: question_number, question_type, content
        """
        if not raw_ocr_text or len(raw_ocr_text.strip()) < 20:
            logger.warning("OCR text too short to split: %d chars", len(raw_ocr_text))
            return []

        # Chunk nếu text quá dài
        if len(raw_ocr_text) > self.MAX_CHARS_PER_CHUNK:
            chunks = self._split_into_chunks(raw_ocr_text)
            logger.info(
                "📄 OCR text dài (%d chars) → chia %d chunks để split",
                len(raw_ocr_text), len(chunks),
            )
            # Process chunks song song
            import asyncio
            chunk_results = await asyncio.gather(
                *[self._split_chunk(chunk) for chunk in chunks]
            )
            # Merge, bỏ duplicates theo question_type + question_number
            merged = []
            seen = set()
            for chunk_qs in chunk_results:
                for q in chunk_qs:
                    key = (q.get("question_type", ""), q.get("question_number", ""))
                    if key not in seen:
                        seen.add(key)
                        merged.append(q)
            logger.info("📋 Merged: %d unique questions from %d chunks", len(merged), len(chunks))
            return merged

        return await self._split_chunk(raw_ocr_text)

    def _split_into_chunks(self, text: str) -> list[str]:
        """Chia text theo section markers (Phần 1/2/3) hoặc page boundaries.

        Ưu tiên:
          1. Tách theo "Phần 1", "Phần 2", "Phần 3" (cấu trúc đề THPT QG)
          2. Nếu không có → tách theo [Trang X] markers
          3. Nếu vẫn không → tách theo MAX_CHARS_PER_CHUNK cứng
        """
        import re

        # Thử tách theo section markers
        section_pattern = re.compile(
            r"(?=(?:PHẦN|Phần|phần)\s+(?:[IVX]+|\d+)[\s\.\:\—\-])",
            re.IGNORECASE,
        )
        sections = section_pattern.split(text)
        sections = [s.strip() for s in sections if s.strip()]

        if len(sections) >= 2:
            # Merge sections nhỏ vào nhau nếu cần
            chunks = []
            current = ""
            for sec in sections:
                if len(current) + len(sec) <= self.MAX_CHARS_PER_CHUNK:
                    current += "\n\n" + sec if current else sec
                else:
                    if current:
                        chunks.append(current)
                    current = sec
            if current:
                chunks.append(current)
            if chunks:
                return chunks

        # Thử tách theo [Trang X] markers
        page_pattern = re.compile(r"(?=\[Trang \d+\])")
        pages = page_pattern.split(text)
        pages = [p.strip() for p in pages if p.strip()]

        if len(pages) >= 2:
            chunks = []
            current = ""
            for page in pages:
                if len(current) + len(page) <= self.MAX_CHARS_PER_CHUNK:
                    current += "\n\n" + page if current else page
                else:
                    if current:
                        chunks.append(current)
                    current = page
            if current:
                chunks.append(current)
            if chunks:
                return chunks

        # Fallback: tách cứng theo MAX_CHARS_PER_CHUNK (tìm điểm ngắt tại dòng mới)
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.MAX_CHARS_PER_CHUNK
            if end < len(text):
                # Tìm điểm ngắt gần nhất (dòng mới)
                newline_pos = text.rfind("\n", start, end)
                if newline_pos > start:
                    end = newline_pos
            chunks.append(text[start:end].strip())
            start = end
        return [c for c in chunks if c]

    async def _split_chunk(self, chunk_text: str) -> list[dict]:
        """Gọi LLM để split một chunk text. Internal helper."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SPLITTER_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": f"Nội dung đề thi:\n\n{chunk_text}",
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "exam_questions",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "questions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "question_number": {"type": "string"},
                                            "question_type": {
                                                "type": "string",
                                                "enum": ["mcq", "true_false", "short_answer", "essay"],
                                            },
                                            "has_figure": {"type": "boolean"},
                                            "figure_description": {"type": ["string", "null"]},
                                            "figure_pages": {
                                                "type": "array",
                                                "items": {"type": "integer"},
                                            },
                                            "content": {"type": "string"},
                                        },
                                        "required": [
                                            "question_number", "question_type",
                                            "has_figure", "figure_description",
                                            "figure_pages", "content",
                                        ],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["questions"],
                            "additionalProperties": False,
                        },
                    },
                },
                temperature=0.0,
            )

            content = response.choices[0].message.content.strip()

            # Cost tracking
            from app.utils.cost_tracker import log_call
            usage = response.usage
            input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            log_call(
                agent="QuestionSplitter",
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            parsed = json.loads(content)
            questions = parsed.get("questions", [])

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
                        "has_figure": q.get("has_figure", False),
                        "figure_description": q.get("figure_description"),
                        "figure_pages": q.get("figure_pages", []),
                        "content": q["content"],
                    })

            return validated

        except json.JSONDecodeError as e:
            logger.error("Splitter JSON parse error: %s", e)
            return self._fallback_split(chunk_text)
        except Exception as e:
            logger.error("Splitter error: %s", e)
            return self._fallback_split(chunk_text)


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
