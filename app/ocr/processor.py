"""OCR processor — uses OpenAI Vision (GPT-4o-mini) to extract math LaTeX from images.
No Tesseract or external binary required.
"""

import base64
from typing import Optional

from openai import AsyncOpenAI
from app.config import get_settings

settings = get_settings()

# Prompt: yêu cầu output LaTeX chuẩn để downstream LLM xử lý tốt hơn
VISION_PROMPT = """Bạn là công cụ nhận dạng bài toán toán học từ ảnh.
Hãy đọc và trích xuất TOÀN BỘ nội dung trong ảnh (đề bài, biểu thức, hình vẽ, chú thích, ...).

QUY TẮC ĐỊNH DẠNG:
- Công thức toán học → viết dưới dạng LaTeX đúng chuẩn:
  + Phân số: \\frac{tử}{mẫu}
  + Căn bậc hai: \\sqrt{x}
  + Tích phân: \\int_a^b f(x)\\,dx
  + Lũy thừa: x^{2}, x^{n}
  + Giới hạn: \\lim_{x \\to a}
  + Logarithm: \\log, \\ln
  + Inline math: $biểu_thức$   Block math: $$biểu_thức$$
- Giữ nguyên cấu trúc bài (câu hỏi / dữ kiện / yêu cầu).
- Nếu ảnh có nhiều câu/phần, giữ nguyên phân cách.
- Nếu ảnh không chứa toán học, mô tả ngắn gọn những gì thấy.

CHỈ trả về nội dung đã trích xuất (text + LaTeX). KHÔNG giải thích. KHÔNG thêm lời dẫn."""


class OCRProcessor:
    """Process uploaded images to extract math LaTeX via OpenAI Vision."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        # MINI: vision OCR for math images — balanced speed/quality
        self.vision_model = settings.LLM_MODEL_MINI

    async def process_image(self, image_bytes: bytes, mime_type: str = "image/jpeg", is_exam: bool = False) -> str:
        """
        Extract LaTeX-formatted math text from an image using OpenAI Vision.
        Returns the extracted LaTeX string.
        """
        try:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            
            prompt = VISION_PROMPT
            if is_exam:
                prompt += "\n\n*** CHÚ Ý QUAN TRỌNG ***\nNẾU BẠN THẤY PHẦN ĐÁP ÁN, HƯỚNG DẪN GIẢI, LỜI GIẢI CHI TIẾT, HOẶC BẢNG ĐÁP ÁN (BẢNG KẾT QUẢ ĐÁP ÁN CÁC MÃ ĐỀ): BẠN CHỈ CẦN TRẢ VỀ DUY NHẤT CHUỖI `<END_OF_EXAM>` VÀ TUYỆT ĐỐI KHÔNG TRÍCH XUẤT GÌ THÊM. Điều này giúp hệ thống biết đề thi đã kết thúc."

            response = await self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{b64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_completion_tokens=1500,
                temperature=0.0,
            )

            extracted = response.choices[0].message.content.strip()
            return extracted if extracted else "[Không thể đọc được nội dung từ ảnh. Vui lòng chụp rõ hơn.]"

        except Exception as e:
            return f"[Lỗi nhận dạng ảnh: {str(e)}]"

    def build_image_data_url(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        """Build a base64 data URL string for use in Vision API calls."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    def image_to_base64(self, image_bytes: bytes) -> str:
        """Legacy: Convert image bytes to plain base64 (no data URL prefix)."""
        return base64.b64encode(image_bytes).decode("utf-8")


# Singleton
_processor: Optional[OCRProcessor] = None


def get_ocr_processor() -> OCRProcessor:
    global _processor
    if _processor is None:
        _processor = OCRProcessor()
    return _processor
