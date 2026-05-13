"""
Dual OCR Engine — Strategy Pattern.

Provides a unified interface for two OCR backends:
  • CloudVisionOCR  → GPT-4o-mini Vision API (high accuracy, costs money)
  • LocalGOTOCR     → GOT-OCR2.0 local model (free, offline, 580M params)

Usage:
    engine = get_ocr_engine("cloud")   # or "local"
    text = await engine.ocr_image(image_bytes, "image/png")
    pages = await engine.ocr_pdf(pdf_bytes)

Solution Boundary Detection:
    Nhiều đề thi VN đính kèm phần HƯỚNG DẪN GIẢI / ĐÁP ÁN ở cuối PDF.
    ocr_pdf() tự động phát hiện và dừng trước trang đó để:
      - Không OCR phần giải (chatbot chỉ cần câu hỏi)
      - Tiết kiệm API cost
    Dùng PyMuPDF native text scan → hoàn toàn miễn phí, không tốn API.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Solution Section Keywords ──────────────────────────────────────────────────
# Pattern 1: Tiêu đề rõ ràng (đầu trang) — "HƯỚNG DẪN GIẢI", "ĐÁP ÁN", v.v.
_SOLUTION_HEADER = re.compile(
    r"""(?xi)
    \b(h[uư][ớơ]ng\s+d[aẫ]n\s+gi[aả]i)\b   # HƯỚNG DẪN GIẢI
    | \b([\u0111d][aáà]p\s+[aá]n)\b             # ĐÁP ÁN
    | \b(l[oờ][iy]\s+gi[aả]i)\b              # LỜI GIẢI
    | \b(b[aà]i\s+gi[aả]i)\b                 # BÀI GIẢI
    | \b(gi[aả]i\s+chi\s+ti[eế]t)\b         # GIẢI CHI TIẾT
    | \b(h[uư][ớơ]ng\s+d[aẫ]n)\b            # HƯỚNG DẪN
    | \b(solution[s]?)\b                      # SOLUTIONS (English)
    | \b(answer\s+key)\b                      # ANSWER KEY
    | \b(ph[aầ]n\s+gi[aả]i)\b               # PHẦN GIẢI
    """,
    re.IGNORECASE | re.UNICODE,
)

# Pattern 2: Nội dung bước giải (không cần tiêu đề) — xuất hiện ở đầu trang
# Dùng để bắt "a) Đúng.", "Giải:", "Ta có:", v.v. thường gặp trong phần giải không có header
_SOLUTION_BODY = re.compile(
    r"""(?xi)
    ^\s*[a-d][)\.]\s*(đ[uú]ng|đ[uú]ng\.?|sai\.?)\b   # a) Đúng. / b) Sai.
    | ^\s*gi[aả]i\s*:                             # Giải:
    | ^\s*ta\s+có\s*:                              # Ta có:
    | ^\s*ch[uú]ng\s+ta\s+có                      # Chúng ta có
    | ^\s*\+\s+ch[oọ]\s+                           # + Chọn ...
    | ^\s*·\s+ch[oọ]\s+                           # · Chọn
    | ^\s*cách\s+\d+\s*:                           # Cách 1:
    | ^\s*ph\u01b0[oơ]ng\s+pháp\s*:                   # Phương pháp:
    """,
    re.IGNORECASE | re.UNICODE | re.MULTILINE,
)

# Số dòng match tối thiểu để xác nhận là trang giải (tránh false positive)
_SOLUTION_BODY_MIN_MATCHES = 3

# Chỉ kiểm tra N dòng đầu của trang (tiêu đề phần giải thường ở đầu)
_SOLUTION_CHECK_LINES = 8
# Trang có ít hơn N chars → trang trắng / ảnh, bỏ qua kiểm tra
_MIN_CHARS_FOR_SOLUTION_CHECK = 30
# Số dòng cần kiểm tra body (rộng hơn header check)
_BODY_CHECK_LINES = 15


# ── Question Counter approach ──────────────────────────────────────────────────
# Cấu trúc đề thi Toán THPT chuẩn:
#   Phần I  : 12 câu trắc nghiệm 4 lựa chọn (Câu 1–12)
#   Phần II : 4 câu đúng/sai       (Câu 1–4, mỗi câu 4 ý a,b,c,d)
#   Phần III: 6 câu trả lời ngắn   (Câu 1–6)
#   Tổng: 22 câu (nhưng số thứ tự restart trong mỗi phần)

# Số câu hỏi trong đề thi Toán THPT chuẩn (số thứ tự liên tục)
STANDARD_EXAM_QUESTIONS = 22

# Pattern nhận dạng "Câu X" trong text OCR native
_QUESTION_MARKER = re.compile(
    r"(?:Câu|CAU|câu)\s*(\d{1,2})\s*[:\.\)]",
    re.UNICODE,
)

# Phần III là section cuối cùng của đề (6 câu trả lời ngắn)
PHAN_III_LAST_QUESTION = 6

# Pattern nhận dạng "PHẦN I/II/III" header (các đậnh dạng tiêu đề phần)
_SECTION_HEADER = re.compile(
    r"PH[AẦÂ][NẠN]\s+(I{1,3}|IV|VI{0,3}|\d+)[\s\.,:\.]",
    re.IGNORECASE | re.UNICODE,
)


def detect_solution_boundary(doc, max_pages: int) -> int:
    """Phát hiện trang bắt đầu phần GIẢI / ĐÁP ÁN bằng PyMuPDF native text.

    3 signal (theo thứ tự ưu tiên):

    1. **Header keyword**: "HƯỚNG DẪN GIẢI", "ĐÁP ÁN", v.v. → dừng ngay.

    2. **Câu 22** (số liên tục): Đề 1 mã đề, tìm trang cuối chứa Câu 22
       → trang tiếp theo là boundary.

    3. **Section structure** (PHẦN I/II/III + Câu cuối):
       Cấu trúc chuẩn: PHẦN III kết thúc tại Câu 6.
       Trang cuối cùng thấy PHẦN III + Câu 6 → trang tiếp theo là boundary.
       Hoạt động với đề 2 mã: PHẦN III xuất hiện 2 lần → lấy lần cuối.

    Chi phí: 0 API calls — PyMuPDF get_text() miễn phí.

    Returns
    -------
    int
        0-indexed page đầu tiên thuộc phần giải.
        Bằng min(len(doc), max_pages) nếu không tìm thấy.
    """
    n = min(len(doc), max_pages)
    last_q22_page = None       # Signal 2: trang cuối thấy Câu 22
    last_phan3_q6_page = None  # Signal 3: trang cuối thấy PHẦN III + Câu 6
    in_phan3 = False           # Đang trong PHẦN III?

    for page_num in range(n):
        page = doc[page_num]
        raw = page.get_text("text").strip()

        if len(raw) < _MIN_CHARS_FOR_SOLUTION_CHECK:
            continue

        non_empty = [l for l in raw.splitlines() if l.strip()]

        # ── Signal 1: Header keyword rõ ràng ──────────────────────────
        first_lines = "\n".join(non_empty[:_SOLUTION_CHECK_LINES])
        if _SOLUTION_HEADER.search(first_lines):
            logger.info(
                "📌 [Header] Boundary tại trang %d. Preview: %r",
                page_num + 1, first_lines[:80].replace("\n", " "),
            )
            return page_num

        # ── Signal 2 & 3: Phân tích section và số câu ─────────────────
        # Cập nhật trạng thái section từ header trên trang này
        raw_upper = raw.upper()
        sections_on_page = _SECTION_HEADER.findall(raw_upper)
        if sections_on_page:
            last_sec = sections_on_page[-1].strip()
            if "III" in last_sec or last_sec == "3":
                in_phan3 = True
                logger.debug("📄 Page %d: vào PHẦN III", page_num + 1)
            elif last_sec in ("I", "1", "II", "2"):
                # PHẦN I hoặc PHẦN II → chưa/không còn trong PHẦN III
                in_phan3 = False

        q_numbers = [int(m) for m in _QUESTION_MARKER.findall(raw)]
        if not q_numbers:
            continue

        max_q = max(q_numbers)

        # Signal 2: đề 1 mã, số thứ tự liên tục đến 22
        if max_q >= STANDARD_EXAM_QUESTIONS:
            last_q22_page = page_num
            logger.debug("📄 Page %d: thấy Câu 22+", page_num + 1)

        # Signal 3: đang trong PHẦN III và thấy Câu 6 (câu cuối PHẦN III)
        if in_phan3 and PHAN_III_LAST_QUESTION in q_numbers:
            last_phan3_q6_page = page_num
            logger.debug(
                "📄 Page %d: PHẦN III + Câu %d → cập nhật last_phan3_q6_page",
                page_num + 1, PHAN_III_LAST_QUESTION,
            )

    # ── Tổng hợp kết quả ─────────────────────────────────────────────
    # Ưu tiên Signal 3 (section-based, chính xác hơn)
    if last_phan3_q6_page is not None:
        boundary = last_phan3_q6_page + 1
        if boundary < n:
            logger.info(
                "📌 [Section] PHẦN III Câu 6 tại trang %d → boundary trang %d. OCR 1→%d.",
                last_phan3_q6_page + 1, boundary + 1, boundary,
            )
            return boundary

    # Signal 2: Câu 22 (dự phòng)
    if last_q22_page is not None:
        boundary = last_q22_page + 1
        if boundary < n:
            logger.info(
                "📌 [Q-Count] Câu 22 tại trang %d → boundary trang %d. OCR 1→%d.",
                last_q22_page + 1, boundary + 1, boundary,
            )
            return boundary

    # Không tìm thấy signal → OCR toàn bộ (an toàn)
    logger.info(
        "📄 Không tìm thấy boundary trong %d trang → OCR toàn bộ %d trang.", n, n,
    )
    return n


class OCRStrategy(ABC):
    """Base interface for all OCR engines."""

    @abstractmethod
    async def ocr_image(self, image_bytes: bytes, mime_type: str = "image/jpeg", is_exam: bool = False) -> str:
        """OCR a single image, return text with LaTeX formatting."""
        ...

    # Minimum chars expected per page — below this, OCR likely failed
    MIN_PAGE_CHARS = 100
    MAX_OCR_RETRIES = 2

    async def ocr_pdf(self, pdf_bytes: bytes) -> list[str]:
        """Convert PDF → images → OCR chỉ phần câu hỏi.

        Pipeline:
          1. Pre-scan với PyMuPDF native text (miễn phí) để phát hiện
             trang bắt đầu phần GIẢI / ĐÁP ÁN → dừng trước đó.
          2. Áp dụng MAX_EXAM_PAGES limit.
          3. OCR từng trang câu hỏi bằng Vision API.
          4. Retry nếu trang trả về quá ít ký tự.

        Returns
        -------
        list[str]
            Danh sách OCR text, mỗi phần tử là một trang câu hỏi.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return ["[Lỗi: PyMuPDF chưa được cài. Chạy: pip install PyMuPDF]"]

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        max_pages = settings.MAX_EXAM_PAGES

        # ── Step 1: Detect solution boundary (free pre-scan) ─────────────────
        solution_start = detect_solution_boundary(doc, max_pages)

        # Số trang cần OCR = trước phần giải VÀ không vượt MAX_EXAM_PAGES
        n_pages = min(solution_start, max_pages, total_pages)

        if solution_start < total_pages:
            logger.info(
                "📄 PDF: %d trang tổng | OCR trang 1–%d (câu hỏi) | "
                "Bỏ qua trang %d–%d (phần giải/đáp án)",
                total_pages, n_pages, solution_start + 1, total_pages,
            )
        elif total_pages > max_pages:
            logger.warning(
                "PDF có %d trang, giới hạn OCR %d trang (MAX_EXAM_PAGES). "
                "Không phát hiện phần giải.",
                total_pages, n_pages,
            )
        else:
            logger.info(
                "📄 PDF: %d trang, không có phần giải → OCR toàn bộ %d trang",
                total_pages, n_pages,
            )

        # ── Step 2: OCR each question page ───────────────────────────────────
        page_texts: list[str] = []
        for page_num in range(n_pages):
            page = doc[page_num]
            # 300 DPI for high-quality OCR
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")

            text = await self.ocr_image(img_bytes, mime_type="image/png", is_exam=True)

            if "<END_OF_EXAM>" in text:
                logger.info("🛑 GPT Vision phát hiện phần Đáp án/Giải chi tiết tại trang %d. Dừng OCR.", page_num + 1)
                break

            # Retry if response is suspiciously short
            attempt = 1
            while len(text.strip()) < self.MIN_PAGE_CHARS and attempt <= self.MAX_OCR_RETRIES:
                import asyncio
                wait = attempt  # 1s, 2s backoff
                logger.warning(
                    "⚠️ OCR page %d returned only %d chars (min=%d), "
                    "retrying (%d/%d) after %ds...",
                    page_num + 1, len(text.strip()), self.MIN_PAGE_CHARS,
                    attempt, self.MAX_OCR_RETRIES, wait,
                )
                await asyncio.sleep(wait)
                text = await self.ocr_image(img_bytes, mime_type="image/png", is_exam=True)
                attempt += 1

            if len(text.strip()) < self.MIN_PAGE_CHARS:
                logger.error(
                    "❌ OCR page %d still only %d chars after %d retries",
                    page_num + 1, len(text.strip()), self.MAX_OCR_RETRIES,
                )

            page_texts.append(text)
            logger.debug("OCR page %d/%d done (%d chars)", page_num + 1, n_pages, len(text))

        doc.close()
        return page_texts


class CloudVisionOCR(OCRStrategy):
    """Option A: GPT-4o-mini Vision (Cloud API).

    Trade-offs:
      ✅ Best accuracy for Vietnamese math + LaTeX
      ✅ Understands graphs and figures
      ❌ ~$0.01–0.03/page
      ❌ Requires internet
    """

    async def ocr_image(self, image_bytes: bytes, mime_type: str = "image/jpeg", is_exam: bool = False) -> str:
        from app.ocr.processor import get_ocr_processor
        return await get_ocr_processor().process_image(image_bytes, mime_type, is_exam=is_exam)


class LocalGOTOCR(OCRStrategy):
    """Option B: GOT-OCR2.0 (Local, offline, free).

    Trade-offs:
      ✅ Completely free
      ✅ Offline / privacy-preserving
      ❌ Needs GPU for speed (CPU works but ~5x slower)
      ❌ Vietnamese support is limited
    """

    async def ocr_image(self, image_bytes: bytes, mime_type: str = "image/jpeg", is_exam: bool = False) -> str:
        from app.ocr.got_ocr import get_got_ocr
        return await get_got_ocr().process_image(image_bytes)


# ── Factory ───────────────────────────────────────────────────────────────────

def get_ocr_engine(engine: str = "cloud") -> OCRStrategy:
    """Factory method — select OCR engine by name.

    Parameters
    ----------
    engine : str
        "cloud" → GPT-4o-mini Vision (default)
        "local" → GOT-OCR2.0
    """
    if engine == "local":
        return LocalGOTOCR()
    return CloudVisionOCR()
