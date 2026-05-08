"""
Dual OCR Engine — Strategy Pattern.

Provides a unified interface for two OCR backends:
  • CloudVisionOCR  → GPT-4o-mini Vision API (high accuracy, costs money)
  • LocalGOTOCR     → GOT-OCR2.0 local model (free, offline, 580M params)

Usage:
    engine = get_ocr_engine("cloud")   # or "local"
    text = await engine.ocr_image(image_bytes, "image/png")
    pages = await engine.ocr_pdf(pdf_bytes)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class OCRStrategy(ABC):
    """Base interface for all OCR engines."""

    @abstractmethod
    async def ocr_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        """OCR a single image, return text with LaTeX formatting."""
        ...

    async def ocr_pdf(self, pdf_bytes: bytes) -> list[str]:
        """Convert PDF → images → OCR each page.

        Returns list of OCR text, one per page.
        Enforces MAX_EXAM_PAGES limit.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return ["[Lỗi: PyMuPDF chưa được cài. Chạy: pip install PyMuPDF]"]

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        max_pages = settings.MAX_EXAM_PAGES
        n_pages = min(len(doc), max_pages)

        if len(doc) > max_pages:
            logger.warning(
                "PDF has %d pages, limiting to %d (MAX_EXAM_PAGES)",
                len(doc), max_pages,
            )

        page_texts: list[str] = []
        for page_num in range(n_pages):
            page = doc[page_num]
            # 300 DPI for high-quality OCR
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")

            text = await self.ocr_image(img_bytes, mime_type="image/png")
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

    async def ocr_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        from app.ocr.processor import get_ocr_processor
        return await get_ocr_processor().process_image(image_bytes, mime_type)


class LocalGOTOCR(OCRStrategy):
    """Option B: GOT-OCR2.0 (Local, offline, free).

    Trade-offs:
      ✅ Completely free
      ✅ Offline / privacy-preserving
      ❌ Needs GPU for speed (CPU works but ~5x slower)
      ❌ Vietnamese support is limited
    """

    async def ocr_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
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
