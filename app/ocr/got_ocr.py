"""
GOT-OCR2.0 Wrapper — Local OCR engine for math documents.

Model: stepfun-ai/GOT-OCR2_0 (~580M params)
Capabilities: Plain text, math formulas (LaTeX), tables, charts
Requirements: PyTorch + transformers (GPU optional, CPU fallback)

Trade-offs vs Cloud VLM:
  ✅ Free, offline, privacy-preserving
  ❌ Vietnamese support limited (primarily EN/CN)
  ❌ Cannot understand graph semantics (only extracts text)
  ❌ Slower on CPU (~10-15s/page vs ~2-5s cloud)
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_NAME = "stepfun-ai/GOT-OCR2_0"


class GOTOCR:
    """GOT-OCR2.0 local inference wrapper with lazy model loading."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._load_attempted = False
        self._device = "cpu"

    def _load_model(self) -> None:
        """Lazy-load the GOT-OCR2.0 model on first use."""
        if self._load_attempted:
            if not self._loaded:
                raise RuntimeError(
                    "GOT-OCR2.0 failed to load previously. "
                    "Check logs for details or switch to Cloud OCR."
                )
            return

        self._load_attempted = True

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            logger.info("📦 Loading GOT-OCR2.0: %s", MODEL_NAME)

            self._tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, trust_remote_code=True
            )

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self._device == "cuda" else torch.float32

            if self._device == "cpu":
                logger.warning(
                    "⚠️  GOT-OCR2.0 running on CPU — inference will be slow (~10-15s/page). "
                    "Use GPU for faster processing or switch to Cloud OCR."
                )

            self._model = AutoModel.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(self._device).eval()

            self._loaded = True
            logger.info("✅ GOT-OCR2.0 loaded on %s", self._device)

        except ImportError as e:
            logger.error(
                "❌ GOT-OCR2.0 requires: pip install torch transformers. Error: %s", e
            )
            raise
        except Exception as e:
            logger.error("❌ GOT-OCR2.0 load failed: %s", e)
            raise

    async def process_image(self, image_bytes: bytes) -> str:
        """OCR an image using GOT-OCR2.0 locally (GPU or CPU).

        Uses `format` OCR mode to output LaTeX for math formulas.
        Runs inference in a thread to avoid blocking the event loop.
        """
        self._load_model()

        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # GOT-OCR requires a file path for its chat() interface
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            tmp_path = f.name

        try:
            # Run in thread — critical for CPU mode (avoids blocking async loop)
            result = await asyncio.to_thread(
                self._model.chat,
                self._tokenizer,
                tmp_path,
                ocr_type="format",  # format mode → outputs LaTeX for formulas
            )
            return result.strip() if result else "[OCR Local: không đọc được nội dung từ ảnh]"
        except Exception as e:
            logger.error("GOT-OCR2.0 inference error: %s", e)
            return f"[Lỗi OCR Local: {str(e)}]"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @property
    def is_available(self) -> bool:
        """Check if GOT-OCR2.0 can be used (torch must be installed)."""
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def device_info(self) -> str:
        """Return human-readable device info."""
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                return f"GPU ({name})"
            return "CPU (slow mode)"
        except ImportError:
            return "unavailable (torch not installed)"


# ── Singleton ─────────────────────────────────────────────────────────────────
_got: Optional[GOTOCR] = None


def get_got_ocr() -> GOTOCR:
    global _got
    if _got is None:
        _got = GOTOCR()
    return _got
