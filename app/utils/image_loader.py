"""
Utility để load ảnh câu hỏi từ metadata và encode thành base64 cho Vision API.

Flow:
  1. RAG retrieve → trả về Document có metadata["image_path"]
  2. build_message_content() đọc ảnh → tạo content list cho GPT-4o Vision
  3. TeacherAgent attach vào HumanMessage để LLM "nhìn" được hình
"""

import base64
from pathlib import Path
from typing import Optional

from langchain.schema import Document


# ── Image Loading ─────────────────────────────────────────

def load_image_base64(image_path: str) -> Optional[str]:
    """
    Đọc ảnh từ đường dẫn, trả về chuỗi base64.
    Trả về None nếu file không tồn tại hoặc lỗi.
    """
    # Hỗ trợ cả đường dẫn tuyệt đối lẫn tương đối so với project root
    path = Path(image_path)
    if not path.is_absolute():
        # Resolve relative to project root (thư mục chứa data/)
        project_root = Path(__file__).parent.parent.parent
        path = project_root / image_path

    if not path.exists():
        return None

    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def get_image_mime(image_path: str) -> str:
    """Xác định MIME type từ extension file."""
    ext = Path(image_path).suffix.lower()
    return {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif":  "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/png")


# ── Message Content Builder ───────────────────────────────

def build_multimodal_content(
    text: str,
    retrieved_docs: list[Document],
    max_images: int = 5,
) -> list[dict]:
    """
    Tạo content list cho GPT-4o Vision từ text + ảnh (nếu có trong docs).

    Hỗ trợ multi-image: image_path có thể chứa nhiều đường dẫn phân cách
    bằng dấu phẩy, ví dụ:
        "data/exams/images/de7plus_de01/q14_1.png,data/.../q14_2.png"

    Returns:
        List dạng OpenAI multimodal content:
        [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            ...
        ]
    """
    content: list[dict] = [{"type": "text", "text": text}]

    images_added: int = 0
    for doc in retrieved_docs:
        if images_added >= max_images:
            break

        meta       = doc.metadata or {}
        has_image  = bool(meta.get("has_image", False))
        image_path = str(meta.get("image_path", ""))

        if not has_image or not image_path:
            continue

        # Hỗ trợ multi-image: tách bằng dấu phẩy
        paths = [p.strip() for p in image_path.split(",") if p.strip()]

        for path in paths:
            if images_added >= max_images:
                break

            b64 = load_image_base64(path)
            if b64 is None:
                continue

            # Label trước mỗi ảnh để LLM map đúng [Xem hình: filename]
            filename = Path(path).name
            content.append({"type": "text", "text": f"[Hình: {filename}]"})

            mime = get_image_mime(path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url":    f"data:{mime};base64,{b64}",
                    "detail": "high",  # "high" để LLM nhìn rõ đồ thị toán học
                },
            })
            images_added += 1

    return content


def has_images(docs: list[Document]) -> bool:
    """Kiểm tra nhanh xem batch docs có chứa ảnh không."""
    return any(d.metadata.get("has_image", False) for d in docs)


# ── Image path helper ─────────────────────────────────────

def make_image_path(exam_id: str, question_number: int, ext: str = "png") -> str:
    """
    Tạo đường dẫn ảnh chuẩn cho một câu hỏi.

    Ví dụ:
        make_image_path("de7plus_de01", 1) 
        → "data/exams/images/de7plus_de01/q01.png"
    """
    return f"data/exams/images/{exam_id}/q{question_number:02d}.{ext}"
