"""Select focused figures for a question before sending them to the VLM."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class VisualInput:
    image_bytes: bytes
    mime_type: str = "image/png"
    label: str = "figure"
    page_num: int = 1


_LABEL_PRIORITY = {
    "graph": 0,
    "table": 1,
    "figure": 2,
    "diagram": 3,
    "photo": 4,
}


def _positive_page_numbers(pages: Iterable[int] | None) -> list[int]:
    """Normalize untrusted splitter metadata without aborting visual solving."""
    normalized: list[int] = []
    for page in pages or []:
        try:
            page_num = int(page)
        except (TypeError, ValueError):
            continue
        if page_num > 0 and page_num not in normalized:
            normalized.append(page_num)
    return normalized


def select_question_visuals(
    figure_pages: Iterable[int] | None,
    page_images: dict[int, bytes] | None,
    extracted_images: Iterable[object] | None,
    max_images: int = 4,
) -> list[VisualInput]:
    """Prefer cropped graph/table images from relevant pages, with safe fallback.

    ``ExtractedImage.page_num`` is zero-based while ``figure_pages`` and
    ``page_images`` are one-based.
    """
    if max_images <= 0:
        return []

    page_images = page_images or {}
    requested_pages = _positive_page_numbers(figure_pages)
    if not requested_pages:
        requested_pages = sorted(page_images)
    requested_set = set(requested_pages)

    crops: list[tuple[int, int, int, VisualInput]] = []
    for image in extracted_images or []:
        raw = getattr(image, "image_bytes", b"")
        try:
            page_num = int(getattr(image, "page_num", -1)) + 1
        except (TypeError, ValueError):
            continue
        if not raw or page_num not in requested_set:
            continue
        label = str(getattr(image, "label", "figure") or "figure").lower()
        area = int(getattr(image, "width", 0)) * int(getattr(image, "height", 0))
        visual = VisualInput(
            image_bytes=raw,
            mime_type=str(getattr(image, "mime_type", "image/png") or "image/png"),
            label=label,
            page_num=page_num,
        )
        crops.append((_LABEL_PRIORITY.get(label, 5), page_num, -area, visual))

    crops.sort(key=lambda item: item[:3])
    selected = [item[3] for item in crops]

    # If extraction found no focused visual, retain the old whole-page path.
    if not selected:
        for page_num in requested_pages:
            raw = page_images.get(page_num)
            if raw:
                selected.append(VisualInput(raw, "image/png", "page", page_num))
                break

    deduped: list[VisualInput] = []
    seen_hashes: set[str] = set()
    for visual in selected:
        digest = hashlib.sha1(visual.image_bytes).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        deduped.append(visual)
        if len(deduped) >= max_images:
            break
    return deduped
