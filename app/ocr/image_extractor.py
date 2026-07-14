"""
Image/Graph Extractor — Tách ảnh, đồ thị, hình minh họa từ PDF đề thi.

Pipeline:
  PDF → PyMuPDF → Detect embedded images & vector graphics
  → Filter noise (logos, watermarks nhỏ) → Export từng ảnh riêng

Tại sao cần tách riêng?
  - OCR text-only bỏ qua đồ thị → LLM không có context hình học
  - Ảnh minh họa (đồ thị hàm số, hình không gian) PHẢI được gửi
    kèm câu hỏi để LLM giải chính xác
  - Tách riêng cho phép gán ảnh vào đúng câu hỏi (spatial matching)

Outputs:
  - List[ExtractedImage] — mỗi ảnh có: bytes, page, bbox, loại (graph/figure/table)
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Minimum image dimensions to filter out noise (icons, bullets, watermarks)
MIN_WIDTH = 60   # pixels
MIN_HEIGHT = 60  # pixels
MIN_AREA = 5000  # width * height


@dataclass
class ExtractedImage:
    """A single image extracted from a PDF page."""
    page_num: int           # 0-indexed page number
    index: int              # image index within the page
    width: int
    height: int
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) in page coords
    image_bytes: bytes      # PNG bytes
    mime_type: str = "image/png"
    label: str = ""         # auto-classified: "graph", "figure", "table", "photo"
    y_position: float = 0.0  # normalized Y position (0.0=top, 1.0=bottom)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)


def classify_image(img: ExtractedImage, page_width: float, page_height: float) -> str:
    """Auto-classify image type based on dimensions and position.

    Heuristics:
      - Wide + short → table or chart
      - Square-ish → graph or figure
      - Very wide (>80% page width) → likely a full-width diagram
      - Small → icon/noise (should be filtered before this)
    """
    w_ratio = (img.bbox[2] - img.bbox[0]) / max(page_width, 1)
    h_ratio = (img.bbox[3] - img.bbox[1]) / max(page_height, 1)

    if w_ratio > 0.7 and h_ratio > 0.3:
        return "diagram"  # Full-width large image
    elif img.aspect_ratio > 2.5:
        return "table"     # Very wide → likely table
    elif 0.6 < img.aspect_ratio < 1.8:
        return "graph"     # Square-ish → graph/figure
    else:
        return "figure"


def extract_images_from_pdf(
    pdf_bytes: bytes,
    min_width: int = MIN_WIDTH,
    min_height: int = MIN_HEIGHT,
    min_area: int = MIN_AREA,
    max_pages: int = 10,
) -> list[ExtractedImage]:
    """Extract all meaningful images from a PDF.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw PDF file content.
    min_width, min_height, min_area : int
        Minimum dimensions to filter noise images.
    max_pages : int
        Maximum pages to process.

    Returns
    -------
    list[ExtractedImage]
        All extracted images, sorted by (page_num, y_position).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF not installed. Run: pip install PyMuPDF")
        return []

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = min(len(doc), max_pages)
    extracted: list[ExtractedImage] = []

    for page_num in range(n_pages):
        page = doc[page_num]
        page_width = page.rect.width
        page_height = page.rect.height

        # Method 1: Extract embedded raster images
        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                img_bytes = base_image["image"]
                w = base_image.get("width", 0)
                h = base_image.get("height", 0)

                # Filter noise
                if w < min_width or h < min_height or w * h < min_area:
                    continue

                # Find bbox on page
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    rect = img_rects[0]
                    bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                    y_pos = rect.y0 / max(page_height, 1)
                else:
                    bbox = (0, 0, w, h)
                    y_pos = 0.5

                # Convert to PNG if needed
                ext = base_image.get("ext", "png")
                if ext != "png":
                    from PIL import Image as PILImage
                    pil_img = PILImage.open(io.BytesIO(img_bytes))
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    img_bytes = buf.getvalue()

                ei = ExtractedImage(
                    page_num=page_num,
                    index=img_idx,
                    width=w,
                    height=h,
                    bbox=bbox,
                    image_bytes=img_bytes,
                    y_position=y_pos,
                )
                ei.label = classify_image(ei, page_width, page_height)
                extracted.append(ei)

            except Exception as e:
                logger.warning("Failed to extract image xref=%d page=%d: %s", xref, page_num, e)
                continue

        # Method 2: Detect vector drawings (graphs drawn with lines/curves)
        # PyMuPDF can render page regions as images for vector content
        drawings = page.get_drawings()
        if drawings and len(drawings) >= 3:
            # Group nearby drawings into clusters (likely one graph)
            clusters = _cluster_drawings(drawings, page_width, page_height)
            for cluster_idx, cluster_rect in enumerate(clusters):
                cw = cluster_rect.width
                ch = cluster_rect.height
                if cw < min_width or ch < min_height or cw * ch < min_area:
                    continue

                # Render this region as a high-DPI image
                clip = cluster_rect
                mat = fitz.Matrix(3, 3)  # 3x scale for clarity
                pix = page.get_pixmap(matrix=mat, clip=clip)
                png_bytes = pix.tobytes("png")

                ei = ExtractedImage(
                    page_num=page_num,
                    index=len(image_list) + cluster_idx,
                    width=int(cw * 3),
                    height=int(ch * 3),
                    bbox=(clip.x0, clip.y0, clip.x1, clip.y1),
                    image_bytes=png_bytes,
                    y_position=clip.y0 / max(page_height, 1),
                    label="graph",  # Vector drawings are typically graphs
                )
                extracted.append(ei)

    doc.close()

    # Sort by page then Y position (top to bottom)
    extracted.sort(key=lambda x: (x.page_num, x.y_position))

    logger.info(
        "📸 Extracted %d images from %d pages (filtered noise < %dx%d)",
        len(extracted), n_pages, min_width, min_height,
    )
    return extracted


def _cluster_drawings(
    drawings: list,
    page_width: float,
    page_height: float,
    merge_dist: float = 20.0,
) -> list:
    """Cluster nearby vector drawings into bounding regions.

    Multiple drawing primitives (lines, curves) that form a single graph
    are merged into one bounding box.
    """
    import fitz

    if not drawings:
        return []

    # Get bounding rect for each drawing
    rects = []
    for d in drawings:
        r = d.get("rect")
        if r:
            rects.append(fitz.Rect(r))

    if not rects:
        return []

    # Greedy merge overlapping/nearby rects
    merged = [rects[0]]
    for r in rects[1:]:
        found_merge = False
        for i, m in enumerate(merged):
            # Check if rects overlap or are close enough
            expanded = fitz.Rect(
                m.x0 - merge_dist, m.y0 - merge_dist,
                m.x1 + merge_dist, m.y1 + merge_dist,
            )
            if expanded.intersects(r):
                merged[i] = m | r  # Union
                found_merge = True
                break
        if not found_merge:
            merged.append(r)

    # Filter out tiny clusters and full-page borders
    result = []
    for m in merged:
        if m.width < 40 or m.height < 40:
            continue
        # Skip if it's basically the whole page (border/frame)
        if m.width > page_width * 0.9 and m.height > page_height * 0.9:
            continue
        result.append(m)

    return result
