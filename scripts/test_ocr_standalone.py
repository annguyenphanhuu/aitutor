"""
Standalone OCR + Image Extraction Test
========================================
Test 2 option OCR (Cloud + PyMuPDF text) và Image/Graph extraction
KHÔNG cần chạy server FastAPI hay load app.config.
"""

import asyncio, io, json, os, sys, time, base64
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PDF_PATH = ROOT / "data" / "exams" / "41158149ae5a40ec9e7635e4ea4eb815.pdf"
OUTPUT_DIR = ROOT / "data" / "exams" / "images" / PDF_PATH.stem


# ── 1. Image/Graph Extraction ───────────────────────────────────────────────

def test_image_extraction(pdf_bytes: bytes):
    """Extract images and graphs from PDF using PyMuPDF."""
    from app.ocr.image_extractor import extract_images_from_pdf, extract_page_contents

    print(f"\n{'='*70}")
    print(f"🖼️  TEST 1: Image/Graph Extraction")
    print(f"{'='*70}")

    t0 = time.time()
    images = extract_images_from_pdf(pdf_bytes)
    elapsed = time.time() - t0

    print(f"\n  ✅ Completed in {elapsed:.2f}s")
    print(f"  Images found: {len(images)}")

    if not images:
        print("  ⚠️  No images found — PDF may be text-only or scanned")

    # Save images
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for img in images:
        fname = f"p{img.page_num+1}_img{img.index}_{img.label}.png"
        fpath = OUTPUT_DIR / fname
        with open(fpath, "wb") as f:
            f.write(img.image_bytes)
        print(f"  📸 {fname} — {img.width}x{img.height} [{img.label}] "
              f"({len(img.image_bytes)/1024:.1f}KB)")

    if images:
        print(f"\n  💾 Saved → {OUTPUT_DIR}")

    # Page content summary
    print(f"\n  --- Page Content Summary ---")
    pages = extract_page_contents(pdf_bytes)
    for p in pages:
        n_imgs = len(p.images)
        text_len = len(p.text)
        has_text = "✅ text" if text_len > 50 else "⚠️ no text"
        has_imgs = f"📸 {n_imgs} images" if n_imgs else "—"
        print(f"  Trang {p.page_num+1}: {text_len:>5} chars ({has_text}), {has_imgs}")

    return images, pages


# ── 2. OCR Option A: PyMuPDF Native Text (free, instant) ────────────────────

def test_pymupdf_text(pdf_bytes: bytes):
    """Extract native text from PDF using PyMuPDF (no OCR model needed)."""
    import fitz

    print(f"\n{'='*70}")
    print(f"📄 TEST 2A: PyMuPDF Native Text Extraction (FREE)")
    print(f"{'='*70}")

    t0 = time.time()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_texts = []
    for page in doc:
        text = page.get_text("text")
        page_texts.append(text)
    doc.close()
    elapsed = time.time() - t0

    total_chars = sum(len(t) for t in page_texts)
    print(f"\n  ✅ Completed in {elapsed:.3f}s")
    print(f"  Pages: {len(page_texts)}")
    print(f"  Total chars: {total_chars}")

    for i, text in enumerate(page_texts):
        print(f"\n  --- Trang {i+1} ({len(text)} chars) ---")
        preview = text[:400].replace('\n', '\n   ')
        print(f"   {preview}")
        if len(text) > 400:
            print(f"   ... ({len(text)-400} more)")

    is_scanned = total_chars < 100
    if is_scanned:
        print(f"\n  ⚠️  PDF appears to be SCANNED (very little native text)")
        print(f"      → Need OCR engine (Cloud Vision or Local model)")
    else:
        print(f"\n  ✅ PDF has native text — OCR may not be needed for text content")

    return page_texts, is_scanned


# ── 3. OCR Option B: Cloud Vision (GPT-4o-mini) ─────────────────────────────

async def test_cloud_ocr(pdf_bytes: bytes):
    """OCR using OpenAI GPT-4o-mini Vision (requires API key)."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print(f"\n  ⚠️  OPENAI_API_KEY not set — skipping Cloud OCR test")
        return []

    print(f"\n{'='*70}")
    print(f"☁️  TEST 2B: Cloud Vision OCR (GPT-4o-mini)")
    print(f"{'='*70}")

    import fitz
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    model = os.getenv("LLM_MODEL_MINI", "gpt-4o-mini")

    VISION_PROMPT = """Bạn là công cụ nhận dạng bài toán toán học từ ảnh.
Hãy đọc và trích xuất TOÀN BỘ nội dung trong ảnh (đề bài, biểu thức, hình vẽ, chú thích, ...).
QUY TẮC:
- Công thức toán → LaTeX: $biểu_thức$ hoặc $$biểu_thức$$
- Giữ nguyên cấu trúc bài (câu hỏi / dữ kiện / yêu cầu)
- Nếu có hình vẽ/đồ thị, mô tả chi tiết
CHỈ trả về nội dung trích xuất. KHÔNG giải thích."""

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    max_pages = min(len(doc), 10)
    page_texts = []

    total_t0 = time.time()
    for page_num in range(max_pages):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")

        t0 = time.time()
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        }},
                    ],
                }],
                max_tokens=2000,
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            elapsed_p = time.time() - t0

            page_texts.append(text)
            print(f"\n  Trang {page_num+1}: {len(text)} chars ({elapsed_p:.1f}s)")
            preview = text[:300].replace('\n', '\n   ')
            print(f"   {preview}")
            if len(text) > 300:
                print(f"   ... ({len(text)-300} more)")

        except Exception as e:
            elapsed_p = time.time() - t0
            page_texts.append(f"[OCR Error: {e}]")
            print(f"\n  Trang {page_num+1}: ERROR ({elapsed_p:.1f}s) — {e}")

    doc.close()
    total_elapsed = time.time() - total_t0
    total_chars = sum(len(t) for t in page_texts)

    print(f"\n  ✅ Cloud OCR completed: {max_pages} pages, {total_chars} chars in {total_elapsed:.1f}s")
    return page_texts


# ── 4. Question Splitting (regex-based, no API) ─────────────────────────────

def test_question_split_regex(page_texts: list[str]):
    """Simple regex-based question splitting (no API cost)."""
    import re

    print(f"\n{'='*70}")
    print(f"📋 TEST 3: Question Splitting (regex)")
    print(f"{'='*70}")

    full_text = "\n".join(page_texts)
    # Vietnamese exam patterns
    pattern = r"(Câu\s+\d+[\.:]\s*)"
    parts = re.split(pattern, full_text, flags=re.IGNORECASE)

    questions = []
    i = 1
    while i < len(parts) - 1:
        header = parts[i].strip()
        content = parts[i+1].strip()
        if content:
            # Detect question type
            q_type = "mcq"
            if re.search(r"[a-d]\)\s", content):
                q_type = "true_false"
            elif not re.search(r"[A-D][\.\)]\s", content):
                q_type = "short_answer"

            questions.append({
                "number": header,
                "type": q_type,
                "content_preview": content[:150],
                "content_length": len(content),
            })
        i += 2

    print(f"\n  Questions found: {len(questions)}")
    for q in questions:
        print(f"  {q['number']} [{q['type']}] ({q['content_length']} chars)")
        print(f"    {q['content_preview'][:100]}...")

    return questions


# ── Main ────────────────────────────────────────────────────────────────────

async def main():
    print(f"📄 PDF: {PDF_PATH}")
    print(f"   Size: {PDF_PATH.stat().st_size / 1024:.1f}KB")

    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    # Page count
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = len(doc)
    doc.close()
    print(f"   Pages: {n_pages}")

    # Test 1: Image extraction
    images, pages = test_image_extraction(pdf_bytes)

    # Test 2A: PyMuPDF native text
    native_texts, is_scanned = test_pymupdf_text(pdf_bytes)

    # Test 2B: Cloud OCR (only if needed or for comparison)
    cloud_texts = await test_cloud_ocr(pdf_bytes)

    # Test 3: Question splitting
    best_texts = cloud_texts if cloud_texts and sum(len(t) for t in cloud_texts) > 100 else native_texts
    if sum(len(t) for t in best_texts) > 100:
        questions = test_question_split_regex(best_texts)
    else:
        print("\n⚠️  Not enough text for question splitting")
        questions = []

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'='*70}")
    native_chars = sum(len(t) for t in native_texts)
    cloud_chars = sum(len(t) for t in cloud_texts) if cloud_texts else 0
    print(f"  PDF: {PDF_PATH.name} ({n_pages} pages)")
    print(f"  Images extracted: {len(images)}")
    for img in images:
        print(f"    📸 p{img.page_num+1} [{img.label}] {img.width}x{img.height}")
    print(f"  PyMuPDF text: {native_chars} chars {'(scanned PDF)' if is_scanned else '(has text)'}")
    print(f"  Cloud OCR:    {cloud_chars} chars")
    print(f"  Questions:    {len(questions)}")
    print(f"  Output dir:   {OUTPUT_DIR}")

    # Save summary
    summary = {
        "pdf": PDF_PATH.name,
        "pages": n_pages,
        "is_scanned": is_scanned,
        "images_extracted": len(images),
        "native_text_chars": native_chars,
        "cloud_ocr_chars": cloud_chars,
        "questions_found": len(questions),
        "native_texts": native_texts,
        "cloud_texts": cloud_texts,
        "questions": questions,
    }
    out_path = OUTPUT_DIR / "test_results.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  💾 Results → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
