"""
Test OCR Pipeline — 2 Options (Cloud + Local) + Image/Graph Extraction.

Usage:
  python scripts/test_ocr_pipeline.py [--engine cloud|local|both] [--pdf PATH]

Tests:
  1. OCR text extraction (Cloud GPT-4o-mini / Local GOT-OCR2.0)
  2. Question splitting (LLM-based)
  3. Image/Graph extraction from PDF
  4. Save extracted images to data/exams/images/
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Load .env
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_ocr")

# Default test PDF
DEFAULT_PDF = ROOT / "data" / "exams" / "41158149ae5a40ec9e7635e4ea4eb815.pdf"


async def test_ocr_engine(pdf_bytes: bytes, engine_name: str):
    """Test a single OCR engine on the PDF."""
    from app.ocr.ocr_strategy import get_ocr_engine

    print(f"\n{'='*70}")
    print(f"🔍 Testing OCR Engine: {engine_name.upper()}")
    print(f"{'='*70}")

    engine = get_ocr_engine(engine_name)

    t0 = time.time()
    try:
        page_texts = await engine.ocr_pdf(pdf_bytes)
        elapsed = time.time() - t0

        total_chars = sum(len(t) for t in page_texts)
        print(f"\n✅ OCR completed in {elapsed:.1f}s")
        print(f"   Pages: {len(page_texts)}")
        print(f"   Total chars: {total_chars}")
        print(f"   Avg chars/page: {total_chars // max(len(page_texts), 1)}")

        # Show first 500 chars of each page
        for i, text in enumerate(page_texts):
            print(f"\n--- Trang {i+1} ({len(text)} chars) ---")
            preview = text[:500].replace('\n', '\n   ')
            print(f"   {preview}")
            if len(text) > 500:
                print(f"   ... ({len(text) - 500} chars more)")

        return page_texts

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n❌ OCR FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return []


async def test_question_splitting(page_texts: list[str]):
    """Test question splitting on OCR output."""
    from app.ocr.question_splitter import QuestionSplitter

    print(f"\n{'='*70}")
    print(f"📋 Testing Question Splitter")
    print(f"{'='*70}")

    raw_ocr = "\n\n---\n\n".join(
        f"[Trang {i+1}]\n{text}" for i, text in enumerate(page_texts)
    )

    splitter = QuestionSplitter()
    t0 = time.time()
    try:
        questions = await splitter.split(raw_ocr)
        elapsed = time.time() - t0

        print(f"\n✅ Split completed in {elapsed:.1f}s")
        print(f"   Questions found: {len(questions)}")

        for q in questions:
            q_num = q.get("question_number", "?")
            q_type = q.get("question_type", "?")
            content = q.get("content", "")
            print(f"\n   {q_num} [{q_type}]:")
            preview = content[:200].replace('\n', '\n      ')
            print(f"      {preview}")
            if len(content) > 200:
                print(f"      ... ({len(content) - 200} chars more)")

        return questions

    except Exception as e:
        print(f"\n❌ Split FAILED: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_image_extraction(pdf_bytes: bytes, output_dir: Path):
    """Test image/graph extraction from PDF."""
    from app.ocr.image_extractor import extract_images_from_pdf, extract_page_contents

    print(f"\n{'='*70}")
    print(f"🖼️  Testing Image/Graph Extraction")
    print(f"{'='*70}")

    t0 = time.time()
    images = extract_images_from_pdf(pdf_bytes)
    elapsed = time.time() - t0

    print(f"\n✅ Extraction completed in {elapsed:.1f}s")
    print(f"   Images found: {len(images)}")

    if not images:
        print("   ⚠️  No images found in PDF (may be text-only)")
        return images

    # Save images
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_info = []

    for img in images:
        filename = f"exam_p{img.page_num + 1}_img{img.index}_{img.label}.png"
        filepath = output_dir / filename
        with open(filepath, "wb") as f:
            f.write(img.image_bytes)

        info = {
            "filename": filename,
            "page": img.page_num + 1,
            "label": img.label,
            "size": f"{img.width}x{img.height}",
            "bbox": [round(b, 1) for b in img.bbox],
            "file_size": f"{len(img.image_bytes) / 1024:.1f}KB",
        }
        saved_info.append(info)
        print(f"   📸 {filename} — {img.width}x{img.height} [{img.label}] ({len(img.image_bytes)/1024:.1f}KB)")

    print(f"\n   💾 Saved {len(saved_info)} images → {output_dir}")

    # Also test page-level content extraction
    print(f"\n--- Page Content Summary ---")
    pages = extract_page_contents(pdf_bytes)
    for p in pages:
        n_imgs = len(p.images)
        text_len = len(p.text)
        print(f"   Trang {p.page_num + 1}: {text_len} chars text, {n_imgs} images")

    return images


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test OCR Pipeline")
    parser.add_argument("--engine", default="both", choices=["cloud", "local", "both"],
                        help="OCR engine to test")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF),
                        help="Path to test PDF")
    parser.add_argument("--skip-split", action="store_true",
                        help="Skip question splitting test")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        sys.exit(1)

    print(f"📄 Test PDF: {pdf_path}")
    print(f"   Size: {pdf_path.stat().st_size / 1024:.1f}KB")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Count pages
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        print(f"   Pages: {len(doc)}")
        doc.close()
    except Exception:
        pass

    # ── Test 1: Image Extraction (always run, no API cost) ──
    output_dir = ROOT / "data" / "exams" / "images" / pdf_path.stem
    images = test_image_extraction(pdf_bytes, output_dir)

    # ── Test 2: OCR ──
    engines = ["cloud", "local"] if args.engine == "both" else [args.engine]
    all_results = {}

    for eng in engines:
        page_texts = await test_ocr_engine(pdf_bytes, eng)
        all_results[eng] = page_texts

        # ── Test 3: Question Splitting (only for first successful OCR) ──
        if page_texts and not args.skip_split:
            questions = await test_question_splitting(page_texts)
            all_results[f"{eng}_questions"] = questions

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY")
    print(f"{'='*70}")
    print(f"PDF: {pdf_path.name}")
    print(f"Images extracted: {len(images)}")
    for eng in engines:
        texts = all_results.get(eng, [])
        total = sum(len(t) for t in texts)
        n_q = len(all_results.get(f"{eng}_questions", []))
        print(f"OCR [{eng}]: {len(texts)} pages, {total} chars → {n_q} questions")

    # Save results
    results_dir = ROOT / "data" / "exams" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{pdf_path.stem}_ocr_results.json"

    save_data = {
        "pdf": pdf_path.name,
        "images_extracted": len(images),
        "image_details": [
            {"page": img.page_num + 1, "label": img.label,
             "size": f"{img.width}x{img.height}"}
            for img in images
        ],
    }
    for eng in engines:
        texts = all_results.get(eng, [])
        save_data[f"ocr_{eng}"] = {
            "pages": len(texts),
            "total_chars": sum(len(t) for t in texts),
            "page_texts": texts,
        }
        qs = all_results.get(f"{eng}_questions", [])
        save_data[f"questions_{eng}"] = {
            "count": len(qs),
            "questions": qs,
        }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Results saved → {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
