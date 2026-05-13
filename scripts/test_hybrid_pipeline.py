"""
Test Hybrid Vision Pipeline — OCR + Split (has_figure) + Preview routing.

Tests the upgraded QuestionSplitter that detects figures/graphs
and marks which questions need Vision model vs text-only.

Usage:
  python scripts/test_hybrid_pipeline.py
"""

import asyncio, json, os, sys, time, base64
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

PDF_PATH = ROOT / "data" / "exams" / "41158149ae5a40ec9e7635e4ea4eb815.pdf"

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_hybrid")


async def main():
    import fitz
    from openai import AsyncOpenAI

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    client = AsyncOpenAI(api_key=api_key)
    model_mini = os.getenv("LLM_MODEL_MINI", "gpt-4o-mini")

    print(f"📄 PDF: {PDF_PATH.name}")
    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = len(doc)
    print(f"   Pages: {n_pages}")

    # ── Step 1: OCR all pages ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"☁️  Step 1: Cloud Vision OCR ({n_pages} pages)")
    print(f"{'='*70}")

    VISION_PROMPT = """Bạn là công cụ nhận dạng đề thi Toán 12.
Hãy đọc và trích xuất TOÀN BỘ nội dung trong ảnh.
QUY TẮC:
- Công thức toán → LaTeX: $biểu_thức$ hoặc $$biểu_thức$$
- Giữ nguyên cấu trúc, số thứ tự câu
- Mô tả hình vẽ/đồ thị nếu có (vd: "Đồ thị hàm số có dạng cong, đi qua gốc O")
- Đánh dấu [Trang X] ở đầu mỗi trang
CHỈ trả về nội dung. KHÔNG giải thích."""

    page_texts = []
    page_images = {}  # page_num (1-indexed) → PNG bytes

    t0 = time.time()
    for pg_num in range(min(n_pages, 4)):  # Test first 4 pages (đề thi chính)
        page = doc[pg_num]
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        page_images[pg_num + 1] = page.get_pixmap(dpi=200).tobytes("png")  # 200 DPI for solving

        b64 = base64.b64encode(img_bytes).decode("utf-8")

        try:
            response = await client.chat.completions.create(
                model=model_mini,
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
                max_tokens=2500,
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            page_texts.append(f"[Trang {pg_num+1}]\n{text}")
            print(f"  Trang {pg_num+1}: {len(text)} chars ✅")
        except Exception as e:
            page_texts.append(f"[Trang {pg_num+1}]\n[OCR Error: {e}]")
            print(f"  Trang {pg_num+1}: ERROR — {e}")

    doc.close()
    ocr_elapsed = time.time() - t0
    raw_ocr = "\n\n---\n\n".join(page_texts)
    print(f"\n  ✅ OCR done: {len(raw_ocr)} chars in {ocr_elapsed:.1f}s")

    # ── Step 2: Question Split (with has_figure) ───────────────────────
    print(f"\n{'='*70}")
    print(f"📋 Step 2: Question Splitting (LLM with figure detection)")
    print(f"{'='*70}")

    # Import and use the upgraded splitter
    from app.ocr.question_splitter import QuestionSplitter
    splitter = QuestionSplitter()

    t0 = time.time()
    questions = await splitter.split(raw_ocr)
    split_elapsed = time.time() - t0

    n_total = len(questions)
    n_fig = sum(1 for q in questions if q.get("has_figure"))
    n_text = n_total - n_fig

    print(f"\n  ✅ Split done in {split_elapsed:.1f}s")
    print(f"  Total: {n_total} questions")
    print(f"  📝 Text-only: {n_text} → will use text LLM (cheaper)")
    print(f"  🖼️  With figure: {n_fig} → will use Vision LLM")

    print(f"\n  {'─'*60}")
    for q in questions:
        fig_icon = "🖼️" if q.get("has_figure") else "📝"
        fig_desc = f" [{q.get('figure_description', '')}]" if q.get("has_figure") else ""
        fig_pages = f" (p{q.get('figure_pages', [])})" if q.get("figure_pages") else ""
        content_preview = q["content"][:80].replace("\n", " ")
        print(f"  {fig_icon} {q['question_number']} [{q['question_type']}]{fig_desc}{fig_pages}")
        print(f"     {content_preview}...")

    # ── Save results ────────────────────────────────────────────────
    output_dir = ROOT / "data" / "exams" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{PDF_PATH.stem}_hybrid_test.json"

    save_data = {
        "pdf": PDF_PATH.name,
        "pages_ocr": len(page_texts),
        "ocr_elapsed_s": round(ocr_elapsed, 1),
        "split_elapsed_s": round(split_elapsed, 1),
        "total_questions": n_total,
        "questions_with_figure": n_fig,
        "questions_text_only": n_text,
        "questions": questions,
        "raw_ocr": raw_ocr,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 Results → {results_path}")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY")
    print(f"{'='*70}")
    print(f"  OCR: {len(raw_ocr)} chars ({ocr_elapsed:.1f}s)")
    print(f"  Split: {n_total} questions ({split_elapsed:.1f}s)")
    print(f"  📝 Text-only: {n_text} → respond() [GPT-4o-mini, ~$0.002/q]")
    print(f"  🖼️  Vision:    {n_fig} → respond_with_image() [GPT-4o, ~$0.03/q]")
    est_cost = n_text * 0.002 + n_fig * 0.03
    print(f"  💰 Est. solve cost: ~${est_cost:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
