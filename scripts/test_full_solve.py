"""
Test Full Hybrid Vision Pipeline — End-to-end từ PDF.

Chạy toàn bộ pipeline:
  1. OCR (Cloud Vision)
  2. Question Splitting (Structured Output + has_figure)
  3. Per-question Solving (Text-only OR Vision, with RAG)

Usage:
  python scripts/test_full_solve.py
"""

import asyncio, json, os, sys, time, base64
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_full_solve")

PDF_PATH = ROOT / "data" / "exams" / "41158149ae5a40ec9e7635e4ea4eb815.pdf"


async def main():
    import fitz
    from openai import AsyncOpenAI

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    client = AsyncOpenAI(api_key=api_key)
    model_mini = os.getenv("LLM_MODEL_MINI", "gpt-4o-mini")
    model_main = os.getenv("LLM_MODEL", "gpt-4o")

    print(f"📄 PDF: {PDF_PATH.name}")
    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = len(doc)
    print(f"   Pages: {n_pages}")

    # ── Step 1: OCR (4 pages of exam content) ────────────────────────
    print(f"\n{'='*70}")
    print(f"☁️  Step 1: Cloud Vision OCR")
    print(f"{'='*70}")

    VISION_PROMPT = """Bạn là công cụ nhận dạng đề thi Toán 12.
Đọc và trích xuất TOÀN BỘ nội dung. Công thức → LaTeX ($...$).
Giữ nguyên cấu trúc, số thứ tự câu. Mô tả hình vẽ/đồ thị nếu có.
Đánh dấu [Trang X] ở đầu. CHỈ trả về nội dung."""

    page_texts = []
    page_images = {}

    t0 = time.time()
    for pg_num in range(min(n_pages, 4)):
        page = doc[pg_num]
        pix_300 = page.get_pixmap(dpi=300)
        pix_200 = page.get_pixmap(dpi=200)
        page_images[pg_num + 1] = pix_200.tobytes("png")

        b64 = base64.b64encode(pix_300.tobytes("png")).decode("utf-8")
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
            max_tokens=2500, temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        page_texts.append(f"[Trang {pg_num+1}]\n{text}")
        print(f"  Trang {pg_num+1}: {len(text)} chars ✅")

    doc.close()
    raw_ocr = "\n\n---\n\n".join(page_texts)
    ocr_elapsed = time.time() - t0
    print(f"  ✅ OCR done: {len(raw_ocr)} chars in {ocr_elapsed:.1f}s")

    # ── Step 2: Question Splitting ────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"📋 Step 2: Question Splitting (Structured Output)")
    print(f"{'='*70}")

    from app.ocr.question_splitter import QuestionSplitter
    splitter = QuestionSplitter()

    t0 = time.time()
    questions = await splitter.split(raw_ocr)
    split_elapsed = time.time() - t0

    n_fig = sum(1 for q in questions if q.get("has_figure"))
    print(f"  ✅ {len(questions)} questions ({n_fig} with figures) in {split_elapsed:.1f}s")

    for q in questions:
        fig_icon = "🖼️" if q.get("has_figure") else "📝"
        print(f"  {fig_icon} {q['question_number']} [{q['question_type']}]")

    # ── Step 3: Per-question Solving ──────────────────────────────────
    print(f"\n{'='*70}")
    print(f"🎓 Step 3: Per-question Solving (test first 5 questions)")
    print(f"{'='*70}")

    # Solve first 5 questions as a demo (full solve would be expensive)
    test_questions = questions[:5]

    SOLVE_PROMPT = """Bạn là gia sư Toán 12. Giải câu hỏi sau ngắn gọn, chính xác.
- Công thức → LaTeX
- Trả lời bằng tiếng Việt
- Nếu trắc nghiệm: chọn đáp án + giải thích ngắn
- Nếu có hình vẽ trong ảnh: đọc kỹ hình trước khi giải"""

    results = []
    t0_solve = time.time()

    for i, q in enumerate(test_questions):
        content = q["content"]
        q_num = q["question_number"]
        has_figure = q.get("has_figure", False)
        figure_pages = q.get("figure_pages", [])

        t0_q = time.time()

        if has_figure and figure_pages and page_images:
            # Vision solve
            pg = figure_pages[0] if figure_pages[0] in page_images else 1
            img_bytes = page_images.get(pg, next(iter(page_images.values())))
            b64 = base64.b64encode(img_bytes).decode("utf-8")

            resp = await client.chat.completions.create(
                model=model_main,
                messages=[
                    {"role": "system", "content": SOLVE_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        }},
                        {"type": "text", "text": f"Giải {q_num}:\n{content}"},
                    ]},
                ],
                max_tokens=1500, temperature=0.1,
            )
            solution = resp.choices[0].message.content.strip()
            mode = "🖼️ Vision"
        else:
            # Text-only solve
            resp = await client.chat.completions.create(
                model=model_mini,
                messages=[
                    {"role": "system", "content": SOLVE_PROMPT},
                    {"role": "user", "content": f"Giải {q_num}:\n{content}"},
                ],
                max_tokens=1500, temperature=0.1,
            )
            solution = resp.choices[0].message.content.strip()
            mode = "📝 Text"

        elapsed_q = time.time() - t0_q

        results.append({
            "question_number": q_num,
            "question_type": q["question_type"],
            "has_figure": has_figure,
            "mode": mode,
            "solution": solution,
            "elapsed": round(elapsed_q, 1),
        })

        print(f"\n  ── {q_num} [{q['question_type']}] {mode} ({elapsed_q:.1f}s) ──")
        # Show first 300 chars of solution
        preview = solution[:300].replace('\n', '\n   ')
        print(f"   {preview}")
        if len(solution) > 300:
            print(f"   ... ({len(solution)-300} chars more)")

    solve_elapsed = time.time() - t0_solve

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  PDF: {PDF_PATH.name} ({n_pages} pages)")
    print(f"  OCR: {len(raw_ocr)} chars ({ocr_elapsed:.1f}s)")
    print(f"  Split: {len(questions)} questions, {n_fig} with figures ({split_elapsed:.1f}s)")
    print(f"  Solved: {len(results)} questions ({solve_elapsed:.1f}s)")

    n_vision = sum(1 for r in results if "Vision" in r["mode"])
    n_text = len(results) - n_vision
    print(f"    📝 Text-only: {n_text}")
    print(f"    🖼️  Vision:    {n_vision}")
    print(f"  Total time: {ocr_elapsed + split_elapsed + solve_elapsed:.1f}s")

    # Save
    output_dir = ROOT / "data" / "exams" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{PDF_PATH.stem}_full_solve.json"

    save_data = {
        "pdf": PDF_PATH.name,
        "ocr_chars": len(raw_ocr),
        "total_questions": len(questions),
        "questions_with_figure": n_fig,
        "solved_questions": len(results),
        "ocr_elapsed": round(ocr_elapsed, 1),
        "split_elapsed": round(split_elapsed, 1),
        "solve_elapsed": round(solve_elapsed, 1),
        "questions": questions,
        "solutions": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"  💾 Results → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
