import asyncio
import logging
import fitz
import time
from app.ocr.ocr_strategy import get_ocr_engine, detect_solution_boundary
from app.ocr.question_splitter import QuestionSplitter
from app.config import get_settings

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

settings = get_settings()

files_to_test = [
    "data/exams/03a1a1a17294463cb4667ef9b6c3b34b.pdf",
    "data/exams/9f2d1b536eaa4ab29d8d28f4dadfbb60.pdf",
    "data/exams/41158149ae5a40ec9e7635e4ea4eb815.pdf"
]

async def test_file(file_path: str, run_idx: int):
    logger.info(f"\n--- Testing {file_path.split('/')[-1]} (Run {run_idx}/5) ---")
    
    # Check boundary detection
    doc = fitz.open(file_path)
    max_pages = settings.MAX_EXAM_PAGES
    boundary = detect_solution_boundary(doc, max_pages)
    total_pages = len(doc)
    n_pages = min(boundary, max_pages, total_pages)
    logger.info(f"Boundary detected: page {boundary} (will OCR {n_pages}/{total_pages} pages)")
    doc.close()
    
    # Run full OCR
    with open(file_path, "rb") as f:
        file_bytes = f.read()
        
    ocr_engine = get_ocr_engine()
    ocr_engine.ocr_engine_name = "cloud"
    
    start_time = time.time()
    page_texts = await ocr_engine.ocr_pdf(file_bytes)
    
    raw_ocr = "\n\n---\n\n".join(f"[Trang {i+1}]\n{text}" for i, text in enumerate(page_texts))
    
    splitter = QuestionSplitter()
    questions = await splitter.split(raw_ocr)
    
    elapsed = time.time() - start_time
    logger.info(f"Result: {len(page_texts)} pages OCR'd, {len(questions)} questions extracted in {elapsed:.1f}s")
    return len(page_texts), len(questions)

async def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    for f in files_to_test:
        logger.info(f"\n{'='*50}\nTesting File: {f}\n{'='*50}")
        tasks = [test_file(f, i) for i in range(1, 4)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.error(f"Error on {f} run {i+1}: {res}")
            else:
                logger.info(f"Run {i+1} result: {res}")

if __name__ == "__main__":
    asyncio.run(main())
