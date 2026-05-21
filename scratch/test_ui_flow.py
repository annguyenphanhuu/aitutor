import asyncio
import logging
from app.ocr.ocr_strategy import OCRStrategy
from app.config import get_settings

logging.basicConfig(level=logging.DEBUG)

async def test_ui_flow():
    with open("data/exams/9f2d1b536eaa4ab29d8d28f4dadfbb60.pdf", "rb") as f:
        file_bytes = f.read()

    # Flow 1: Extract OCR
    ocr = OCRStrategy(engine="cloud")
    page_texts = await ocr.ocr_pdf(file_bytes)
    raw_ocr = "\n\n---\n\n".join(
        f"[Trang {i+1}]\n{text}" for i, text in enumerate(page_texts)
    )
    
    print(f"OCR returned {len(page_texts)} pages.")
    
    # Check length
    from app.ocr.question_splitter import QuestionSplitter
    splitter = QuestionSplitter()
    questions = await splitter.split(raw_ocr)
    
    print(f"Splitter extracted {len(questions)} questions.")

if __name__ == "__main__":
    asyncio.run(test_ui_flow())
