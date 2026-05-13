import asyncio
import sys
import logging
sys.path.append('.')

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from app.ocr.ocr_strategy import get_ocr_engine

files = [
    "data/exams/03a1a1a17294463cb4667ef9b6c3b34b.pdf",
    "data/exams/9f2d1b536eaa4ab29d8d28f4dadfbb60.pdf",
    "data/exams/41158149ae5a40ec9e7635e4ea4eb815.pdf"
]

async def main():
    ocr = get_ocr_engine("cloud")
    
    for file_path in files:
        print(f"\n{'='*50}\nTesting file: {file_path}")
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
                
            page_texts = await ocr.ocr_pdf(file_bytes)
            print(f"[OK] OCR hoan thanh. So trang thuc te da doc: {len(page_texts)}")
            for i, text in enumerate(page_texts):
                print(f"  - Trang {i+1}: {len(text)} ky tu")
                if "<END_OF_EXAM>" in text:
                    print(f"    --> (Phat hien <END_OF_EXAM> tai trang nay)")
                    
        except Exception as e:
            print(f"[ERROR] Loi: {e}")

if __name__ == "__main__":
    asyncio.run(main())
