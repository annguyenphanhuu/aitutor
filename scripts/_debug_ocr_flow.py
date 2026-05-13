import asyncio
import sys
sys.path.append('.')
from app.ocr.ocr_strategy import get_ocr_engine

async def main():
    ocr = get_ocr_engine()
    with open('data/exams/03a1a1a17294463cb4667ef9b6c3b34b.pdf', 'rb') as f:
        file_bytes = f.read()
    res = await ocr.ocr_pdf(file_bytes)
    print(f"OCR returned {len(res)} pages")
    if len(res) > 0:
        print(f"Page 1 length: {len(res[0])}")
        print(f"Last page length: {len(res[-1])}")

if __name__ == '__main__':
    asyncio.run(main())
