import logging
import fitz
from app.ocr.ocr_strategy import detect_solution_boundary

logging.basicConfig(level=logging.DEBUG)

def test():
    pdf_path = "data/exams/9f2d1b536eaa4ab29d8d28f4dadfbb60.pdf"
    doc = fitz.open(pdf_path)
    boundary = detect_solution_boundary(doc, max_pages=20)
    print(f"Boundary detected at: {boundary}")

if __name__ == "__main__":
    test()
