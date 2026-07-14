import asyncio
from app.ocr.question_splitter import QuestionSplitter

async def main():
    text = """
Câu 5. Cho hàm số $y = f(x)$ có bảng biến thiên như sau:
Mệnh đề nào sau đây đúng?
A. Hàm số đồng biến trên $\mathbb{R}$.
B. Hàm số nghịch biến trên $(-\infty; 1)$.
C. Hàm số đồng biến trên $(0; 1)$.
D. Hàm số nghịch biến trên $(0; 1)$.

PHẦN III. Câu trắc nghiệm trả lời ngắn.
Câu 6. Tìm giá trị lớn nhất của hàm số $y = -x^2 + 2x$ trên đoạn $[0; 3]$.

HƯỚNG DẪN GIẢI CHI TIẾT
PHẦN I.
Câu 1: Đáp án A.
Hàm số đã cho đồng biến trên khoảng $(0; +\infty)$.
Câu 2: Đáp án B.
Ta có $y' = 3x^2 - 3 = 0 \Leftrightarrow x = \pm 1$.
    """
    splitter = QuestionSplitter()
    qs = await splitter.split(text)
    import json
    print(json.dumps(qs, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
