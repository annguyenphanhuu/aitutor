# Báo Cáo Dự Án AITutor: Gia Sư AI Thích Nghi Cho Toán Lớp 12

## 1. Tổng Quan Dự Án
**AITutor** là một hệ thống gia sư AI thông minh chuyên hỗ trợ học sinh học Toán lớp 12. Hệ thống được thiết kế để không chỉ giải quyết các bài toán mà còn theo dõi quá trình học tập, nắm bắt năng lực của học sinh và đưa ra lộ trình cá nhân hóa dựa trên mô hình Bayesian Knowledge Tracing (BKT) và Spaced Repetition (Lặp lại ngắt quãng).

## 2. Kiến Trúc Kỹ Thuật (Tech Stack)
Dự án được xây dựng với các công nghệ hiện đại, đáp ứng hiệu năng cao và tính module hóa:
- **Backend Framework:** FastAPI (chạy với Uvicorn), hỗ trợ async toàn diện.
- **Cơ Sở Dữ Liệu:** SQLite qua `aiosqlite` và ORM `SQLAlchemy`.
- **LLM & AI Orchestration:** LangChain, OpenAI API (hỗ trợ các model như GPT-4o-mini, o4-mini).
- **Hệ Thống Vector & RAG:** 
  - ChromaDB lưu trữ vector.
  - Hybrid Search kết hợp BM25 (tìm kiếm từ khóa) và Vector search.
  - Cross-Encoder Re-ranking (ms-marco-MiniLM-L-6-v2) để tăng độ chính xác truy xuất.
- **Xử Lý OCR & PDF:** OpenAI Vision cho OCR đám mây, tích hợp chiến lược fallback/local với GOT-OCR2.0. Sử dụng `PyMuPDF` để trích xuất ảnh và chuyển đổi PDF sang ảnh phục vụ Multi-Modal Vision.
- **Giám Sát & Tracking:** Langfuse, OpenTelemetry phục vụ logging và cost tracking.
- **Evaluation:** RAGAS framework để đánh giá chất lượng câu trả lời.

## 3. Các Chức Năng Cốt Lõi

### 3.1. Hỗ Trợ Trò Chuyện & Streaming Chat (Agents)
- **Orchestrator Agent:** Đóng vai trò phân loại ngữ cảnh, điều phối yêu cầu người dùng tới các Agent phù hợp.
- **Teacher Agent:** Trực tiếp giải quyết các bài toán thông qua quá trình RAG, phân tích sâu và cung cấp lời giải (Socratic method hoặc Direct Answer).
- Hỗ trợ **Server-Sent Events (SSE)** giúp truyền dữ liệu trực tuyến (streaming) lời giải cho người dùng.

### 3.2. Hệ Thống Giải Đề Thi Tự Động (Exam Solver Pipeline)
Một quy trình (pipeline) mạnh mẽ xử lý nguyên một đề thi:
1. **Tải lên & Trích xuất:** Hỗ trợ định dạng PDF và Hình ảnh. Tự động trích xuất các biểu đồ, hình vẽ (Standalone images).
2. **OCR:** Sử dụng OpenAI Vision (GPT-4o-mini) để quét toàn bộ văn bản và công thức Toán học.
3. **Phân tách câu hỏi (Question Splitter):** Dùng LLM để tách đề thành từng câu hỏi độc lập.
4. **Giải song song (Concurrency):** Chạy giải các câu hỏi bằng Teacher Agent (sử dụng kỹ thuật Semaphore để tối ưu thời gian giải và không vượt quá limit API). Tích hợp RAG riêng biệt cho từng câu hỏi nhằm đảm bảo độ chính xác.
5. **Tổng hợp:** Kết xuất báo cáo Markdown và LaTeX kèm thống kê.

### 3.3. Dấu Vết Kiến Thức & Lặp Lại Ngắt Quãng (Knowledge Tracing & Spaced Repetition)
- **Bayesian Knowledge Tracing (BKT):** Ước tính xác suất thành thạo từng kỹ năng (skill) của học sinh sau mỗi lần tương tác, giúp xác định cấp độ (Novice, Intermediate, Proficient...).
- **Adaptive Quiz:** Đề xuất mức độ câu hỏi linh hoạt phụ thuộc vào chỉ số thành thạo.
- **Spaced Repetition:** Hệ thống thẻ ghi nhớ tự động sinh ra khi học sinh thực hành các kỹ năng mới, nhắc nhở ôn tập định kỳ dựa vào thuật toán giãn cách thời gian.

### 3.4. RAG và Cây Kiến Thức
- Duy trì kho dữ liệu cấu trúc (GraphRAG) chứa toàn bộ công thức và các kỹ năng Toán 12 (skill graph), phục vụ cho việc nhúng ngữ cảnh trực tiếp vào LLM prompt, hạn chế các lỗi suy diễn nhầm kiến thức.

## 4. Tiến Độ Hiện Tại & Hoạt Động Gần Đây

Dựa trên cấu trúc thư mục, code và nhật ký tương tác:
1. **Phát triển Core AI & OCR:** Pipeline giải đề đa phương thức (Multi-Modal) đã được xây dựng thành công. Hệ thống đang được tinh chỉnh khả năng nhận diện điểm dừng (Solution Boundary) để tránh đưa các phần "Đáp án có sẵn" vào quá trình xử lý, giúp tối ưu chi phí API.
2. **Đánh Giá Mô Hình (Benchmarking):** Đang thực hiện các đợt chạy thử tự động qua các tệp script `run_eval_monitor.py` và `run_o4_evals.py` trên các bộ đề chuẩn (de01, de03, de04, de05, de06). Mục tiêu so sánh độ chính xác và chi phí của các model AI như `o4-mini-2025-04-16`, các thế hệ gpt-5.x.
3. **Làm giàu cơ sở dữ liệu bài tập:** Tệp đề thi `de7plus_de06.json` vừa được số hóa bổ sung hệ thống.
4. **Tối ưu Prompt & Template Logic:** Gần đây đã xử lý các vấn đề trùng lặp trong việc trích xuất tham số, tinh chỉnh lời nhắc liên quan tới nhận diện đặc điểm toán học phức tạp.

## 5. Kết Luận
**AITutor** là một dự án công nghệ giáo dục chuyên sâu, hoàn thiện ở mức độ cao về cả kiến trúc thiết kế hệ thống lẫn quy trình áp dụng AI tạo sinh (Generative AI). Pipeline xử lý tài liệu thông minh và BKT biến dự án thành một giải pháp học tập cá nhân hóa lý tưởng, có tiềm năng thương mại hóa và mở rộng sang các môn học khác ngoài Toán 12.
