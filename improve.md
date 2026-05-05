# Kế Hoạch Refactor: Định Hướng RAG Thuần Lý Thuyết & Kho Công Thức Chuyên Biệt

Dựa trên yêu cầu thay đổi luồng xử lý: **Loại bỏ exams khỏi RAG, chỉ dùng lý thuyết (theory) và áp dụng một kho công thức (formulas) độc lập định dạng JSON**, đồng thời thiết kế lại **Metadata đa kỹ năng (Multi-skills)**. Dưới đây là kế hoạch kiến trúc và lộ trình refactor chi tiết.

---

## I. Phân Tích Vấn Đề Với Luồng Hiện Tại
1. **Lẫn lộn Exams trong RAG**: Đưa trực tiếp đề thi vào VectorDB dễ gây ra rủi ro "Data Leakage" (LLM vô tình nhìn thấy đáp án và nói thẳng cho học sinh thay vì giảng giải).
2. **Thiếu chính xác với Công Thức Toán**: Các embedding models kém trong việc mã hóa toán học. Nếu vector search để tìm công thức lượng giác hay logarit, hệ thống rất dễ lấy nhầm công thức do độ tương đồng chữ (ví dụ $\sin(2x)$ và $\cos(2x)$).
3. **Mô hình 1 Câu Hỏi - 1 Kỹ Năng đã lỗi thời**: Một bài toán lớp 12 (ví dụ: Giải phương trình logarit kết hợp đặt ẩn phụ) luôn đòi hỏi **nhiều kỹ năng** và **nhiều công thức** đan xen. Việc giới hạn `skill_id` ở dạng string đơn lẻ làm mất đi bản chất của việc theo dõi kiến thức.

---

## II. Kiến Trúc RAG & Metadata Mới

### 1. Xóa bỏ VectorDB Exams, Tập trung Theory
- VectorDB (Chroma) từ nay **chỉ chứa văn bản Lý thuyết** (định nghĩa, tính chất, phương pháp giải, các ví dụ mẫu kinh điển).
- Loại bỏ hoàn toàn collection hoặc logic `search_exams`.

### 2. Xây dựng "Formula Registry" bằng JSON
Thay vì tìm kiếm vector cho công thức, ta sẽ dùng **Exact-match / Key-value Lookup**.
Tạo file tĩnh `data/formulas.json` chứa công thức Toán 12. Mỗi công thức là một thực thể độc lập, phân tách hoàn toàn khỏi skill.

```json
[
  {
    "id": "formula_log_product",
    "content": "**Công thức tích Logarit:**\n$$\\log_b(x \\cdot y) = \\log_b(x) + \\log_b(y)$$\n*Điều kiện:* $x > 0, y > 0, b > 0, b \\neq 1$",
    "metadata": {
      "chapter": "Hàm số Lũy thừa, Mũ và Logarit",
      "type": "formula"
    }
  },
  {
    "id": "formula_trig_double_sin",
    "content": "**Công thức nhân đôi hàm Sin:**\n$$\\sin(2x) = 2\\sin(x)\\cos(x)$$\n*Điều kiện:* Mọi $x \\in \\mathbb{R}$",
    "metadata": {
      "chapter": "Lượng giác",
      "type": "formula"
    }
  }
]
```

### 3. Thiết Kế Lại Metadata (Multi-Skills & Formulas Độc Lập)
Cấu trúc dữ liệu của một bài tập giờ đây sẽ gộp chung các kỹ năng thành một danh sách (không phân biệt chính/phụ) và có một trường riêng biệt cho công thức.
**Cũ:**
```json
{
  "question_id": "Q_123",
  "text": "...",
  "metadata": {"skill_id": "SKILL_LOG_EQ"}
}
```
**Mới:**
```json
{
  "question_id": "Q_123",
  "text": "Giải phương trình ...",
  "metadata": {
    "skill_ids": ["SKILL_LOG_EQ", "SKILL_QUADRATIC_EQ", "SKILL_DOMAIN"],
    "formula_ids": ["formula_log_product", "formula_log_power"]
  }
}
```

---

## III. Cập Nhật Prompt Cho Các Agent

Việc truyền nhiều kỹ năng và công thức đòi hỏi Teacher Agent và Assessor Agent phải có System Prompt rõ ràng hơn để tận dụng tốt Context.

### 1. Prompt cho Teacher Agent (Gia sư giảng dạy)
```text
Bạn là một gia sư Toán thông minh. Học sinh đang gặp khó khăn ở một bài toán tổng hợp cần vận dụng các kỹ năng: {skill_names_list}.

[CÔNG THỨC TRỌNG TÂM]
{formulas_list}
*Lưu ý: Luôn nhắc nhở học sinh kiểm tra Điều Kiện Xác Định của công thức trước khi áp dụng.*

[LÝ THUYẾT & PHƯƠNG PHÁP GIẢI CHUNG]
{theory_context}

[NHIỆM VỤ]
- KHÔNG cung cấp đáp án cuối cùng.
- Dựa vào lý thuyết và công thức ở trên, hãy đưa ra gợi ý bước 1 để học sinh tự làm.
- Nếu bài toán yêu cầu kết hợp nhiều kỹ năng, hãy hướng dẫn học sinh giải quyết từng kỹ năng một theo thứ tự logic.
```

### 2. Prompt cho Assessor Agent (Người chấm điểm)
```text
Bạn là người chấm bài. Hãy đánh giá câu trả lời của học sinh cho bài toán sau.

Bài toán yêu cầu các kỹ năng: {skill_names_list}.
Công thức áp dụng: {formulas_list}.

[NHIỆM VỤ]
1. Kiểm tra xem học sinh có áp dụng đúng công thức không? Có bỏ quên điều kiện xác định không?
2. Trong số các kỹ năng yêu cầu ({skill_names_list}), học sinh đã làm tốt kỹ năng nào và sai ở bước của kỹ năng nào?
3. Trả về định dạng JSON:
{
  "is_correct": false,
  "skills_assessed": {
     "SKILL_LOG_EQ": {"passed": true, "feedback": "Biến đổi logarit đúng"},
     "SKILL_DOMAIN": {"passed": false, "feedback": "Quên đặt điều kiện x > 0"}
  },
  "overall_feedback": "..."
}
```

---

## IV. Quy Trình RAG Pipeline Mới

Khi có câu hỏi, quy trình Retrieval diễn ra như sau:
1. **Trích xuất Metadata:** Truy xuất DB lấy danh sách `skill_ids` và `formula_ids` của bài toán.
2. **Dual-Retrieval (Truy xuất kép):**
   - **Nhánh 1 (Formulas):** Truy vấn thẳng vào `data/formulas.json` theo mảng `formula_ids` để lấy block LaTeX & điều kiện (O(1) lookup).
   - **Nhánh 2 (Theory):** Lấy danh sách `skill_ids` để filter trong ChromaDB, sau đó trích xuất văn bản phương pháp giải từ bộ `theory`.
3. **Prompt Injection:** Lắp ráp dữ liệu vào Prompt Template của Teacher/Assessor Agent như thiết kế ở phần III.

---

## V. Các Bước Triển Khai Chi Tiết (Action Plan Dành Cho Coding Agent)

Để một Coding Agent có thể đọc và tự động triển khai chính xác, dưới đây là file map và các thay đổi code cụ thể cần thực hiện:

- [ ] **Bước 1: Thanh lọc Dữ liệu VectorDB**
  - **File:** `app/rag/knowledge_base.py` (hoặc các scripts khởi tạo ChromaDB).
  - **Action:** Xóa bỏ hoàn toàn hàm `search_exams` và các logic liên quan. Chỉ giữ lại hàm `search_theory`. Đảm bảo không còn bất kỳ collection `exams` nào được gọi.

- [ ] **Bước 2: Cập nhật Schema Metadata của Database**
  - **File:** `app/db/models.py` (và các file Pydantic Schemas liên quan).
  - **Action:** Sửa đổi cấu trúc của bảng/entity `Question`. Xóa bỏ trường chuỗi tĩnh `skill_id: str`, thay thế bằng `skill_ids: list[str]`. Bổ sung thêm trường mới `formula_ids: list[str]`. Chỉnh sửa lại các hàm mock data hoặc insert DB để khớp với Schema mới này.

- [ ] **Bước 3: Nâng cấp Logic Knowledge Tracing (BKT)**
  - **File:** `app/knowledge_tracing/bkt.py` (hoặc module tương đương).
  - **Action:** Sửa đổi hàm `update_mastery(...)`. Thay vì chỉ nhận 1 biến `skill_id` đơn lẻ, hàm cần được refactor để nhận một JSON object `skills_assessed` (trả về từ Assessor Agent). Hàm sẽ lặp (loop) qua từng skill trong đó để tính toán và cập nhật xác suất thành thạo $P(L)$ độc lập.

- [ ] **Bước 4: Cập nhật RAG Pipeline & Hàm Lookup Công Thức**
  - **File:** `app/rag/knowledge_base.py` (hoặc tạo file mới `app/rag/formula_registry.py`).
  - **Action:** Viết thêm hàm `get_formulas_by_ids(formula_ids: list[str]) -> list[dict]`. Hàm này sẽ đọc file `data/formulas.json`, tìm kiếm các ID tương ứng và trả về nguyên block content (LaTeX và điều kiện). Đảm bảo lookup nhanh dạng O(1) hoặc cache in-memory.

- [ ] **Bước 5: Cập nhật System Prompt cho Các Agent**
  - **File:** `app/agents/teacher_agent.py` và `app/agents/assessor_agent.py`.
  - **Action:** Ghi đè biến `SYSTEM_PROMPT` theo cấu trúc đã đề cập ở Mục III. Cập nhật hàm `invoke()` hoặc `run()` của Agent để nhận thêm tham số list (`formulas_list`, `skill_names_list`) và format chúng thành chuỗi Markdown gọn gàng trước khi tiêm vào Prompt Context.
