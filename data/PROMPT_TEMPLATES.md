# 📋 PROMPT TEMPLATES — Dán vào LLM để sinh JSON chuẩn

Dùng file này như một **thư viện prompt**. Copy prompt phù hợp, điền nội dung vào `[...]`.

Sau khi LLM trả về JSON → lưu vào `data/new_batch.json` → chạy:
```
python scripts/append_data.py
python scripts/ingest.py
```

---

## DANH SÁCH SKILL_ID hợp lệ (6 chương thi)

```
── ĐẠO HÀM ────────────────────────────────────────────────
derivative_basic        | Đạo hàm cơ bản
derivative_rules        | Quy tắc đạo hàm (tích, thương, hàm hợp)
derivative_applications | Ứng dụng đạo hàm (cực trị, đơn điệu)
function_survey         | Khảo sát hàm số (bảng biến thiên, đồ thị)

── NGUYÊN HÀM VÀ TÍCH PHÂN ────────────────────────────────
primitive_basic         | Nguyên hàm cơ bản
integral_definite       | Tích phân xác định (Newton–Leibniz)
integral_applications   | Ứng dụng tích phân (diện tích, thể tích)

── DÃY SỐ ─────────────────────────────────────────────────
sequence_basic          | Dãy số cơ bản (số hạng tổng quát, đơn điệu)
arithmetic_sequence     | Cấp số cộng (CSC)
geometric_sequence      | Cấp số nhân (CSN)

── HÌNH HỌC KHÔNG GIAN ─────────────────────────────────────
geometry_vectors        | Vectơ trong không gian (tọa độ Oxyz)
geometry_line_plane     | Đường thẳng và mặt phẳng
geometry_distance_angle | Khoảng cách và góc
geometry_sphere         | Mặt cầu

── TỔ HỢP – XÁC SUẤT ───────────────────────────────────────
combinatorics_basic     | Tổ hợp – Chỉnh hợp – Hoán vị – Nhị thức Newton
probability_basic       | Xác suất cơ bản
probability_distribution| Phân phối xác suất (nhị thức, kỳ vọng)

── THỐNG KÊ ────────────────────────────────────────────────
statistics_descriptive  | Thống kê mô tả (trung bình, trung vị, mốt, tứ phân vị)
statistics_inference    | Độ lệch chuẩn, tần suất, ước lượng
```

## TYPE hợp lệ (chỉ 2 loại)

```
theory                        | Lý thuyết: định nghĩa, công thức, tính chất
exam_mcq                      | Câu trắc nghiệm từ đề thi thật (4 đáp án)
exam_true_false               | Câu đúng/sai 4 mệnh đề từ đề thi thật
exam_short_answer             | Câu trả lời ngắn từ đề thi thật
```

---

## ═══════════════════════════════════════════════════════
## PROMPT A: Lý thuyết SGK → JSON
## ═══════════════════════════════════════════════════════

```
Bạn là giáo viên Toán 12, am hiểu chương trình GDPT 2018 Việt Nam.
Chuyển nội dung lý thuyết sau thành JSON array.

THÔNG TIN:
- skill_id : [ĐIỀN SKILL_ID, ví dụ: integral_definite]
- chapter  : [ĐIỀN TÊN CHƯƠNG, ví dụ: Tích phân]
- Tiêu đề  : [ĐIỀN TIÊU ĐỀ, ví dụ: Tích phân xác định và tính chất]

NỘI DUNG LÝ THUYẾT CẦN CHUYỂN:
---
[DÁN NỘI DUNG VÀO ĐÂY]
---

FORMAT TỪNG DOCUMENT (trả về JSON array):
{
  "id"      : "{skill_id_viết_tắt}_theory_{số 3 chữ số}",
  "content" : "CHƯƠNG: {CHAPTER UPPERCASE}\n\n{TIÊU ĐỀ UPPERCASE}:\n\n{nội dung đầy đủ}",
  "metadata": {
    "skill_id" : "{skill_id}",
    "chapter"  : "{chapter}",
    "type"     : "theory"
  }
}

QUY TẮC:
- Mỗi khái niệm/định lý lớn = 1 document riêng (không nhồi nhét)
- Giữ nguyên công thức toán học, ký hiệu đặc biệt (∀, ∈, ⟹, …)
- CHỈ TRẢ VỀ JSON ARRAY, không có text hay markdown fence khác
```

---

## ═══════════════════════════════════════════════════════
## PROMPT B: Đề thi MCQ (Phần 1) → JSON
## ═══════════════════════════════════════════════════════

```
Bạn là giáo viên Toán 12, am hiểu đề thi THPT Quốc gia.
Chuyển các câu trắc nghiệm sau thành JSON array.

THÔNG TIN ĐỀ THI:
- exam_id     : [ĐIỀN, ví dụ: thpt_2024_ma001]
- year        : [ĐIỀN NĂM, ví dụ: 2024]
- exam_source : [ĐIỀN NGUỒN, ví dụ: "THPT Quốc gia" hoặc "ĐGNL HUST"]

CÂU HỎI CẦN CHUYỂN (Phần 1 - Trắc nghiệm nhiều phương án):
---
[DÁN NỘI DUNG CÁC CÂU HỎI + ĐÁP ÁN VÀO ĐÂY]
---

FORMAT TỪNG DOCUMENT (trả về JSON array):
{
  "id"      : "exam_{year}_{exam_id}_p1_q{số 2 chữ số}",
  "content" : "ĐỀ {EXAM_SOURCE UPPERCASE} {YEAR} ({EXAM_ID UPPERCASE}) - CÂU {N} [Phần 1 - Trắc nghiệm]:\n\n{đề bài}\n\nA. {đáp án A}\nB. {đáp án B}\nC. {đáp án C}\nD. {đáp án D}\n\nĐÁP ÁN: {đáp án đúng}\n\nGIẢI THÍCH:\n{lời giải ngắn gọn}",
  "metadata": {
    "exam_id"         : "{exam_id}",
    "year"            : {year — số nguyên},
    "exam_source"     : "{exam_source}",
    "question_number" : {số thứ tự — số nguyên},
    "difficulty_part" : 1,
    "skill_id"        : "{skill_id phù hợp nhất}",
    "chapter"         : "{tên chương}",
    "type"            : "exam_mcq",
    "correct_answer"  : "{A|B|C|D}"
  }
}

QUY TẮC:
- difficulty_part luôn là 1 cho Phần 1 (MCQ)
- skill_id phải là một trong các giá trị hợp lệ ở trên
- correct_answer là chữ cái đáp án đúng: "A", "B", "C" hoặc "D"
- CHỈ TRẢ VỀ JSON ARRAY, không có text hay markdown fence khác
```

---

## ═══════════════════════════════════════════════════════
## PROMPT C: Đề thi Đúng/Sai (Phần 2) → JSON
## ═══════════════════════════════════════════════════════

```
Bạn là giáo viên Toán 12, am hiểu đề thi THPT Quốc gia.
Chuyển các câu Đúng/Sai sau thành JSON array.

THÔNG TIN ĐỀ THI:
- exam_id     : [ĐIỀN, ví dụ: thpt_2024_ma001]
- year        : [ĐIỀN NĂM]
- exam_source : [ĐIỀN NGUỒN]

CÂU HỎI CẦN CHUYỂN (Phần 2 - Đúng/Sai 4 mệnh đề):
---
[DÁN NỘI DUNG VÀO ĐÂY]
---

FORMAT TỪNG DOCUMENT:
{
  "id"      : "exam_{year}_{exam_id}_p2_q{số 2 chữ số}",
  "content" : "ĐỀ {EXAM_SOURCE UPPERCASE} {YEAR} - CÂU {N} [Phần 2 - Đúng/Sai]:\n\n{đề bài}\n\na) {mệnh đề a}  →  {ĐÚNG|SAI}\nb) {mệnh đề b}  →  {ĐÚNG|SAI}\nc) {mệnh đề c}  →  {ĐÚNG|SAI}\nd) {mệnh đề d}  →  {ĐÚNG|SAI}\n\nĐÁP ÁN: a-{Đ|S}, b-{Đ|S}, c-{Đ|S}, d-{Đ|S}\n\nGIẢI THÍCH:\n{lời giải}",
  "metadata": {
    "exam_id"         : "{exam_id}",
    "year"            : {year},
    "exam_source"     : "{exam_source}",
    "question_number" : {số thứ tự},
    "difficulty_part" : 2,
    "skill_id"        : "{skill_id}",
    "chapter"         : "{tên chương}",
    "type"            : "exam_true_false",
    "correct_answer"  : "a-T,b-F,c-T,d-T"
  }
}

QUY TẮC:
- difficulty_part luôn là 2 cho Phần 2 (Đúng/Sai)
- correct_answer dạng: "a-T,b-F,c-T,d-T" (T=True/Đúng, F=False/Sai)
- CHỈ TRẢ VỀ JSON ARRAY, không có text hay markdown fence khác
```

---

## ═══════════════════════════════════════════════════════
## PROMPT D: Đề thi Trả lời ngắn (Phần 3) → JSON
## ═══════════════════════════════════════════════════════

```
Bạn là giáo viên Toán 12, am hiểu đề thi THPT Quốc gia.
Chuyển các câu Trả lời ngắn sau thành JSON array.

THÔNG TIN ĐỀ THI:
- exam_id     : [ĐIỀN]
- year        : [ĐIỀN NĂM]
- exam_source : [ĐIỀN NGUỒN]

CÂU HỎI CẦN CHUYỂN (Phần 3 - Trả lời ngắn):
---
[DÁN NỘI DUNG VÀO ĐÂY]
---

FORMAT TỪNG DOCUMENT:
{
  "id"      : "exam_{year}_{exam_id}_p3_q{số 2 chữ số}",
  "content" : "ĐỀ {EXAM_SOURCE UPPERCASE} {YEAR} - CÂU {N} [Phần 3 - Trả lời ngắn]:\n\n{đề bài}\n\nĐÁP ÁN: {đáp án}\n\nGIẢI THÍCH:\n{lời giải chi tiết}",
  "metadata": {
    "exam_id"         : "{exam_id}",
    "year"            : {year},
    "exam_source"     : "{exam_source}",
    "question_number" : {số thứ tự},
    "difficulty_part" : 3,
    "skill_id"        : "{skill_id}",
    "chapter"         : "{tên chương}",
    "type"            : "exam_short_answer",
    "correct_answer"  : "{đáp án dạng số hoặc biểu thức}"
  }
}

QUY TẮC:
- difficulty_part luôn là 3 cho Phần 3 (Trả lời ngắn)
- correct_answer: số hoặc biểu thức đơn giản (ví dụ: "3", "1/2", "√5")
- CHỈ TRẢ VỀ JSON ARRAY, không có text hay markdown fence khác
```

---

## ═══════════════════════════════════════════════════════
## PROMPT E: MEGA — LLM tự sinh lý thuyết toàn bộ cho 1 skill
## (dùng khi chưa có tài liệu gốc)
## ═══════════════════════════════════════════════════════

```
Bạn là giáo viên Toán 12 giỏi, am hiểu chương trình GDPT 2018 Việt Nam.

Hãy tạo bộ lý thuyết đầy đủ cho kỹ năng:
- Tên kỹ năng : [ĐIỀN TÊN]
- skill_id    : [ĐIỀN SKILL_ID]
- chapter     : [ĐIỀN TÊN CHƯƠNG]
- Mô tả       : [ĐIỀN MÔ TẢ NGẮN]

Tạo CHÍNH XÁC 3–5 documents type "theory" bao gồm:
1. Định nghĩa, khái niệm cơ bản
2. Các tính chất, công thức quan trọng
3. Ví dụ minh họa có lời giải (nếu cần)

FORMAT mỗi document:
{
  "id"      : "{skill_id_viết_tắt}_theory_{001/002/...}",
  "content" : "CHƯƠNG: {CHAPTER UPPERCASE}\n\n{TIÊU ĐỀ}:\n\n{nội dung đầy đủ}",
  "metadata": {
    "skill_id" : "{skill_id}",
    "chapter"  : "{chapter}",
    "type"     : "theory"
  }
}

CHỈ TRẢ VỀ JSON ARRAY, KHÔNG CÓ TEXT KHÁC.
```

---

## Quy tắc đặt exam_id

```
thpt_{năm}_ma{mã đề}    → ví dụ: thpt_2024_ma001
dgnl_hust_{năm}         → ví dụ: dgnl_hust_2024
dgnl_vnu_{năm}          → ví dụ: dgnl_vnu_2023
```

## Workflow hoàn chỉnh

```
1. Chọn prompt phù hợp ở trên (A/B/C/D/E)
2. Điền thông tin vào [...]
3. Paste vào ChatGPT / Gemini / Claude
4. Copy JSON → lưu vào: data/new_batch.json
5. Chạy: python scripts/append_data.py
6. Chạy: python scripts/ingest.py
```

## Kiểm tra JSON nhanh

Dán JSON vào https://jsonlint.com hoặc Ctrl+Shift+P → "Format Document" trong VS Code.
