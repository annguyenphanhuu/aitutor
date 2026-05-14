# Báo Cáo Đánh Giá AITutor — 6 Đề Thi (Model: gpt-5.4)

---

## I. Bảng Tổng Hợp Điểm Theo Đề

| Đề | Tổng câu | Accuracy | Đúng/Tổng | LLM Examiner | Step Clarity | Visual Avg | Vision Q |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Đề 1** | 22 | 95.5% 🟢 | 21/22 | 96.1% | 97.7% | 100.0% | 9 |
| **Đề 3** | 22 | 94.3% 🟡 | 20/22 | 93.6% | 95.5% | 100.0% | 6 |
| **Đề 4** | 22 | 89.8% 🟡 | 19/22 | 89.8% | 94.3% | 92.9% | 7 |
| **Đề 5** | 22 | 88.6% 🟡 | 18/22 | 92.3% | 96.6% | 100.0% | 5 |
| **Đề 8** | 22 | 77.3% 🔴 | 15/22 | 83.6% | 93.2% | 93.8% | 8 |
| **Đề 9** | 10 | 80.0% 🔴 | 7/10 | 79.5% | 87.5% | 90.0% | 5 |

> **Tổng cộng**: 120 câu | Đúng: 100 câu | **Accuracy TB: 87.6%** | LLM Examiner TB: 89.2%

---

## II. Bảng Chi Tiết Điểm Từng Câu

### Đề 1 (DE01) — Accuracy: 95.5% (21/22 câu) | LLM Examiner: 96.1%

| Câu | Loại | Chương | MathJudge | WS | Acc | Method | Steps | Know | Visual | Ghi chú |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **C1** ✅ | Trắc nghiệ | Đạo hàm | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C2** ✅ | Trắc nghiệ | Đạo hàm | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C3** ✅ | Trắc nghiệ | Nguyên hàm và Tích phân | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C4** ✅ | Trắc nghiệ | Hình học không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C5** ✅ | Trắc nghiệ | HÌNH HỌC KHÔNG GIAN | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C6** ✅ | Trắc nghiệ | HÌNH HỌC KHÔNG GIAN | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C7** ✅ | Trắc nghiệ | TỔ HỢP – XÁC SUẤT | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C8** ✅ | Trắc nghiệ | Các số đặc trưng đo mức độ phâ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C9** ✅ | Trắc nghiệ | Các số đặc trưng đo mức độ phâ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C10** ✅ | Trắc nghiệ | NGUYÊN HÀM VÀ TÍCH PHÂN | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C11** ✅ | Trắc nghiệ | Các số đặc trưng đo mức độ phâ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C12** ✅ | Trắc nghiệ | ĐẠO HÀM | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C13** ✅ | Đúng / Sai | HÌNH HỌC KHÔNG GIAN | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C14** ⚠️ | Đúng / Sai | ĐẠO HÀM | 1.00 | 0.95 | 1.00 | 1.00 | 0.75 | 1.00 | 1.00 |  |
| **C15** ✅ | Đúng / Sai | TỔ HỢP – XÁC SUẤT | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C16** ⚠️ | Đúng / Sai | TỔ HỢP – XÁC SUẤT | 1.00 | 0.85 | 1.00 | 0.50 | 1.00 | 1.00 | — |  |
| **C17** ✅ | Trả lời ng | ĐẠO HÀM | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C18** ✅ | Trả lời ng | HÌNH HỌC KHÔNG GIAN | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C19** ⚠️ | Trả lời ng | NGUYÊN HÀM VÀ TÍCH PHÂN | 0.00 | 0.35 | 0.00 | 0.50 | 0.75 | 0.50 | 1.00 | ❌ Sai đáp án |
| **C20** ✅ | Trả lời ng | ĐẠO HÀM | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C21** ✅ | Trả lời ng | TỔ HỢP – XÁC SUẤT | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C22** ✅ | Trả lời ng | TỔ HỢP – XÁC SUẤT | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |

### Đề 3 (DE03) — Accuracy: 94.3% (20/22 câu) | LLM Examiner: 93.6%

| Câu | Loại | Chương | MathJudge | WS | Acc | Method | Steps | Know | Visual | Ghi chú |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **C1** ✅ | Trắc nghiệ | Đạo hàm | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C2** ✅ | Trắc nghiệ | Đạo hàm | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C3** ✅ | Trắc nghiệ | Nguyên hàm và Tích phân | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C4** ✅ | Trắc nghiệ | Hình học không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C5** ✅ | Trắc nghiệ | Hình học không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C6** ✅ | Trắc nghiệ | Hình học không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C7** ✅ | Trắc nghiệ | Đạo hàm | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C8** ✅ | Trắc nghiệ | Hình học không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C9** ✅ | Trắc nghiệ | Nguyên hàm và Tích phân | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C10** ✅ | Trắc nghiệ | Các số đặc trưng đo mức độ phâ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C11** ✅ | Trắc nghiệ | Hình học không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C12** ✅ | Trắc nghiệ | Nguyên hàm và Tích phân | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C13** ⚠️ | Đúng / Sai | Hình học không gian | 1.00 | 0.80 | 1.00 | 0.50 | 0.75 | 1.00 | — |  |
| **C14** ⚠️ | Đúng / Sai | Đạo hàm | 0.75 | 0.90 | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | ⚠️ Đáp án chưa đủ |
| **C15** ✅ | Đúng / Sai | Nguyên hàm và Tích phân | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C16** ✅ | Đúng / Sai | Tổ hợp - Xác suất | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C17** ✅ | Trả lời ng | Nguyên hàm và Tích phân | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C18** ✅ | Trả lời ng | Ứng dụng Hình học | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C19** ✅ | Trả lời ng | Dãy số - Cấp số cộng và Cấp số | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C20** ✅ | Trả lời ng | Tổ hợp - Xác suất | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C21** ⚠️ | Trả lời ng | Tổ hợp - Xác suất | 1.00 | 0.75 | 1.00 | 0.50 | 0.75 | 0.50 | — |  |
| **C22** ⚠️ | Trả lời ng | Ứng dụng Đạo hàm - Khối đa diệ | 0.00 | 0.15 | 0.00 | 0.00 | 0.50 | 0.50 | 1.00 | ❌ Sai đáp án |

### Đề 4 (DE04) — Accuracy: 89.8% (19/22 câu) | LLM Examiner: 89.8%

| Câu | Loại | Chương | MathJudge | WS | Acc | Method | Steps | Know | Visual | Ghi chú |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **C1** ⚠️ | Trắc nghiệ | Đạo hàm | 0.00 | 0.10 | 0.00 | 0.00 | 0.50 | 0.00 | — | ❌ Sai đáp án |
| **C2** ✅ | Trắc nghiệ | Đạo hàm | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C3** ✅ | Trắc nghiệ | Nguyên hàm và Tích phân | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C4** ✅ | Trắc nghiệ | Hình học không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C5** ✅ | Trắc nghiệ | Hình học không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C6** ✅ | Trắc nghiệ | Hình học không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C7** ✅ | Trắc nghiệ | Đạo hàm | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C8** ✅ | Trắc nghiệ | Đạo hàm | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C9** ✅ | Trắc nghiệ | Các số đặc trưng đo mức độ phâ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C10** ✅ | Trắc nghiệ | Tổ hợp - Xác suất | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C11** ⚠️ | Trắc nghiệ | Dãy số | 0.00 | 0.60 | 0.00 | 1.00 | 1.00 | 1.00 | — | ❌ Sai đáp án |
| **C12** ✅ | Trắc nghiệ | Hình học không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C13** ✅ | Đúng / Sai | Hình học không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C14** ⚠️ | Đúng / Sai | Đạo hàm | 0.75 | 0.65 | 0.75 | 0.50 | 0.75 | 0.50 | — | ⚠️ Đáp án chưa đủ |
| **C15** ⚠️ | Đúng / Sai | Đạo hàm | 1.00 | 0.80 | 1.00 | 0.50 | 0.75 | 1.00 | 1.00 |  |
| **C16** ⚠️ | Đúng / Sai | Tổ hợp - Xác suất | 1.00 | 0.60 | 1.00 | 0.00 | 0.75 | 0.50 | — |  |
| **C17** ✅ | Trả lời ng | Hàm số luy thừa, mũ và logarit | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C18** ✅ | Trả lời ng | Hình học không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C19** ✅ | Trả lời ng | Hình học không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C20** ✅ | Trả lời ng | Đạo hàm | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C21** ✅ | Trả lời ng | Nguyên hàm và Tích phân | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C22** ✅ | Trả lời ng | Tổ hợp - Xác suất | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |

### Đề 5 (DE05) — Accuracy: 88.6% (18/22 câu) | LLM Examiner: 92.3%

| Câu | Loại | Chương | MathJudge | WS | Acc | Method | Steps | Know | Visual | Ghi chú |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **C1** ✅ | Trắc nghiệ | Hàm số lượng giác và phương tr | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C2** ✅ | Trắc nghiệ | Đạo hàm | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C3** ✅ | Trắc nghiệ | Nguyên hàm và Tích phân | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C4** ✅ | Trắc nghiệ | Nguyên hàm và Tích phân | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C5** ✅ | Trắc nghiệ | Hình học không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C6** ✅ | Trắc nghiệ | Hình học không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C7** ✅ | Trắc nghiệ | Đạo hàm | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C8** ✅ | Trắc nghiệ | Hình học không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C9** ✅ | Trắc nghiệ | Hàm số luy thừa, mũ và logarit | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C10** ✅ | Trắc nghiệ | Hình học không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C11** ✅ | Trắc nghiệ | Các số đặc trưng đo mức độ phâ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C12** ✅ | Trắc nghiệ | Đạo hàm | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C13** ✅ | Đúng / Sai | Đạo hàm | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C14** ⚠️ | Đúng / Sai | Nguyên hàm và Tích phân | 0.75 | 0.90 | 0.75 | 1.00 | 1.00 | 1.00 | — | ⚠️ Đáp án chưa đủ |
| **C15** ✅ | Đúng / Sai | Hình học không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C16** ⚠️ | Đúng / Sai | Xác suất | 0.75 | 0.70 | 0.75 | 0.50 | 0.75 | 1.00 | — | ⚠️ Đáp án chưa đủ |
| **C17** ✅ | Trả lời ng | Hàm số lũy thừa, mũ và logarit | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C18** ✅ | Trả lời ng | Các số đặc trưng đo mức độ phâ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C19** ✅ | Trả lời ng | Hàm số lũy thừa, mũ và logarit | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C20** ✅ | Trả lời ng | Đạo hàm | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C21** ⚠️ | Trả lời ng | Tổ hợp - Xác suất | 0.00 | 0.35 | 0.00 | 0.50 | 0.75 | 0.50 | — | ❌ Sai đáp án |
| **C22** ⚠️ | Trả lời ng | Nguyên hàm và Tích phân | 0.00 | 0.35 | 0.00 | 0.50 | 0.75 | 0.50 | 1.00 | ❌ Sai đáp án |

### Đề 8 (DE08) — Accuracy: 77.3% (15/22 câu) | LLM Examiner: 83.6%

| Câu | Loại | Chương | MathJudge | WS | Acc | Method | Steps | Know | Visual | Ghi chú |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **C1** ✅ | Trắc nghiệ | Dãy số - Cấp số cộng - Cấp số  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C2** ✅ | Trắc nghiệ | Các số đặc trưng đo mức độ phâ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C3** ✅ | Trắc nghiệ | Hệ tọa độ trong không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C4** ⚠️ | Trắc nghiệ | Ứng dụng đạo hàm để khảo sát v | 0.00 | 0.35 | 0.00 | 0.50 | 0.75 | 0.50 | — | ❌ Sai đáp án |
| **C5** ✅ | Trắc nghiệ | Khối đa diện | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C6** ✅ | Trắc nghiệ | Ứng dụng đạo hàm để khảo sát v | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C7** ✅ | Trắc nghiệ | Ứng dụng đạo hàm để khảo sát v | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C8** ⚠️ | Trắc nghiệ | Ứng dụng đạo hàm để khảo sát v | 0.00 | 0.40 | 0.00 | 0.50 | 0.75 | 1.00 | 1.00 | ❌ Sai đáp án |
| **C9** ✅ | Trắc nghiệ | Ứng dụng đạo hàm để khảo sát v | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C10** ✅ | Trắc nghiệ | Hệ tọa độ trong không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C11** ✅ | Trắc nghiệ | Vectơ trong không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C12** ✅ | Trắc nghiệ | Tổ hợp - Xác suất | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C13** ⚠️ | Đúng / Sai | Hệ tọa độ trong không gian | 0.50 | 0.40 | 0.50 | 0.00 | 0.75 | 0.50 | 1.00 | ⚠️ Đáp án chưa đủ |
| **C14** ⚠️ | Đúng / Sai | Hàm số mũ và hàm số lôgarit | 0.75 | 0.65 | 0.75 | 0.50 | 0.75 | 0.50 | — | ⚠️ Đáp án chưa đủ |
| **C15** ⚠️ | Đúng / Sai | Ứng dụng đạo hàm để khảo sát v | 0.75 | 0.65 | 0.75 | 0.50 | 0.75 | 0.50 | 1.00 | ⚠️ Đáp án chưa đủ |
| **C16** ✅ | Đúng / Sai | Thống kê | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C17** ✅ | Trả lời ng | Tổ hợp - Xác suất | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C18** ✅ | Trả lời ng | Hệ tọa độ trong không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C19** ⚠️ | Trả lời ng | Đạo hàm | 0.00 | 0.35 | 0.00 | 0.50 | 0.75 | 0.50 | 1.00 | ❌ Sai đáp án |
| **C20** ✅ | Trả lời ng | Hình học không gian | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C21** ⚠️ | Trả lời ng | Ứng dụng đạo hàm để khảo sát v | 0.00 | 0.60 | 0.00 | 1.00 | 1.00 | 1.00 | — | ❌ Sai đáp án |
| **C22** ✅ | Trả lời ng | Thống kê | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |

### Đề 9 (DE09) — Accuracy: 80.0% (7/10 câu) | LLM Examiner: 79.5%

| Câu | Loại | Chương | MathJudge | WS | Acc | Method | Steps | Know | Visual | Ghi chú |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **C13** ✅ | Đúng / Sai | Hàm số lũy thừa, hàm số mũ và  | 1.00 | — | 1.00 | — | — | — | — |  |
| **C14** ⚠️ | Đúng / Sai | Quan hệ vuông góc trong không  | 0.50 | 0.55 | 0.50 | 0.50 | 0.75 | 0.50 | 1.00 | ⚠️ Đáp án chưa đủ |
| **C15** ✅ | Đúng / Sai | Đạo hàm | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C16** ⚠️ | Đúng / Sai | Ứng dụng đạo hàm để khảo sát v | 0.50 | 0.55 | 0.50 | 0.50 | 0.75 | 0.50 | 1.00 | ⚠️ Đáp án chưa đủ |
| **C17** ✅ | Trả lời ng | Khối đa diện | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C18** ✅ | Trả lời ng | Hàm số lũy thừa, hàm số mũ và  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C19** ✅ | Trả lời ng | Ứng dụng đạo hàm để khảo sát v | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C20** ✅ | Trả lời ng | Ứng dụng đạo hàm để khảo sát v | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| **C21** ✅ | Trả lời ng | Ứng dụng đạo hàm để khảo sát v | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |  |
| **C22** ⚠️ | Trả lời ng | Ứng dụng đạo hàm để khảo sát v | 0.00 | 0.35 | 0.00 | 0.50 | 0.75 | 0.50 | — | ❌ Sai đáp án |

---

## III. Phân Tích Các Câu Chưa Đạt Điểm Tối Đa

### Đề 1 — 3 câu chưa đạt tối đa

#### C14 — Đúng / Sai (T/F) | ĐẠO HÀM
- **Skill**: `derivative_applications`
- **Mức độ**: 🟡 **Thiếu sót nhỏ**
- **Điểm**: MathJudge=1.0 | Weighted Score=0.95 | Acc=1.0 | Method=1.0 | Steps=0.75 | Know=1.0
- **Đáp án đúng**: `a-T,b-F,c-F,d-F`
- **Đánh giá của LLM Examiner**: Phương pháp giải đúng, nhưng bảng biến thiên trình bày chưa đầy đủ và có một số sai sót nhỏ trong việc xác định khoảng đồng biến và nghịch biến.

#### C16 — Đúng / Sai (T/F) | TỔ HỢP – XÁC SUẤT
- **Skill**: `probability_basic`
- **Mức độ**: 🟡 **Thiếu sót nhỏ**
- **Điểm**: MathJudge=1.0 | Weighted Score=0.85 | Acc=1.0 | Method=0.5 | Steps=1.0 | Know=1.0
- **Đáp án đúng**: `a-F,b-T,c-T,d-F`
- **Đánh giá của LLM Examiner**: Phương pháp đúng hướng nhưng có sai sót nhỏ ở phần a và d. Trình bày rõ ràng, đầy đủ và vận dụng lý thuyết chính xác.

#### C19 — Trả lời ngắn (SA) | NGUYÊN HÀM VÀ TÍCH PHÂN
- **Skill**: `integral_applications`
- **Mức độ**: 🔴 **Sai hoàn toàn**
- **Điểm**: MathJudge=0.0 | Weighted Score=0.35 | Acc=0.0 | Method=0.5 | Steps=0.75 | Know=0.5
- **Đáp án đúng**: `9,8`
- **Đánh giá của LLM Examiner**: Phương pháp giải đúng hướng nhưng có một số sai sót trong việc xác định hàm số và tính diện tích. Trình bày có thiếu sót nhỏ và một số bước chưa rõ ràng.
- **Nguyên nhân**:
  - **Lỗi hỗn hợp**: Phương pháp chưa đúng dẫn đến kết quả sai

### Đề 3 — 4 câu chưa đạt tối đa

#### C13 — Đúng / Sai (T/F) | Hình học không gian
- **Skill**: `geometry_line_plane`
- **Mức độ**: 🟡 **Thiếu sót nhỏ**
- **Điểm**: MathJudge=1.0 | Weighted Score=0.8 | Acc=1.0 | Method=0.5 | Steps=0.75 | Know=1.0
- **Đáp án đúng**: `a-F,b-T,c-T,d-F`
- **Đánh giá của LLM Examiner**: Phương pháp giải đúng hướng nhưng có sai sót nhỏ ở phần a. Trình bày rõ ràng nhưng thiếu một số bước nhỏ trong phần tính toán.

#### C14 — Đúng / Sai (T/F) | Đạo hàm
- **Skill**: `function_survey`
- **Mức độ**: 🟡 **Sai một phần**
- **Điểm**: MathJudge=0.75 | Weighted Score=0.9 | Acc=0.75 | Method=1.0 | Steps=1.0 | Know=1.0
- **Đáp án đúng**: `a-T,b-T,c-F,d-T`
- **Đánh giá của LLM Examiner**: Phương pháp giải đúng, trình bày rõ ràng và vận dụng lý thuyết chính xác.
- **Nguyên nhân**:
  - **Sai 1/4 mệnh đề** (câu T/F): Đúng 3/4 phần, sai 1 mệnh đề

#### C21 — Trả lời ngắn (SA) | Tổ hợp - Xác suất
- **Skill**: `probability_basic`
- **Mức độ**: 🟡 **Thiếu sót nhỏ**
- **Điểm**: MathJudge=1.0 | Weighted Score=0.75 | Acc=1.0 | Method=0.5 | Steps=0.75 | Know=0.5
- **Đáp án đúng**: `0,02`
- **Đánh giá của LLM Examiner**: Phương pháp đúng hướng nhưng chưa tính đến sự phụ thuộc giữa các lần lấy sản phẩm. Trình bày có phần thiếu sót trong việc giải thích các bước.

#### C22 — Trả lời ngắn (SA) | Ứng dụng Đạo hàm - Khối đa diện
- **Skill**: `geometry_optimization`
- **Mức độ**: 🔴 **Sai hoàn toàn**
- **Điểm**: MathJudge=0.0 | Weighted Score=0.15 | Acc=0.0 | Method=0.0 | Steps=0.5 | Know=0.5
- **Đáp án đúng**: `7,3`
- **Đánh giá của LLM Examiner**: Phương pháp giải sai hoàn toàn, không đúng với lý thuyết về khối chóp tứ giác đều. Trình bày có một số bước rõ ràng nhưng thiếu logic và không đầy đủ.
- **Nguyên nhân**:
  - **Lỗi phương pháp**: Áp dụng sai kiến thức/công thức hoàn toàn
  - **Thiếu lập luận phương pháp**: Không trình bày rõ cách tiếp cận
  - **Trình bày thiếu các bước**: Bỏ qua bước trung gian quan trọng

### Đề 4 — 5 câu chưa đạt tối đa

#### C1 — Trắc nghiệm (MCQ) | Đạo hàm
- **Skill**: `function_survey`
- **Mức độ**: 🔴 **Sai hoàn toàn**
- **Điểm**: MathJudge=0.0 | Weighted Score=0.1 | Acc=0.0 | Method=0.0 | Steps=0.5 | Know=0.0
- **Đáp án đúng**: `B` | **Model trả lời**: `D`
- **Đánh giá của LLM Examiner**: Phương pháp giải sai hoàn toàn, không áp dụng đúng lý thuyết về tiệm cận và tính đồng biến của hàm số.
- **Nguyên nhân**:
  - **Lỗi phương pháp**: Áp dụng sai kiến thức/công thức hoàn toàn
  - **Thiếu lập luận phương pháp**: Không trình bày rõ cách tiếp cận
  - **Vận dụng kiến thức yếu**: Chưa nêu rõ công thức/lý thuyết áp dụng
  - **Trình bày thiếu các bước**: Bỏ qua bước trung gian quan trọng

#### C11 — Trắc nghiệm (MCQ) | Dãy số
- **Skill**: `sequence_arithmetic`
- **Mức độ**: 🔴 **Sai hoàn toàn**
- **Điểm**: MathJudge=0.0 | Weighted Score=0.6 | Acc=0.0 | Method=1.0 | Steps=1.0 | Know=1.0
- **Đáp án đúng**: `B` | **Model trả lời**: `D`
- **Đánh giá của LLM Examiner**: Phương pháp giải đúng, trình bày rõ ràng và áp dụng công thức chính xác.
- **Nguyên nhân**:
  - **Lỗi tính toán**: Phương pháp đúng nhưng tính toán sai kết quả cuối cùng

#### C14 — Đúng / Sai (T/F) | Đạo hàm
- **Skill**: `derivative_applications`
- **Mức độ**: 🟡 **Sai một phần**
- **Điểm**: MathJudge=0.75 | Weighted Score=0.65 | Acc=0.75 | Method=0.5 | Steps=0.75 | Know=0.5
- **Đáp án đúng**: `a-F,b-T,c-F,d-T`
- **Đánh giá của LLM Examiner**: Phương pháp giải đúng hướng nhưng có một số sai sót trong việc xác định giá trị lớn nhất và thời điểm tốc độ thay đổi lớn nhất. Trình bày có phần thiếu sót và cần rõ ràng hơn.
- **Nguyên nhân**:
  - **Sai 1/4 mệnh đề** (câu T/F): Đúng 3/4 phần, sai 1 mệnh đề

#### C15 — Đúng / Sai (T/F) | Đạo hàm
- **Skill**: `function_survey`
- **Mức độ**: 🟡 **Thiếu sót nhỏ**
- **Điểm**: MathJudge=1.0 | Weighted Score=0.8 | Acc=1.0 | Method=0.5 | Steps=0.75 | Know=1.0
- **Đáp án đúng**: `a-T,b-F,c-T,d-T`
- **Đánh giá của LLM Examiner**: Phương pháp đúng nhưng có sai sót nhỏ trong việc xác định tiệm cận đứng. Trình bày có phần thiếu rõ ràng ở một số bước.

#### C16 — Đúng / Sai (T/F) | Tổ hợp - Xác suất
- **Skill**: `probability_conditional`
- **Mức độ**: 🟠 **Trình bày kém**
- **Điểm**: MathJudge=1.0 | Weighted Score=0.6 | Acc=1.0 | Method=0.0 | Steps=0.75 | Know=0.5
- **Đáp án đúng**: `a-F,b-F,c-T,d-T`
- **Đánh giá của LLM Examiner**: Phương pháp sai hoàn toàn, nhưng trình bày có logic và rõ ràng, tuy nhiên thiếu một số bước nhỏ và không nêu rõ công thức áp dụng.
- **Nguyên nhân**:
  - **Thiếu lập luận phương pháp**: Không trình bày rõ cách tiếp cận

### Đề 5 — 4 câu chưa đạt tối đa

#### C14 — Đúng / Sai (T/F) | Nguyên hàm và Tích phân
- **Skill**: `integral_applications`
- **Mức độ**: 🟡 **Sai một phần**
- **Điểm**: MathJudge=0.75 | Weighted Score=0.9 | Acc=0.75 | Method=1.0 | Steps=1.0 | Know=1.0
- **Đáp án đúng**: `a-T,b-F,c-F,d-T`
- **Đánh giá của LLM Examiner**: Phương pháp giải đúng, trình bày rõ ràng và logic, vận dụng lý thuyết chính xác.
- **Nguyên nhân**:
  - **Sai 1/4 mệnh đề** (câu T/F): Đúng 3/4 phần, sai 1 mệnh đề

#### C16 — Đúng / Sai (T/F) | Xác suất
- **Skill**: `probability_conditional`
- **Mức độ**: 🟡 **Sai một phần**
- **Điểm**: MathJudge=0.75 | Weighted Score=0.7 | Acc=0.75 | Method=0.5 | Steps=0.75 | Know=1.0
- **Đáp án đúng**: `a-T,b-F,c-F,d-F`
- **Đánh giá của LLM Examiner**: Phương pháp giải đúng hướng nhưng có sai sót nhỏ ở mệnh đề (b). Trình bày có thiếu một bước nhỏ trong việc tính xác suất toàn phần.
- **Nguyên nhân**:
  - **Sai 1/4 mệnh đề** (câu T/F): Đúng 3/4 phần, sai 1 mệnh đề

#### C21 — Trả lời ngắn (SA) | Tổ hợp - Xác suất
- **Skill**: `combinatorics_probability`
- **Mức độ**: 🔴 **Sai hoàn toàn**
- **Điểm**: MathJudge=0.0 | Weighted Score=0.35 | Acc=0.0 | Method=0.5 | Steps=0.75 | Know=0.5
- **Đáp án đúng**: `17`
- **Đánh giá của LLM Examiner**: Phương pháp đúng hướng nhưng có một số sai sót trong cách tính số cách chọn và sắp xếp các kí tự. Trình bày có phần thiếu rõ ràng và cần thêm một số bước để dễ theo dõi.
- **Nguyên nhân**:
  - **Lỗi hỗn hợp**: Phương pháp chưa đúng dẫn đến kết quả sai

#### C22 — Trả lời ngắn (SA) | Nguyên hàm và Tích phân
- **Skill**: `integral_applications_area`
- **Mức độ**: 🔴 **Sai hoàn toàn**
- **Điểm**: MathJudge=0.0 | Weighted Score=0.35 | Acc=0.0 | Method=0.5 | Steps=0.75 | Know=0.5
- **Đáp án đúng**: `0,79`
- **Đánh giá của LLM Examiner**: Phương pháp giải đúng hướng nhưng có một số sai sót trong việc tính toán và áp dụng công thức. Trình bày có phần thiếu rõ ràng và cần thêm một số bước để dễ theo dõi.
- **Nguyên nhân**:
  - **Lỗi hỗn hợp**: Phương pháp chưa đúng dẫn đến kết quả sai

### Đề 8 — 7 câu chưa đạt tối đa

#### C4 — Trắc nghiệm (MCQ) | Ứng dụng đạo hàm để khảo sát và vẽ đồ thị hàm số
- **Skill**: `function_min_max`
- **Mức độ**: 🔴 **Sai hoàn toàn**
- **Điểm**: MathJudge=0.0 | Weighted Score=0.35 | Acc=0.0 | Method=0.5 | Steps=0.75 | Know=0.5
- **Đáp án đúng**: `A` | **Model trả lời**: `D`
- **Đánh giá của LLM Examiner**: Phương pháp đúng hướng nhưng có sai sót trong việc xác định các điểm cần xét và tính giá trị hàm số. Trình bày có phần thiếu rõ ràng và không nêu rõ các bước tính toán.
- **Nguyên nhân**:
  - **Lỗi hỗn hợp**: Phương pháp chưa đúng dẫn đến kết quả sai

#### C8 — Trắc nghiệm (MCQ) | Ứng dụng đạo hàm để khảo sát và vẽ đồ thị hàm số
- **Skill**: `function_survey`
- **Mức độ**: 🔴 **Sai hoàn toàn**
- **Điểm**: MathJudge=0.0 | Weighted Score=0.4 | Acc=0.0 | Method=0.5 | Steps=0.75 | Know=1.0
- **Đáp án đúng**: `D` | **Model trả lời**: `C`
- **Đánh giá của LLM Examiner**: Phương pháp đúng hướng nhưng có một số sai sót nhỏ trong việc xác định tiệm cận. Trình bày có phần thiếu sót và cần rõ ràng hơn.
- **Nguyên nhân**:
  - **Lỗi hỗn hợp**: Phương pháp chưa đúng dẫn đến kết quả sai

#### C13 — Đúng / Sai (T/F) | Hệ tọa độ trong không gian
- **Skill**: `geometry_coordinate`
- **Mức độ**: 🟡 **Sai một phần**
- **Điểm**: MathJudge=0.5 | Weighted Score=0.4 | Acc=0.5 | Method=0.0 | Steps=0.75 | Know=0.5
- **Đáp án đúng**: `a-F,b-T,c-T,d-T`
- **Đánh giá của LLM Examiner**: Phương pháp giải hoàn toàn sai, không dựa trên lý thuyết đúng. Trình bày có logic nhưng thiếu một số bước nhỏ và không nêu rõ các công thức áp dụng.
- **Nguyên nhân**:
  - **Sai 2/4 mệnh đề** (câu T/F): Chỉ đúng 2/4 phần
  - **Thiếu lập luận phương pháp**: Không trình bày rõ cách tiếp cận

#### C14 — Đúng / Sai (T/F) | Hàm số mũ và hàm số lôgarit
- **Skill**: `logarithm_equation_inequation`
- **Mức độ**: 🟡 **Sai một phần**
- **Điểm**: MathJudge=0.75 | Weighted Score=0.65 | Acc=0.75 | Method=0.5 | Steps=0.75 | Know=0.5
- **Đáp án đúng**: `a-T,b-T,c-F,d-T`
- **Đánh giá của LLM Examiner**: Phương pháp giải đúng hướng nhưng có một số sai sót trong việc xác định nghiệm và điều kiện. Trình bày có phần thiếu sót và không rõ ràng ở một số bước.
- **Nguyên nhân**:
  - **Sai 1/4 mệnh đề** (câu T/F): Đúng 3/4 phần, sai 1 mệnh đề

#### C15 — Đúng / Sai (T/F) | Ứng dụng đạo hàm để khảo sát và vẽ đồ thị hàm số
- **Skill**: `function_survey_properties`
- **Mức độ**: 🟡 **Sai một phần**
- **Điểm**: MathJudge=0.75 | Weighted Score=0.65 | Acc=0.75 | Method=0.5 | Steps=0.75 | Know=0.5
- **Đáp án đúng**: `a-T,b-T,c-F,d-F`
- **Đánh giá của LLM Examiner**: Phương pháp giải đúng hướng nhưng có một số sai sót trong việc tính toán và kết luận. Trình bày có phần thiếu rõ ràng và logic ở một số bước, cần bổ sung thêm để dễ theo dõi.
- **Nguyên nhân**:
  - **Sai 1/4 mệnh đề** (câu T/F): Đúng 3/4 phần, sai 1 mệnh đề

#### C19 — Trả lời ngắn (SA) | Đạo hàm
- **Skill**: `derivative_application_physics`
- **Mức độ**: 🔴 **Sai hoàn toàn**
- **Điểm**: MathJudge=0.0 | Weighted Score=0.35 | Acc=0.0 | Method=0.5 | Steps=0.75 | Know=0.5
- **Đáp án đúng**: `8`
- **Đánh giá của LLM Examiner**: Phương pháp đúng hướng nhưng có sai sót trong việc xác định khoảng thời gian vật chuyển động nhanh dần. Trình bày có thiếu sót nhỏ và không nêu rõ một số công thức.
- **Nguyên nhân**:
  - **Lỗi hỗn hợp**: Phương pháp chưa đúng dẫn đến kết quả sai

#### C21 — Trả lời ngắn (SA) | Ứng dụng đạo hàm để khảo sát và vẽ đồ thị hàm số
- **Skill**: `optimization_revenue_profit`
- **Mức độ**: 🔴 **Sai hoàn toàn**
- **Điểm**: MathJudge=0.0 | Weighted Score=0.6 | Acc=0.0 | Method=1.0 | Steps=1.0 | Know=1.0
- **Đáp án đúng**: `41`
- **Đánh giá của LLM Examiner**: Phương pháp giải đúng, trình bày rõ ràng và áp dụng đúng lý thuyết.
- **Nguyên nhân**:
  - **Lỗi tính toán**: Phương pháp đúng nhưng tính toán sai kết quả cuối cùng

### Đề 9 — 3 câu chưa đạt tối đa

#### C14 — Đúng / Sai (T/F) | Quan hệ vuông góc trong không gian
- **Skill**: `geometry_solid_distance_volume`
- **Mức độ**: 🟡 **Sai một phần**
- **Điểm**: MathJudge=0.5 | Weighted Score=0.55 | Acc=0.5 | Method=0.5 | Steps=0.75 | Know=0.5
- **Đáp án đúng**: `a-T,b-T,c-F,d-T`
- **Đánh giá của LLM Examiner**: Phương pháp giải đúng hướng nhưng có một số sai sót nhỏ trong việc xác định mệnh đề và trình bày chưa đầy đủ. Cần làm rõ hơn trong việc áp dụng lý thuyết và các bước giải.
- **Nguyên nhân**:
  - **Sai 2/4 mệnh đề** (câu T/F): Chỉ đúng 2/4 phần

#### C16 — Đúng / Sai (T/F) | Ứng dụng đạo hàm để khảo sát và vẽ đồ thị hàm số
- **Skill**: `function_properties_from_graph`
- **Mức độ**: 🟡 **Sai một phần**
- **Điểm**: MathJudge=0.5 | Weighted Score=0.55 | Acc=0.5 | Method=0.5 | Steps=0.75 | Know=0.5
- **Đáp án đúng**: `a-F,b-F,c-T,d-T`
- **Đánh giá của LLM Examiner**: Phương pháp giải đúng hướng nhưng có sai sót trong việc xác định hệ số a và b. Trình bày có phần thiếu rõ ràng và logic, cần thêm một số bước để giải thích rõ hơn. Vận dụng lý thuyết chưa đầy đủ khi không nêu rõ các công thức liên quan.
- **Nguyên nhân**:
  - **Sai 2/4 mệnh đề** (câu T/F): Chỉ đúng 2/4 phần

#### C22 — Trả lời ngắn (SA) | Ứng dụng đạo hàm để khảo sát và vẽ đồ thị hàm số
- **Skill**: `composite_function_equation`
- **Mức độ**: 🔴 **Sai hoàn toàn**
- **Điểm**: MathJudge=0.0 | Weighted Score=0.35 | Acc=0.0 | Method=0.5 | Steps=0.75 | Know=0.5
- **Đáp án đúng**: `9`
- **Đánh giá của LLM Examiner**: Phương pháp đúng hướng nhưng có sai sót trong việc tính tổng số nghiệm. Trình bày có phần thiếu rõ ràng và không nêu rõ các bước giải.
- **Nguyên nhân**:
  - **Lỗi hỗn hợp**: Phương pháp chưa đúng dẫn đến kết quả sai

---

## IV. Tổng Hợp Nguyên Nhân & Nhận Xét Chung

### Thống kê câu sai theo loại câu

| Loại câu | Tổng | Chưa đạt | Tỉ lệ lỗi |
|---|:---:|:---:|:---:|
| Trắc nghiệm (MC | 60 | 4 | 6.7% |
| Đúng / Sai (T/F | 24 | 14 | 58.3% |
| Trả lời ngắn (S | 36 | 8 | 22.2% |

### Thống kê theo chương

| Chương | Tổng | Lỗi | Tỉ lệ |
|---|:---:|:---:|:---:|
| Ứng dụng đạo hàm để khảo sát và vẽ  🔴 | 12 | 6 | 50.0% |
| Đạo hàm 🟡 | 20 | 5 | 25.0% |
| Tổ hợp - Xác suất 🟡 | 9 | 3 | 33.3% |
| Nguyên hàm và Tích phân | 12 | 2 | 16.7% |
| Hình học không gian | 20 | 1 | 5.0% |
| TỔ HỢP – XÁC SUẤT | 5 | 1 | 20.0% |
| NGUYÊN HÀM VÀ TÍCH PHÂN 🔴 | 2 | 1 | 50.0% |
| ĐẠO HÀM 🟡 | 4 | 1 | 25.0% |
| Ứng dụng Đạo hàm - Khối đa diện 🔴 | 1 | 1 | 100.0% |
| Dãy số 🔴 | 1 | 1 | 100.0% |
| Xác suất 🔴 | 1 | 1 | 100.0% |
| Hệ tọa độ trong không gian 🟡 | 4 | 1 | 25.0% |
| Hàm số mũ và hàm số lôgarit 🔴 | 1 | 1 | 100.0% |
| Quan hệ vuông góc trong không gian 🔴 | 1 | 1 | 100.0% |
| HÌNH HỌC KHÔNG GIAN | 4 | 0 | 0.0% |
| Các số đặc trưng đo mức độ phân tán | 8 | 0 | 0.0% |
| Ứng dụng Hình học | 1 | 0 | 0.0% |
| Dãy số - Cấp số cộng và Cấp số nhân | 1 | 0 | 0.0% |
| Hàm số luy thừa, mũ và logarit | 2 | 0 | 0.0% |
| Hàm số lượng giác và phương trình l | 1 | 0 | 0.0% |
| Hàm số lũy thừa, mũ và logarit | 2 | 0 | 0.0% |
| Dãy số - Cấp số cộng - Cấp số nhân | 1 | 0 | 0.0% |
| Khối đa diện | 2 | 0 | 0.0% |
| Vectơ trong không gian | 1 | 0 | 0.0% |
| Thống kê | 2 | 0 | 0.0% |
| Hàm số lũy thừa, hàm số mũ và hàm s | 2 | 0 | 0.0% |

### Nhận xét tổng quan

1. **Hiệu suất tổng thể tốt**: Accuracy trung bình ~88%, đặc biệt Đề 1 đạt 95.5%
2. **Điểm yếu ở Đề 8**: Accuracy chỉ 77.3% (15/22) — đề khó nhất trong bộ
3. **Câu Trả lời ngắn (SA) khó nhất**: Tỉ lệ lỗi cao, thường do tính toán sai bước cuối
4. **Câu T/F (Đúng/Sai)**: Hay sai 1 mệnh đề trong 4 — thường là mệnh đề (c) hoặc (d)
5. **Điểm Method thấp nhất**: Nhiều câu đúng đáp án nhưng thiếu lập luận phương pháp
6. **Chương hay sai nhất**:
   - Ứng dụng đạo hàm (khảo sát hàm số)
   - Tổ hợp – Xác suất (câu phức tạp)
   - Hình học không gian (câu tối ưu hóa)
7. **Lỗi đặc trưng theo dạng câu**:
   - MCQ: Chọn nhầm do phân tích đồ thị chưa chính xác (Vision questions)
   - SA: Phương pháp đúng hướng nhưng tính toán sai giá trị cuối
   - T/F: Sai mệnh đề liên quan đến điều kiện biên (min/max, tiệm cận)