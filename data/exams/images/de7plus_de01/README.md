# 📂 Thư mục ảnh — Đề 7+ (Đề số 1)

Đặt ảnh câu hỏi vào đây theo quy tắc đặt tên sau:

## Quy tắc đặt tên file

```
q{số thứ tự 2 chữ số}.png

Ví dụ:
  q01.png  → ảnh câu 1  (Phần 1, câu 1)
  q02.png  → ảnh câu 2
  q13.png  → ảnh câu 13
  ...
```

## Câu nào cần có ảnh?

Chỉ cần ảnh cho câu có **`"has_image": true`** trong file JSON.
Câu nào có `"has_image": false` thì bỏ qua.

Xem file: `data/exams/de7plus_de01.json`

## Cách crop ảnh nhanh

1. Mở file đề thi (PDF/scan)
2. Dùng **Snipping Tool** (Windows) hoặc **ShareX** để crop từng câu
3. Lưu vào thư mục này với tên `q{number}.png`

## Format ảnh được hỗ trợ

`.png` `.jpg` `.jpeg` `.webp`

> 💡 Khuyên dùng PNG để giữ độ nét của đồ thị toán học
