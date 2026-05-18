# Kế hoạch Triển khai AITutor lên Production (AWS Free Tier & CI/CD)

Kế hoạch này hướng dẫn chi tiết cách đưa dự án AITutor lên môi trường production, sử dụng **AWS RDS PostgreSQL** (Cơ sở dữ liệu - Gói Free Tier 12 tháng), **Hugging Face Spaces** (Backend API), kết hợp **AWS S3** (Lưu trữ file) và **GitHub Actions** (CI/CD Tự động).

## Giai đoạn 1: Khởi tạo Cơ sở dữ liệu (AWS RDS PostgreSQL - Free Tier)

1. Đăng nhập vào [AWS Management Console](https://aws.amazon.com/console/) và truy cập dịch vụ **RDS**.
2. Nhấn **Create database**.
   - Method: **Standard create**.
   - Engine options: **PostgreSQL**.
   - Templates: Chọn **Free tier** (Rất quan trọng để không bị tính phí trong 12 tháng đầu tiên).
3. Cấu hình thông tin (Settings):
   - **DB instance identifier**: `aitutor-db`
   - **Master username** & **Master password**: Đặt tên đăng nhập và mật khẩu (nhớ lưu lại cẩn thận).
4. Cấu hình kết nối (Connectivity):
   - **Public access**: Chọn **Yes** (Bắt buộc để backend trên Hugging Face có thể kết nối được tới database này).
   - **VPC security group**: Chọn *Create new* và đặt tên (ví dụ: `rds-hf-sg`).
5. Cuộn xuống cuối và nhấn **Create database**. Quá trình tạo sẽ mất vài phút.
6. Khi trạng thái Database chuyển thành *Available*, click vào tên database. Ở tab **Connectivity & security**, sao chép giá trị **Endpoint**.
7. Chỉnh sửa Inbound Rules của Security Group: 
   - Nhấn vào tên Security Group trong phần VPC security groups.
   - Chọn **Edit inbound rules**, thêm một rule mới: Type `PostgreSQL`, Source `Anywhere-IPv4` (`0.0.0.0/0`). Lưu lại để cho phép Hugging Face kết nối.
8. Xây dựng chuỗi kết nối (Connection String):
   - Định dạng: `postgresql+asyncpg://<username>:<password>@<endpoint>:5432/postgres`
   - Ví dụ: `postgresql+asyncpg://postgres:matkhau123@aitutor-db.abc123xyz.ap-southeast-1.rds.amazonaws.com:5432/postgres`

## Giai đoạn 2: Điều chỉnh Source Code cho Hugging Face Spaces

Hugging Face Spaces bắt buộc ứng dụng phải chạy ở cổng **7860** (thay vì 8000 như bạn đang dùng ở local).

1. **Sửa file `Dockerfile`:**
   Mở `Dockerfile` và sửa 2 dòng cuối cùng:
   ```dockerfile
   # Đổi port EXPOSE từ 8000 sang 7860
   EXPOSE 7860
   
   # Đổi port khởi động Uvicorn
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
   ```

2. **Cập nhật CORS (nếu cần thiết):**
   Trong file khởi tạo app FastAPI (`app/main.py`), đảm bảo bạn cho phép truy cập từ các domain frontend của bạn. Khi lên Hugging Face, domain của backend sẽ có dạng `https://ten-user-ten-space.hf.space`.

## Giai đoạn 3: Khởi tạo Hugging Face Space

1. Truy cập [Hugging Face](https://huggingface.co/) và tạo tài khoản.
2. Vào trang cá nhân, chọn **New Space**.
   - Tên Space: `AITutor-API` (hoặc tùy ý).
   - License: MIT.
   - Select the Space SDK: Chọn **Docker** (Blank).
   - Space hardware: Để mặc định **Free** (16GB RAM, 2 CPU).
   - Nhấn **Create Space**.

3. **Cấu hình Biến môi trường (Secrets):**
   Hugging Face sẽ không đọc file `.env` của bạn vì tính bảo mật.
   Trong giao diện Space vừa tạo, chuyển sang tab **Settings** -> cuộn xuống phần **Variables and secrets** -> nhấn **New secret**.
   Bạn cần thêm toàn bộ các thông tin quan trọng từ file `.env` của dự án vào đây:
   - `DATABASE_URL`: Chèn chuỗi kết nối AWS RDS PostgreSQL vừa tạo ở Bước 1.
   - `OPENAI_API_KEY`: Khóa API của bạn.
   - Các biến AWS: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET`, `AWS_REGION`.
   - Các biến Langfuse: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.

## Giai đoạn 4: Thiết lập CI/CD bằng GitHub Actions

Chúng ta sẽ thiết lập để mỗi khi bạn push code lên GitHub, GitHub sẽ tự động đẩy sang Hugging Face để server cập nhật tự động.

1. **Lấy Token từ Hugging Face:**
   - Trên Hugging Face, vào góc trên bên phải ảnh đại diện -> **Settings** -> **Access Tokens**.
   - Tạo Token mới (Type: `Write`), đặt tên là `DEPLOY_TOKEN`. Sao chép đoạn mã này.

2. **Thêm Token vào GitHub:**
   - Vào repository code của bạn trên **GitHub** -> **Settings** -> **Secrets and variables** -> **Actions**.
   - Nhấn **New repository secret**.
   - Name: `HF_TOKEN`
   - Secret: *Dán mã token vừa copy ở Hugging Face vào*.

3. **Tạo luồng CI/CD trong Source Code:**
   Mở terminal / VSCode trên máy của bạn, tạo một thư mục `.github/workflows` trong thư mục gốc của project (nếu chưa có).
   Tạo file `.github/workflows/deploy.yml` với nội dung sau:

   ```yaml
   name: Deploy API to Hugging Face Spaces
   
   on:
     push:
       branches: [ main ] # Tự động chạy khi bạn gộp hoặc push code mới vào nhánh main
     workflow_dispatch: # Bật nút kích hoạt thủ công từ giao diện GitHub (nếu cần)
   
   jobs:
     sync-to-hub:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
           with:
             fetch-depth: 0
             lfs: true
             
         - name: Push to Hugging Face Hub
           env:
             HF_TOKEN: ${{ secrets.HF_TOKEN }}
           run: git push --force https://TEN_USER_HF_CUA_BAN:$HF_TOKEN@huggingface.co/spaces/TEN_USER_HF_CUA_BAN/AITutor-API main
   ```
   *Lưu ý thay `TEN_USER_HF_CUA_BAN` và `AITutor-API` bằng thông tin chính xác mà bạn đã đặt ở Giai đoạn 3.*

## Giai đoạn 5: Triển khai & Kiểm tra

1. **Thực thi Deploy:**
   Bạn hãy commit các thay đổi của `Dockerfile` và thư mục `.github` vừa tạo, sau đó thực hiện lệnh:
   ```bash
   git add .
   git commit -m "Configure Docker for HF and Setup CI/CD"
   git push origin main
   ```

2. **Theo dõi tự động quá trình lên sóng:**
   - Qua tab **Actions** trên GitHub, bạn sẽ thấy workflow "Deploy API to Hugging Face Spaces" bắt đầu chạy và đánh dấu tick Xanh sau vài giây.
   - Ngay sau đó, bạn chuyển sang trang **Space của bạn trên Hugging Face**. Bạn sẽ thấy trạng thái đang hiện là **Building** (Chờ khoảng 5-10 phút để Docker tải bản Pytorch và các gói khác).
   - Khi trạng thái chuyển sang **Running** (màu xanh lá) là ứng dụng đã thành công!
   - Bạn có thể nhấn vào nút **App** phía trên cùng hoặc vào `https://ten-user-hf-cua-ban-aitutor-api.hf.space/docs` để kiểm tra trực tiếp giao diện Swagger UI của FastAPI.

> **💡 Lưu ý quan trọng về File Uploads:** 
> Hugging Face Docker Space không lưu trữ vĩnh viễn các file tạo ra bên trong ổ cứng của nó (bị xóa khi server restart). Tuy nhiên, dự án của bạn đã thông minh cấu hình sẵn lưu trữ lên AWS S3 (`S3_ENABLED=true`), vì vậy các file bài kiểm tra hay hình ảnh học sinh upload lên sẽ được lưu giữ an toàn vĩnh viễn trên kho S3. Cơ chế này cực kỳ hoàn hảo cho kiến trúc triển khai Serverless/Container!
