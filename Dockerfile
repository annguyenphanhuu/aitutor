# Sử dụng Python 3.12 bản nhẹ (slim) làm base image
FROM python:3.12-slim

# Thiết lập biến môi trường
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy file requirements trước để tận dụng cơ chế cache của Docker layer
COPY requirements.txt .

# Cài đặt PyTorch phiên bản CPU trước để tránh tải bản CUDA 2.5GB siêu nặng
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các dependencies còn lại
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn dự án vào container
COPY . .

# Tạo thư mục data nếu chưa có
RUN mkdir -p data

# Expose port mà FastAPI sẽ chạy (Hugging Face yêu cầu port 7860)
EXPOSE 7860

# Lệnh khởi chạy server bằng Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
