---
title: AITutor API
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# AITutor API Backend
AITutor là hệ thống gia sư Toán 12 thích nghi, gồm chat/streaming, GraphRAG,
SymPy tool-calling, OCR/VLM, quiz theo cấu trúc THPT, Bayesian Knowledge
Tracing (BKT) và spaced repetition.

## Chạy local

Yêu cầu: Conda, Python 3.12 và Node.js 20+ (Node chỉ cần cho frontend test).

```powershell
conda activate aitutor
Copy-Item .env.example .env
python -m pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000
```

Mở `http://127.0.0.1:8000`. Mặc định ứng dụng dùng SQLite tại
`data/tutor.db`; có thể đặt `DATABASE_URL` sang PostgreSQL/asyncpg.

## Cấu hình

Các biến quan trọng nằm trong `.env.example`:

- `OPENAI_API_KEY` và ba tier model LLM/VLM.
- `DATABASE_URL`, `CHROMA_PERSIST_DIR`.
- `USE_FUNCTION_CALLING` để bật agentic tool-calling cho mọi câu; các bài cần
  tính toán vẫn tự động ép dùng SymPy.
- `OCR_ENGINE=cloud|local`.
- S3 và Langfuse đều tắt mặc định và chỉ hoạt động khi có credentials.

Không commit `.env`, database local, PDF upload, OCR debug hoặc evaluation log.

## Test

```powershell
conda activate aitutor
python -m pytest -q
node --test tests/frontend_chat_visualization.test.cjs
python -m compileall -q app scripts
```

Test DB chạy bằng SQLite in-memory, không cần Docker/PostgreSQL. CI chạy cả
Python và frontend test trước khi đồng bộ lên Hugging Face.

## Dependency profiles

- `requirements.txt`: runtime API.
- `requirements-eval.txt`: runtime + RAGAS/dataset analysis.
- `requirements-dev.txt`: evaluation + toàn bộ test dependencies.

## Docker

```powershell
docker compose up --build
```

API container nghe cổng `7860` và được map ra `localhost:8000`; PostgreSQL nằm
trong service `db`. Dockerfile cũng tương thích Hugging Face Spaces.

## Evaluation

```powershell
conda activate aitutor
python scripts/run_eval_monitor.py --exam de7plus_de01 --questions 1-5 --no-ragas
```

Bỏ `--no-ragas` để chạy judge RAGAS. Các lệnh evaluation/LLM có thể phát sinh
chi phí API; unit test không thực hiện live call.

## Các module chính

- `app/agents`: routing, Teacher/Assessor/Planner, SymPy tools.
- `app/rag`: hybrid retrieval, query rewriting, GraphRAG và reranking.
- `app/quiz`: sinh/chấm quiz, đề thi và adaptive difficulty.
- `app/ocr`, `app/exam_solver`: OCR, trích xuất hình và giải đề.
- `app/knowledge_tracing`, `app/spaced_repetition`: cá nhân hóa học tập.
- `app/evaluation`: RAGAS, MathJudge và báo cáo chất lượng.
- `tests`: unit/integration tests chạy offline.
