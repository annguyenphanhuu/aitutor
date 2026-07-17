"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

from app.config import get_settings
from app.db.database import init_db
from app.api.routes import router
import logging

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
# Silence noisy third-party loggers
for _noisy in ("httpx", "httpcore", "openai", "chromadb", "urllib3",
               "sqlalchemy.engine", "sqlalchemy.pool", "sqlalchemy.dialects"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

settings = get_settings()

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Create data directory
    os.makedirs("data", exist_ok=True)

    # Initialize database tables
    await init_db()
    print("[OK] Database initialized")

    # ── Langfuse startup check ────────────────────────────────────────
    from app.utils.langfuse_client import langfuse_auth_check
    langfuse_auth_check()

    yield

    from app.utils.cost_tracker import session_summary
    # Flush Langfuse trước khi shutdown
    from app.utils.langfuse_client import get_langfuse
    lf = get_langfuse()
    if lf:
        lf.flush()
    print(f"\n[BYE] {session_summary()}")
    print("[BYE] Shutting down")


app = FastAPI(
    title=settings.APP_TITLE,
    description="Gia sư AI thích nghi cho Toán lớp 12",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — app dùng header X-User-Id (không cookie) nên không cần credentials.
# Spec CORS không cho phép wildcard origin đi kèm allow_credentials=True.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router)

# Serve static files
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Only expose exam images, never databases, logs, or evaluation artifacts.
EXAM_IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "exams", "images"
)
if os.path.exists(EXAM_IMAGES_DIR):
    app.mount(
        "/data/exams/images",
        StaticFiles(directory=EXAM_IMAGES_DIR),
        name="exam-images",
    )


@app.get("/")
async def root():
    """Serve the main HTML page."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": settings.APP_TITLE, "docs": "/docs"}


@app.get("/dashboard")
async def dashboard_page():
    """Serve the dashboard page."""
    path = os.path.join(STATIC_DIR, "dashboard.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "Dashboard page not found"}


if __name__ == "__main__":
    import uvicorn
    # Windows: auto-reload gây crash do multiprocessing + encoding cp1258 (tiếng Việt)
    # Dùng: uvicorn app.main:app --reload  nếu muốn hot-reload khi dev
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
