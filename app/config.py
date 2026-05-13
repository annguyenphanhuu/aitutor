"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    """App settings from .env file."""

    # LLM Models — 3 tiers
    OPENAI_API_KEY: str = ""
    # Tier 1 — Full reasoning: Teacher, Reflection-Correct
    LLM_MODEL: str = "gpt-5.4"
    # Tier 2 — Balanced: Classifier, Planner, Assessor, Reflection-Extract, OCR
    LLM_MODEL_MINI: str = "gpt-5.4-mini-2026-03-17"
    # Tier 3 — Fast & cheap: ExprExtractor, simple utilities
    LLM_MODEL_NANO: str = "gpt-5.4-nano-2026-03-17"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    VISION_LLM_MODEL: str = "gpt-5.4"  # Vision-capable model for image questions

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/tutor.db"

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"

    # App
    APP_TITLE: str = "AI Tutor - Gia sư Toán 12"
    DEBUG: bool = True

    # ── Re-ranking ─────────────────────────────────────────────────────────
    RERANKER_ENABLED: bool = True
    # "local" → cross-encoder/ms-marco-MiniLM-L-6-v2 (offline, ~68MB)
    # "cohere" → Cohere Rerank API (requires COHERE_API_KEY)
    # "disabled" → fallback to hybrid score only
    RERANKER_MODEL: str = "local"
    COHERE_API_KEY: str = ""
    RERANKER_TOP_K: int = 5           # final docs returned to Teacher
    RERANKER_CANDIDATE_K: int = 20   # pool size for re-ranking

    # ── Function Calling (ReAct agentic tool-use) ─────────────────────────
    USE_FUNCTION_CALLING: bool = False   # Bật qua .env sau khi A/B test
    FUNCTION_CALLING_MAX_ROUNDS: int = 5  # Giới hạn vòng lặp tool-call

    # ── Guardrails / Safety ───────────────────────────────────────────────
    GUARDRAILS_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 30
    RATE_LIMIT_PER_HOUR: int = 200
    MAX_MESSAGE_LENGTH: int = 4000   # chars — truncate nếu vượt quá

    # AWS S3
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "ap-southeast-1"
    S3_BUCKET: str = "aitutor-bucket-vn"
    S3_ENABLED: bool = False   # Bật qua .env: S3_ENABLED=true

    # ── Exam Solver (OCR + Per-Question Pipeline) ───────────────────────
    EXAM_SOLVER_MODEL: str = "o4-mini-2025-04-16"
    OCR_ENGINE: str = "cloud"        # "cloud" (GPT Vision) | "local" (GOT-OCR2.0)
    MAX_EXAM_PAGES: int = 20         # Max PDF pages to scan for questions (solution detection handles the real cutoff)
    SOLVE_CONCURRENCY: int = 5       # Parallel question solving (semaphore)

    # ── Langfuse Observability ────────────────────────────────────────────
    LANGFUSE_ENABLED: bool = False
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Allow extra vars in .env (docker, infra, etc.)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
