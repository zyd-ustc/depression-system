from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def _default_db_path() -> str:
    if os.getenv("VERCEL"):
        return "/tmp/product_app.sqlite3"
    return str(ROOT_DIR / "data" / "product_app.sqlite3")


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


class ProductSettings:
    """Environment-backed settings for the P0 product app."""

    APP_NAME = "Depression Support Assistant"
    APP_SECRET = os.getenv("APP_SECRET", "vercel-demo-change-me")
    CONSENT_VERSION = os.getenv("CONSENT_VERSION", "p0-placeholder")
    DATABASE_URL = os.getenv("PRODUCT_DB_PATH", _default_db_path())

    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    DEEPSEEK_TIMEOUT_SECONDS = float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "30"))

    MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "80"))

    MINI_RAG_ENABLED = _env_bool("MINI_RAG_ENABLED", "1")
    MINI_RAG_DB_PATH = os.getenv("MINI_RAG_DB_PATH", str(ROOT_DIR / "data" / "knowledge_index.db"))
    MINI_RAG_TOP_K = int(os.getenv("MINI_RAG_TOP_K", "4"))
    MINI_RAG_CANDIDATE_LIMIT = int(os.getenv("MINI_RAG_CANDIDATE_LIMIT", "10"))
    MINI_RAG_MAX_CHARS = int(os.getenv("MINI_RAG_MAX_CHARS", "2400"))
    MINI_RAG_ENABLE_EMBEDDING = _env_bool("MINI_RAG_ENABLE_EMBEDDING", "0")
    MINI_RAG_ENABLE_RERANK = _env_bool("MINI_RAG_ENABLE_RERANK", "0")


settings = ProductSettings()
