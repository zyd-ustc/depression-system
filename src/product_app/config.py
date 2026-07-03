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


def _env_csv(name: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


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

    RAGFLOW_ENABLED = _env_bool("RAGFLOW_ENABLED")
    RAGFLOW_BASE_URL = os.getenv("RAGFLOW_BASE_URL", "http://127.0.0.1:9380").rstrip("/")
    RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY", "")
    RAGFLOW_DATASET_IDS = _env_csv("RAGFLOW_DATASET_IDS")
    RAGFLOW_TOP_K = int(os.getenv("RAGFLOW_TOP_K", "5"))
    RAGFLOW_CANDIDATE_TOP_K = int(os.getenv("RAGFLOW_CANDIDATE_TOP_K", "1024"))
    RAGFLOW_SIMILARITY_THRESHOLD = float(os.getenv("RAGFLOW_SIMILARITY_THRESHOLD", "0.25"))
    RAGFLOW_VECTOR_SIMILARITY_WEIGHT = float(os.getenv("RAGFLOW_VECTOR_SIMILARITY_WEIGHT", "0.3"))
    RAGFLOW_TIMEOUT_SECONDS = float(os.getenv("RAGFLOW_TIMEOUT_SECONDS", "4"))
    RAGFLOW_MAX_CONTEXT_CHARS = int(os.getenv("RAGFLOW_MAX_CONTEXT_CHARS", "2400"))


settings = ProductSettings()
