from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def _default_db_path() -> str:
    if os.getenv("VERCEL"):
        return "/tmp/product_app.sqlite3"
    return str(ROOT_DIR / "data" / "product_app.sqlite3")


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

    MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))


settings = ProductSettings()
