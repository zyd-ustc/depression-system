from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = str(Path(__file__).parent.parent.parent)


class AppSettings(BaseSettings):
    # Minimal settings for local inference/UI.
    model_config = SettingsConfigDict(env_file=ROOT_DIR, env_file_encoding="utf-8")

    # LLM
    MODEL_ID: str = "pauliusztin/LLMTwin-Llama-3.1-8B"

    MAX_INPUT_TOKENS: int = 1536
    MAX_TOTAL_TOKENS: int = 2048


settings = AppSettings()
