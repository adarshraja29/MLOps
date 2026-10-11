"""Application configuration and memory threshold management."""

from functools import lru_cache
from typing import Optional
import os

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    # Fallback to pydantic v1 / v2 direct import
    from pydantic import BaseSettings, Field  # type: ignore


class Settings(BaseSettings):
    """Central configuration for academic summarization backend."""

    # Application Metadata
    APP_NAME: str = "Computational Linguistics Academic Summarizer"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="production", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")

    # Host & Port
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")

    # Hardware & Memory Constraints (Strict CPU-only enforcement)
    DEVICE: str = Field(default="cpu", env="DEVICE")
    FORCE_CPU: bool = Field(default=True, env="FORCE_CPU")
    MAX_WORKERS: int = Field(default=1, env="MAX_WORKERS")  # Single worker to prevent multiple model instances in RAM

    # Model Identifiers
    SCIBERT_MODEL_NAME: str = Field(
        default="allenai/scibert_scivocab_uncased",
        env="SCIBERT_MODEL_NAME",
        description="Pretrained model for scientific text tokenization and semantic chunking"
    )
    LONGT5_MODEL_NAME: str = Field(
        default="google/long-t5-tglobal-base",
        env="LONGT5_MODEL_NAME",
        description="Base LongT5 model optimized for long-document summarization"
    )

    # Token and Character Safety Limits (Mitigate OOM crashes)
    MIN_INPUT_CHARS: int = Field(default=50, env="MIN_INPUT_CHARS")
    MAX_INPUT_CHARS: int = Field(default=50000, env="MAX_INPUT_CHARS")
    MAX_INPUT_TOKENS: int = Field(default=2048, env="MAX_INPUT_TOKENS")
    MAX_SUMMARY_TOKENS: int = Field(default=256, env="MAX_SUMMARY_TOKENS")
    MIN_SUMMARY_TOKENS: int = Field(default=40, env="MIN_SUMMARY_TOKENS")

    # Inference Memory Tuning
    NUM_BEAMS: int = Field(default=2, env="NUM_BEAMS", description="Low beam search count reduces decoding memory overhead")
    NO_REPEAT_NGRAM_SIZE: int = Field(default=3, env="NO_REPEAT_NGRAM_SIZE")
    MEMORY_THRESHOLD_MB: float = Field(default=3800.0, env="MEMORY_THRESHOLD_MB", description="2.5 GB warning threshold")

    # Mock Mode for rapid local testing / low RAM CI
    USE_MOCK_MODEL: bool = Field(
        default=False,
        env="USE_MOCK_MODEL",
        description="If True, uses a deterministic lightweight mock summarizer without downloading 1GB+ model weights"
    )

    # SQLite Cache Settings
    CACHE_DB_PATH: str = Field(
        default="summaries.db",
        env="CACHE_DB_PATH",
        description="Path to SQLite zero-compute hash cache database"
    )

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "*"
    ]

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Retrieve cached application settings instance."""
    return Settings()
