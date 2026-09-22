"""Application configuration, read from environment variables (with sane
defaults for local `docker compose up`)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ROCKSEG_")

    # Root directory for all project storage: images, masks, datasets, models, runs.
    storage_dir: Path = Path("./storage")

    # SQLite file. Kept as a single file for trivial backup.
    database_url: str = "sqlite:///./storage/db.sqlite3"

    cors_origins: list[str] = ["http://localhost:5173"]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return settings
