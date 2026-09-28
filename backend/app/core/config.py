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

    # SQLite file. Kept as a single file for trivial backup. Left unset by
    # default so it's derived from storage_dir (see get_settings) — that
    # way overriding ROCKSEG_STORAGE_DIR alone (e.g. a docker volume mount)
    # can't silently point the two at different, half-created directories.
    # Set ROCKSEG_DATABASE_URL explicitly only to move the DB elsewhere.
    database_url: str | None = None

    cors_origins: list[str] = ["http://localhost:5173"]

    # Signs the session JWT. The default is for local dev only — set
    # ROCKSEG_SECRET_KEY to a long random value in any shared deployment.
    secret_key: str = "dev-insecure-secret-change-me"
    token_ttl_minutes: int = 60 * 24 * 7
    # Set true when served over HTTPS so the session cookie is Secure.
    cookie_secure: bool = False


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    if settings.database_url is None:
        db_path = (settings.storage_dir / "db.sqlite3").resolve()
        settings.database_url = f"sqlite:///{db_path}"
    return settings
