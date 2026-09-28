from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from ..core.config import get_settings
from .models import Base

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
        _engine = create_engine(settings.database_url, connect_args=connect_args)
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    return _SessionLocal


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    # create_all never alters existing tables: add the ownership column to
    # databases created before auth existed.
    if "owner_id" not in {c["name"] for c in inspect(engine).get_columns("projects")}:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE projects ADD COLUMN owner_id VARCHAR REFERENCES users(id)"))


def get_db() -> Iterator[Session]:
    session_factory = get_session_factory()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
