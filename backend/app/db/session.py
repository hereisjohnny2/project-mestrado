from __future__ import annotations

import json
from collections.abc import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from ..core.config import get_settings
from .models import Base, default_classes

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


def _columns(engine, table: str) -> set[str]:
    return {c["name"] for c in inspect(engine).get_columns(table)}


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    # create_all never alters existing tables, so columns added after a
    # database was created are patched in here, with their legacy values.
    with engine.begin() as conn:
        if "owner_id" not in _columns(engine, "projects"):
            conn.execute(text("ALTER TABLE projects ADD COLUMN owner_id VARCHAR REFERENCES users(id)"))
        if "classes" not in _columns(engine, "projects"):
            conn.execute(text("ALTER TABLE projects ADD COLUMN classes JSON"))
            conn.execute(text("UPDATE projects SET classes = :c"), {"c": json.dumps(default_classes())})
        if "classes" not in _columns(engine, "datasets"):
            conn.execute(text("ALTER TABLE datasets ADD COLUMN classes JSON"))
            conn.execute(text("ALTER TABLE datasets ADD COLUMN class_counts JSON"))
            conn.execute(text("UPDATE datasets SET classes = :c"), {"c": json.dumps(default_classes())})
            for row in conn.execute(text("SELECT id, n_pore, n_solid FROM datasets")).all():
                counts = {"Poro": row.n_pore, "Sólido": row.n_solid}
                conn.execute(text("UPDATE datasets SET class_counts = :cc WHERE id = :id"), {"cc": json.dumps(counts), "id": row.id})
            # The old NOT NULL counters would reject new rows (the ORM no
            # longer fills them); SQLite >= 3.35 can drop them in place.
            conn.execute(text("ALTER TABLE datasets DROP COLUMN n_pore"))
            conn.execute(text("ALTER TABLE datasets DROP COLUMN n_solid"))
        if "architecture" not in _columns(engine, "models"):
            conn.execute(text("ALTER TABLE models ADD COLUMN architecture VARCHAR NOT NULL DEFAULT '1.0.0'"))
        if "class_fractions" not in _columns(engine, "results"):
            conn.execute(text("ALTER TABLE results ADD COLUMN class_fractions JSON"))
            for row in conn.execute(text("SELECT id, porosity FROM results")).all():
                fractions = {"Poro": row.porosity, "Sólido": 1.0 - row.porosity}
                conn.execute(text("UPDATE results SET class_fractions = :f WHERE id = :id"), {"f": json.dumps(fractions), "id": row.id})


def get_db() -> Iterator[Session]:
    session_factory = get_session_factory()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
