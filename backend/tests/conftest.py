"""Makes the legacy CLI importable as a plain Python package for the parity
test, without touching its code (it uses bare imports like
``from rock_model import RockNetModel``, so it needs its own directory on
``sys.path``, exactly as if it were being run as a script from inside
``legacy/rock-nn/``)."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROCKNN_DIR = REPO_ROOT / "legacy" / "rock-nn"

if str(LEGACY_ROCKNN_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROCKNN_DIR))


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """A TestClient wired to a throwaway storage dir + SQLite file, isolated
    per test (the app otherwise memoizes settings/engine as module globals)."""
    from fastapi.testclient import TestClient

    from app.core import config as config_module
    from app.db import session as session_module

    monkeypatch.setenv("ROCKSEG_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("ROCKSEG_DATABASE_URL", f"sqlite:///{tmp_path / 'db.sqlite3'}")
    config_module.get_settings.cache_clear()
    session_module._engine = None
    session_module._SessionLocal = None

    from app.main import app

    with TestClient(app) as client:
        yield client

    config_module.get_settings.cache_clear()
    session_module._engine = None
    session_module._SessionLocal = None
