"""Makes the legacy CLI importable as a plain Python package for the parity
test, without touching its code (it uses bare imports like
``from rock_model import RockNetModel``, so it needs its own directory on
``sys.path``, exactly as if it were being run as a script from inside
``legacy/rock-nn/``)."""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
# ROCKSEG_LEGACY_DIR points the parity tests at another checkout of the
# legacy CLI (e.g. an export of a given commit) while legacy/ is being edited.
LEGACY_ROCKNN_DIR = Path(os.environ.get("ROCKSEG_LEGACY_DIR") or REPO_ROOT / "legacy" / "rock-nn")

if str(LEGACY_ROCKNN_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROCKNN_DIR))


@pytest.fixture
def app_factory(tmp_path, monkeypatch):
    """Builds independent TestClients (each with its own cookie jar) against
    one throwaway storage dir + SQLite file, isolated per test (the app
    otherwise memoizes settings/engine as module globals)."""
    from fastapi.testclient import TestClient

    from app.core import config as config_module
    from app.db import session as session_module

    monkeypatch.setenv("ROCKSEG_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("ROCKSEG_DATABASE_URL", f"sqlite:///{tmp_path / 'db.sqlite3'}")
    config_module.get_settings.cache_clear()
    session_module._engine = None
    session_module._SessionLocal = None

    from app.main import app

    clients = []

    def make() -> TestClient:
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield make

    for client in clients:
        client.__exit__(None, None, None)
    config_module.get_settings.cache_clear()
    session_module._engine = None
    session_module._SessionLocal = None


def register(client, email="ana@example.com", name="Ana", password="senha-forte-123"):
    """Accounts are admin-created (no sign-up endpoint): create the user with
    the same script an admin would use, then log the client in."""
    from app.create_user import main as create_user

    assert create_user([email, "--name", name, "--password", password]) == 0
    resp = client.post("/auth/login", json={"email": email.strip().lower(), "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.fixture
def anon_client(app_factory):
    """No session cookie."""
    return app_factory()


@pytest.fixture
def api_client(app_factory):
    """Registered and logged in as a default user."""
    client = app_factory()
    register(client)
    return client
