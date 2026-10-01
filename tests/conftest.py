"""
Shared pytest setup.

The env vars below MUST be set before `app.main` is imported, because
get_settings() is cached on first call. They point the engine's SQLite
velocity store and the audit log at a throwaway temp folder, so running the
tests never pollutes your real fraud_registry.db / audit_log.jsonl.
"""
import copy
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

_TMP = tempfile.mkdtemp(prefix="fraud_tests_")
os.environ["FRAUD_DB_PATH"] = os.path.join(_TMP, "fraud_registry_test.db")
os.environ["FRAUD_AUDIT_LOG_PATH"] = os.path.join(_TMP, "audit_log_test.jsonl")
os.environ.pop("FRAUD_API_KEY", None)  # auth off by default; test_auth cases patch it explicitly

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

FIXTURES_DIR = Path(__file__).parent / "fixtures"
EVALUATE_URL = "/api/v1/fraud/evaluate"


@pytest.fixture(autouse=True)
def _auth_off_by_default(monkeypatch):
    """Settings also reads a local .env file (which holds your real FRAUD_API_KEY),
    and popping the env var above does not stop that. Force auth OFF for every test
    so results never depend on what is in your .env. The auth tests in
    test_auth_and_audit.py patch it back ON themselves; their patch runs after
    this one, so it wins."""
    monkeypatch.setattr("app.middleware.auth.get_settings", lambda: SimpleNamespace(api_key=None))


@pytest.fixture(scope="session")
def client():
    # `with` runs the FastAPI lifespan, which loads the model artifacts once.
    with TestClient(app) as c:
        yield c


@pytest.fixture
def load_payload():
    """Returns a fresh deep copy each call, so tests can mutate freely."""
    def _load(name: str) -> dict:
        with open(FIXTURES_DIR / f"{name}.json", encoding="utf-8") as f:
            return copy.deepcopy(json.load(f))
    return _load


@pytest.fixture
def audit_log_path():
    return os.environ["FRAUD_AUDIT_LOG_PATH"]