"""
Singleton access to CreditFraudEngine.

We use module-level state populated at FastAPI startup (see app/main.py's
lifespan handler) rather than lazy-loading on first request, so a broken
model artifact fails loudly at container boot -- not silently on whichever
customer's request happens to hit an unwarmed worker first.
"""
from typing import Optional

from fraud_engine import CreditFraudEngine

from app.config import get_settings

_engine: Optional[CreditFraudEngine] = None


def init_engine() -> CreditFraudEngine:
    """Called once from the FastAPI lifespan startup hook."""
    global _engine
    settings = get_settings()
    _engine = CreditFraudEngine(
        db_path=settings.fraud_db_path,
        artifact_dir=settings.fraud_artifact_dir,
    )
    return _engine


def get_engine() -> CreditFraudEngine:
    """FastAPI dependency. Raises if init_engine() wasn't called at startup --
    that's intentional: it means the app is misconfigured, and should fail
    fast rather than silently constructing a fresh (possibly fallback-mode)
    engine per request."""
    if _engine is None:
        raise RuntimeError(
            "Fraud engine not initialized. init_engine() must run in the "
            "FastAPI lifespan startup handler before serving requests."
        )
    return _engine