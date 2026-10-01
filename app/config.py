"""
Central configuration for the Fraud Detection Service.
All values are overridable via environment variables (or a .env file),
so Docker/CI/local runs never need code changes.
"""
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Where PersistentModelManager looks for the 4 joblib/json artifacts.
    # Must contain: isolation_forest_v2.joblib, fraud_gradient_boost_v2.joblib,
    # scaler_v2.joblib, mahalanobis_precision_v2.joblib (optional), feature_names.json, metrics.json
    fraud_artifact_dir: str = "model/artifacts/fraud"

    # SQLite entity-velocity store. Put this on a mounted/persistent volume in Docker,
    # and see the note in README about replacing with Postgres/Redis once you run
    # more than one API replica (each replica would otherwise get its own file and
    # cross-application collision detection stops working across instances).
    fraud_db_path: str = "fraud_registry.db"

    api_title: str = "Fraud Detection Service"
    api_version: str = "v1"

    # Append-only audit trail written by app/middleware/logging.py -- one JSON
    # line per fraud evaluation, containing a hash of the raw input and the
    # decision that was returned. Put this on a persistent volume in Docker,
    # same as fraud_db_path, and never delete/rotate-away old lines -- this is
    # the record you'd hand an auditor asking "why was this application
    # rejected on <date>?".
    # Alias needed: without it pydantic-settings would read AUDIT_LOG_PATH, not FRAUD_AUDIT_LOG_PATH
    # (the name used by the tests, the Dockerfile and docker-compose.yml).
    audit_log_path: str = Field(default="audit_log.jsonl", validation_alias="FRAUD_AUDIT_LOG_PATH")

    # Simple shared-secret API key check (swap for OAuth2/JWT later if needed).
    # Env var is FRAUD_API_KEY (the alias); without it pydantic-settings would read plain API_KEY.
    api_key: str | None = Field(default=None, validation_alias="FRAUD_API_KEY")
    api_key_header_name: str = "X-API-Key"


@lru_cache
def get_settings() -> Settings:
    return Settings()