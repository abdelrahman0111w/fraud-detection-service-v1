import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app.config import get_settings
from app.dependencies import get_engine, init_engine
from app.middleware.auth import verify_api_key
from app.middleware.logging import AuditLoggingMiddleware
from app.routers import fraud

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model artifacts + open the SQLite entity store ONCE per worker
    # process, at boot -- not per request, and not lazily on first hit.
    settings = get_settings()
    engine = init_engine()
    if engine.model_manager.is_loaded:
        logger.info("Fraud engine started with production ML artifacts from %s", settings.fraud_artifact_dir)
    else:
        # This is a page-someone situation in production: real applications
        # would be scored by the random-seeded heuristic fallback, not the
        # trained model. See PersistentModelManager._fallback_init().
        logger.critical(
            "Fraud engine started in HEURISTIC FALLBACK mode -- "
            "trained artifacts not found at %s. DO NOT serve production traffic like this.",
            settings.fraud_artifact_dir,
        )
    yield
    # (No explicit teardown needed: sqlite3 connections are opened/closed
    # per-call inside SQLiteEntityStore, nothing to release here.)


settings = get_settings()
app = FastAPI(title=settings.api_title, version=settings.api_version, lifespan=lifespan)

# Order matters: middleware runs outside-in on the way in, inside-out on the
# way out. Registering AuditLoggingMiddleware here means it wraps the whole
# request/response cycle -- including the 401 short-circuit from
# verify_api_key -- so even rejected/unauthorized attempts against the fraud
# endpoint get an audit record (status_code will show 401, no decision fields).
app.add_middleware(AuditLoggingMiddleware)

app.include_router(fraud.router, dependencies=[Depends(verify_api_key)])
app.include_router(fraud.enrich_router, dependencies=[Depends(verify_api_key)])


@app.get("/health", tags=["ops"])
def health(engine=Depends(get_engine)):
    """Never hide model-loading problems behind a generic 200 -- a caller
    (or your uptime monitor) should be able to tell the difference between
    "service is up and scoring with trained artifacts" and "service is up
    but silently running the fallback heuristic"."""
    return {
        "status": "ok",
        "model_status": "production_artifacts" if engine.model_manager.is_loaded else "heuristic_fallback",
        "artifact_dir": get_settings().fraud_artifact_dir,
    }


@app.get("/version", tags=["ops"])
def version():
    return {"api_version": get_settings().api_version}