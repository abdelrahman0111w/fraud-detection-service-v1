"""
Shared-secret API key check for the fraud router.

This is intentionally simple (a single static key compared via a header) --
it's meant to keep this endpoint from being wide open on day one, not to be
your final auth story. Before exposing this beyond trusted internal callers,
replace it with OAuth2 client-credentials or mTLS, whichever your bank's
integration standards require.
"""
from fastapi import Header, HTTPException, status

from app.config import get_settings


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """
    FastAPI dependency -- raise 401 on mismatch, otherwise return None (allow).

    NOTE: if settings.api_key is unset (None), this check is a no-op and every
    request is allowed through. That's convenient for local dev but means an
    unset FRAUD_API_KEY env var in production silently disables auth entirely.
    Consider making `api_key` a required setting (no default) once you deploy
    for real, so a missing env var fails startup instead of failing open.
    """
    settings = get_settings()
    if settings.api_key is None:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )