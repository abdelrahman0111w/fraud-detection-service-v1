"""
Append-only audit logging for every call to the fraud router.

Why middleware and not just logging inside the router function: middleware
sits outside the response_model validation and business logic, so it logs
exactly what went out over the wire (status code included) and can't be
skipped by a code path inside the endpoint that returns early or raises.

Each record is one JSON line containing:
  - a SHA-256 hash of the raw request body (not the raw PII itself -- the
    hash lets you prove "this exact payload produced this exact decision"
    without duplicating sensitive applicant data into a second log store)
  - application_id, pulled from the request body for easy lookup
  - the decision actually returned (risk level, score, action)
  - status code and latency

This is a minimal, dependency-free implementation (plain file appends).
Before real production volume, swap AUDIT_LOG_PATH for a proper sink (a
database table, or shipping these lines to your centralized log platform) --
a local file has no retention/rotation/backup story on its own.
"""
import hashlib
import json
import logging
import time
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config import get_settings

logger = logging.getLogger("fraud_audit")


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    AUDITED_PATH_PREFIXES = ("/api/v1/fraud/", "/api/v1/application/")

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith(self.AUDITED_PATH_PREFIXES):
            return await call_next(request)

        request_body = await request.body()
        request_hash = hashlib.sha256(request_body).hexdigest()

        application_id = None
        try:
            application_id = json.loads(request_body or b"{}").get("application_id")
        except Exception:
            pass  # malformed JSON will be caught by schema validation downstream; just skip the ID here

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        # Response bodies are single-use ASGI streams -- buffer it fully so we
        # can both log it AND still return it intact to the real caller.
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "application_id": application_id,
            "request_sha256": request_hash,
        }

        if response.status_code == 200:
            try:
                body_json = json.loads(response_body)
                # /evaluate returns the assessment at the top level; /enrich
                # nests it under "fraud_assessment" alongside the echoed
                # application. Check both shapes so both endpoints get a
                # complete audit record rather than one of them logging nulls.
                assessment = body_json.get("fraud_assessment", body_json)
                record["fraud_risk_level"] = assessment.get("fraud_risk_level")
                record["fraud_risk_score"] = assessment.get("fraud_risk_score")
                record["recommended_action"] = assessment.get("recommended_action")
            except Exception:
                logger.warning("Could not parse response body for audit record (application_id=%s)", application_id)

        self._write_record(record)

        headers = dict(response.headers)
        headers.pop("content-length", None)  # let Starlette recompute this for the rebuilt body
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
        )

    @staticmethod
    def _write_record(record: dict) -> None:
        try:
            with open(get_settings().audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            # Never let an audit-log write failure take down the actual API response --
            # log the failure itself (to stdout/stderr, not the file that just failed)
            # so it's visible to ops, but don't raise.
            logger.exception("Failed to write audit log record: %s", record)