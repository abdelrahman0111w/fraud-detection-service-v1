"""Tests for POST /api/v1/application/enrich -- returns the original payload
with a fraud_assessment block merged in, unlike /evaluate which returns only
the assessment."""
from tests.conftest import FIXTURES_DIR
import json

ENRICH_URL = "/api/v1/application/enrich"


def test_enrich_returns_original_fields_plus_assessment(client, load_payload):
    payload = load_payload("sample_fraudulent_applicant_payload")
    resp = client.post(ENRICH_URL, json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Original payload fields are echoed back.
    assert body["application_id"] == payload["application_id"]
    assert body["is_returning_customer"] == payload["is_returning_customer"]

    # New field: the full fraud assessment, nested.
    assert "fraud_assessment" in body
    assert body["fraud_assessment"]["fraud_risk_level"] == "CRITICAL"
    assert body["fraud_assessment"]["recommended_action"] == "REJECT_SUSPECTED_FRAUD"


def test_enrich_updates_consistency_checks_with_verification_results(client, load_payload):
    payload = load_payload("sample_fraudulent_applicant_payload")
    body = client.post(ENRICH_URL, json=payload).json()
    # engine.enrich_payload() merges verification_checklist into consistency_checks
    assert body["consistency_checks"]["employer_verified"] is False


def test_enrich_requires_api_key_like_evaluate(client, load_payload, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("app.middleware.auth.get_settings", lambda: SimpleNamespace(api_key="secret"))
    payload = load_payload("sample_new_to_bank_payload")
    assert client.post(ENRICH_URL, json=payload).status_code == 401


def test_enrich_is_also_audit_logged(client, load_payload, audit_log_path):
    payload = load_payload("sample_new_to_bank_payload")
    payload["application_id"] = "APP-ENRICH-AUDIT-TEST"
    assert client.post(ENRICH_URL, json=payload).status_code == 200

    with open(audit_log_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    mine = [r for r in records if r.get("application_id") == "APP-ENRICH-AUDIT-TEST"]
    assert len(mine) == 1
    assert mine[0]["fraud_risk_level"] == "LOW"
    assert mine[0]["path"] == ENRICH_URL