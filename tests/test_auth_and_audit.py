"""API-key auth and the append-only audit trail."""
import json
from types import SimpleNamespace

from tests.conftest import EVALUATE_URL


def test_wrong_api_key_is_rejected(client, load_payload, monkeypatch):
    monkeypatch.setattr("app.middleware.auth.get_settings", lambda: SimpleNamespace(api_key="secret"))
    resp = client.post(EVALUATE_URL, json=load_payload("sample_new_to_bank_payload"),
                       headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_missing_api_key_is_rejected_when_auth_enabled(client, load_payload, monkeypatch):
    monkeypatch.setattr("app.middleware.auth.get_settings", lambda: SimpleNamespace(api_key="secret"))
    assert client.post(EVALUATE_URL, json=load_payload("sample_new_to_bank_payload")).status_code == 401


def test_correct_api_key_is_accepted(client, load_payload, monkeypatch):
    monkeypatch.setattr("app.middleware.auth.get_settings", lambda: SimpleNamespace(api_key="secret"))
    resp = client.post(EVALUATE_URL, json=load_payload("sample_new_to_bank_payload"),
                       headers={"X-API-Key": "secret"})
    assert resp.status_code == 200


def test_health_does_not_require_api_key(client, monkeypatch):
    monkeypatch.setattr("app.middleware.auth.get_settings", lambda: SimpleNamespace(api_key="secret"))
    assert client.get("/health").status_code == 200


def test_every_decision_is_written_to_audit_log(client, load_payload, audit_log_path):
    payload = load_payload("sample_fraudulent_applicant_payload")
    payload["application_id"] = "APP-AUDIT-TEST-001"
    assert client.post(EVALUATE_URL, json=payload).status_code == 200

    with open(audit_log_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    mine = [r for r in records if r.get("application_id") == "APP-AUDIT-TEST-001"]

    assert len(mine) == 1
    rec = mine[0]
    assert rec["status_code"] == 200
    assert rec["fraud_risk_level"] == "CRITICAL"
    assert rec["recommended_action"] == "REJECT_SUSPECTED_FRAUD"
    assert len(rec["request_sha256"]) == 64
    # Raw PII must NOT be copied into the audit log -- only its hash.
    assert "28511201234567" not in json.dumps(rec)