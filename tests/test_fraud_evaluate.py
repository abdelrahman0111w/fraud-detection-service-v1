from tests.conftest import EVALUATE_URL


def test_health_reports_production_artifacts(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_status"] == "production_artifacts"


def test_version_endpoint(client):
    assert client.get("/version").status_code == 200


def test_evaluate_returns_full_assessment_shape(client, load_payload):
    resp = client.post(EVALUATE_URL, json=load_payload("sample_new_to_bank_payload"))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for key in ("fraud_risk_score", "fraud_risk_level", "recommended_action", "action_ar",
                "verification_checklist", "behavioral_anomalies", "metrics",
                "downstream_risk_feeder", "explainable_ai", "triggered_rules"):
        assert key in body, f"missing key in response: {key}"
    assert body["metrics"]["model_engine_status"] == "ONLINE_CALIBRATED_ARTIFACTS"