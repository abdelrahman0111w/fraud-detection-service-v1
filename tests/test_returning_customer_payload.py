"""Returning customer: internal_history populated, customer_id required."""
from tests.conftest import EVALUATE_URL


def test_returning_customer_is_low_risk_and_proceeds(client, load_payload):
    resp = client.post(EVALUATE_URL, json=load_payload("sample_returning_customer_payload"))
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["fraud_risk_level"] == "LOW"
    assert body["recommended_action"] == "PROCEED_TO_CREDIT_EVALUATION"
    assert body["triggered_rules"] == []


def test_returning_customer_salary_untouched(client, load_payload):
    body = client.post(EVALUATE_URL, json=load_payload("sample_returning_customer_payload")).json()
    feeder = body["downstream_risk_feeder"]
    assert feeder["declared_salary"] == 24000.0
    assert feeder["risk_adjusted_salary"] == 24000.0


def test_returning_flag_without_customer_id_is_rejected(client, load_payload):
    payload = load_payload("sample_returning_customer_payload")
    payload["internal_history"]["customer_id"] = None
    assert client.post(EVALUATE_URL, json=payload).status_code == 422