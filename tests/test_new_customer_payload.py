"""New-to-bank (cold start) applicant: internal_history_missing=1, clean profile."""
from tests.conftest import EVALUATE_URL


def test_new_customer_is_low_risk_and_proceeds(client, load_payload):
    resp = client.post(EVALUATE_URL, json=load_payload("sample_new_to_bank_payload"))
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["fraud_risk_level"] == "LOW"
    assert body["recommended_action"] == "PROCEED_TO_CREDIT_EVALUATION"
    assert body["fraud_risk_score"] < 0.28  # below the MEDIUM threshold
    assert body["triggered_rules"] == []


def test_new_customer_gets_no_income_haircut(client, load_payload):
    body = client.post(EVALUATE_URL, json=load_payload("sample_new_to_bank_payload")).json()
    feeder = body["downstream_risk_feeder"]

    assert feeder["declared_salary"] == 18000.0
    assert feeder["risk_adjusted_salary"] == 18000.0
    assert feeder["haircut_percentage"] == 0.0
    assert feeder["proceed_with_credit_model"] is True


def test_new_customer_without_internal_history_block_still_works(client, load_payload):
    """Cold-start defaults must let callers omit internal_history entirely."""
    payload = load_payload("sample_new_to_bank_payload")
    del payload["internal_history"]
    resp = client.post(EVALUATE_URL, json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json()["fraud_risk_level"] == "LOW"