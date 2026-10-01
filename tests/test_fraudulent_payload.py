"""
Multi-vector fraud applicant: tampered salary certificate + salary inflated
~4.6x versus real bank inflows (78% mismatch). This is the CRITICAL path --
the highest-value test in the suite, since it proves the service actually
rejects fraud and not just approves clean applicants.
"""
from tests.conftest import EVALUATE_URL


def _evaluate(client, load_payload):
    resp = client.post(EVALUATE_URL, json=load_payload("sample_fraudulent_applicant_payload"))
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_fraud_applicant_is_critical_and_rejected(client, load_payload):
    body = _evaluate(client, load_payload)
    assert body["fraud_risk_level"] == "CRITICAL"
    assert body["recommended_action"] == "REJECT_SUSPECTED_FRAUD"
    assert body["fraud_risk_score"] >= 0.90  # engine floors CRITICAL at 0.90


def test_fraud_applicant_triggers_expected_rules(client, load_payload):
    codes = {r["rule_code"] for r in _evaluate(client, load_payload)["triggered_rules"]}
    assert "DOC-001-TAMPERED-SALARY_CERTIFICATE" in codes
    assert "INC-001-GROSS-INCOME-INFLATION" in codes


def test_fraud_applicant_gets_full_haircut_and_blocked_downstream(client, load_payload):
    feeder = _evaluate(client, load_payload)["downstream_risk_feeder"]
    assert feeder["risk_adjusted_salary"] == 0.0
    assert feeder["haircut_percentage"] == 100.0
    assert feeder["proceed_with_credit_model"] is False


def test_fraud_applicant_has_bilingual_reason_codes(client, load_payload):
    reasons = _evaluate(client, load_payload)["explainable_ai"]["regulatory_reason_codes"]
    assert len(reasons) >= 2
    for r in reasons:
        assert r["reason_en"] and r["reason_ar"]  # CBE audit requirement: both languages


def test_fraud_applicant_ml_layer_agrees_with_rules(client, load_payload):
    metrics = _evaluate(client, load_payload)["metrics"]
    assert metrics["gradient_boost_fraud_probability"] > 0.85
    assert metrics["critical_violations_count"] >= 1