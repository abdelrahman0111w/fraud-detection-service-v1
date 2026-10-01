"""
Engine-level golden tests (no HTTP). These freeze the current behaviour of
fraud_engine.py + the trained artifacts. If one fails after you retrain a
model, edit a threshold or bump scikit-learn, that is the point: a decision
changed, and a human should confirm the change is intended before deploying.

Score tolerances are deliberately loose (abs=0.03) so harmless float drift
between scikit-learn patch versions doesn't cause false alarms, while any
real behaviour change still trips the test.
"""
import copy
import json
import warnings
from pathlib import Path

import pytest

from fraud_engine import CreditFraudEngine

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"

# name -> (level, action, approx score)
GOLDEN = {
    "sample_new_to_bank_payload": ("LOW", "PROCEED_TO_CREDIT_EVALUATION", 0.039),
    "sample_returning_customer_payload": ("LOW", "PROCEED_TO_CREDIT_EVALUATION", 0.041),
    "sample_fraudulent_applicant_payload": ("CRITICAL", "REJECT_SUSPECTED_FRAUD", 1.0),
}


def _payload(name):
    with open(FIXTURES / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    db = tmp_path_factory.mktemp("engine") / "registry.db"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn version-mismatch unpickle warnings
        eng = CreditFraudEngine(db_path=str(db), artifact_dir=str(ROOT / "model" / "artifacts" / "fraud"))
    assert eng.model_manager.is_loaded, "Artifacts not found in model/artifacts/fraud/ -- engine is in fallback mode"
    return eng


@pytest.mark.parametrize("name", list(GOLDEN))
def test_golden_decisions(engine, name):
    level, action, score = GOLDEN[name]
    result = engine.evaluate(_payload(name))
    assert result["fraud_risk_level"] == level
    assert result["recommended_action"] == action
    assert result["fraud_risk_score"] == pytest.approx(score, abs=0.03)


def test_engine_is_deterministic(engine):
    p = _payload("sample_new_to_bank_payload")
    a = engine.evaluate(copy.deepcopy(p))
    b = engine.evaluate(copy.deepcopy(p))
    assert a["fraud_risk_score"] == b["fraud_risk_score"]
    assert a["fraud_risk_level"] == b["fraud_risk_level"]


def test_higher_income_inflation_never_lowers_fraud_score(engine):
    """Business-logic guard mirroring the monotonic constraints: inflating the
    declared salary relative to bank inflows must not make the applicant look safer."""
    base = _payload("sample_new_to_bank_payload")
    scores = []
    for declared in (18000.0, 24000.0, 30000.0, 45000.0):
        p = copy.deepcopy(base)
        p["salary_certificate_fields"]["declared_net_salary"]["value"] = declared
        scores.append(engine.evaluate(p)["fraud_risk_score"])
    assert scores == sorted(scores), f"score decreased as income inflation grew: {scores}"


def test_invalid_national_id_is_critical(engine):
    p = _payload("sample_new_to_bank_payload")
    p["national_id_fields"]["national_id"]["value"] = "12345"
    r = engine.evaluate(p)
    assert r["fraud_risk_level"] == "CRITICAL"
    assert any(v["rule_code"] == "ID-001-INVALID-NID-FORMAT" for v in r["triggered_rules"])


def test_underage_applicant_flagged(engine):
    p = _payload("sample_new_to_bank_payload")
    p["national_id_fields"]["age_years"]["value"] = 19.0
    r = engine.evaluate(p)
    assert any(v["rule_code"] == "ID-002-AGE-POLICY-BREACH" for v in r["triggered_rules"])


def test_vpn_and_rooted_device_escalate_risk(engine):
    p = _payload("sample_new_to_bank_payload")
    p["device_telemetry"] = {"is_vpn_or_proxy": True, "device_is_rooted_or_emulator": True,
                             "submission_hour_local": 3}
    r = engine.evaluate(p)
    codes = {v["rule_code"] for v in r["triggered_rules"]}
    assert {"SEC-001-VPN-OR-PROXY-DETECTED", "SEC-002-COMPROMISED-DEVICE-ENVIRONMENT",
            "SEC-003-ABNORMAL-OFF-HOURS-SUBMISSION"} <= codes
    assert r["fraud_risk_level"] == "CRITICAL"


def test_phone_reuse_across_identities_triggers_ring_detection(engine):
    """Layer 3: same phone used by 3 different applications within 48h."""
    for i in range(3):
        p = _payload("sample_new_to_bank_payload")
        p["application_id"] = f"APP-RING-{i}"
        p["form_data"]["mobile_phone"] = "01000000001"
        r = engine.evaluate(p)
    codes = {v["rule_code"] for v in r["triggered_rules"]}
    assert "VEL-001-CROSS-APP-PHONE-COLLISION" in codes
    assert r["fraud_risk_level"] == "CRITICAL"


def test_employer_mismatch_from_consistency_checks_fallback_is_caught(engine):
    """Regression test: previously the engine defaulted to similarity=1.0 and
    employer_verified=True whenever bank_statement_fields lacked
    payroll_transfer_employer -- which is every payload under this contract.
    It must now fall back to the caller's own consistency_checks."""
    p = _payload("sample_fraudulent_applicant_payload")
    r = engine.evaluate(p)
    assert r["verification_checklist"]["employer_verified"] is False
    assert r["metrics"]["employer_similarity_score"] == pytest.approx(0.32, abs=0.01)
    assert any(v["rule_code"] == "EMP-001-EMPLOYER-NAME-MISMATCH" for v in r["triggered_rules"])


def test_employer_verification_reported_missing_when_no_source_available(engine):
    """When neither bank-side employer data nor a reported consistency check
    exists, the engine must say NOT VERIFIED, not silently pass."""
    p = _payload("sample_new_to_bank_payload")
    del p["consistency_checks"]["employer_name_match"]
    del p["consistency_checks"]["employer_match_similarity_score"]
    r = engine.evaluate(p)
    assert r["verification_checklist"]["employer_verified"] is False
    assert any(v["rule_code"] == "EMP-005-EMPLOYER-VERIFICATION-UNAVAILABLE" for v in r["triggered_rules"])

def test_stale_iscore_report_caught_via_score_date_fallback(engine):
    """Regression test: previously the engine only read 'inquiry_date', which
    this contract never sends -- so a 48-day-old bureau report (past the CBE
    30-day window) was never flagged. Must now fall back to score_date."""
    p = _payload("sample_fraudulent_applicant_payload")
    assert p["consistency_checks"]["iscore_report_age_days"] == 48
    r = engine.evaluate(p)
    assert any(v["rule_code"] == "BUR-001-STALE-ISCORE-REPORT" for v in r["triggered_rules"])
    assert r["verification_checklist"]["bureau_verified"] is False


def test_bureau_freshness_reported_unverifiable_when_no_date_or_age_available(engine):
    p = _payload("sample_new_to_bank_payload")
    del p["iscore_report_fields"]["score_date"]
    del p["consistency_checks"]["iscore_report_age_days"]
    r = engine.evaluate(p)
    assert any(v["rule_code"] == "BUR-004-ISCORE-FRESHNESS-UNVERIFIABLE" for v in r["triggered_rules"])
    assert r["verification_checklist"]["bureau_verified"] is False

def test_bounced_cheques_caught_via_returned_cheques_count_fallback(engine):
    """Regression test: the engine only checked 'bounced_cheques_count_12m',
    which this contract never sends -- so 2 returned cheques on the fraud
    applicant's account were invisible. Must fall back to returned_cheques_count."""
    p = _payload("sample_fraudulent_applicant_payload")
    assert p["bank_statement_fields"]["returned_cheques_count"]["value"] == 2
    r = engine.evaluate(p)
    assert any(v["rule_code"] == "BNK-001-BOUNCED-CHEQUES" for v in r["triggered_rules"])


def test_balance_math_failure_caught_via_consistency_checks_fallback(engine):
    """Regression test: previously missing bank_statement_fields.running_balance_math_verified
    defaulted to True (assumed balanced) -- a false pass identical to the employer bug.
    Must now use consistency_checks.running_balance_math_valid instead."""
    p = _payload("sample_fraudulent_applicant_payload")
    assert p["consistency_checks"]["running_balance_math_valid"] is False
    r = engine.evaluate(p)
    assert any(v["rule_code"] == "BNK-002-STATEMENT-ARITHMETIC-ANOMALY" for v in r["triggered_rules"])


def test_balance_math_reported_unverifiable_when_no_source_available(engine):
    p = _payload("sample_new_to_bank_payload")
    p["consistency_checks"].pop("running_balance_math_valid", None)
    r = engine.evaluate(p)
    assert any(v["rule_code"] == "BNK-003-BALANCE-MATH-UNVERIFIABLE" for v in r["triggered_rules"])

def test_account_reuse_across_identities_caught_via_account_number_fallback(engine):
    """Regression test: account collision must use account_number when
    bank_account_number is absent."""
    p1 = _payload("sample_fraudulent_applicant_payload")
    p2 = _payload("sample_new_to_bank_payload")
    p3 = _payload("sample_returning_customer_payload")

    shared_account = {"value": "EG9900987654", "confidence": 0.9}

    p1["bank_statement_fields"]["account_number"] = shared_account
    p2["bank_statement_fields"]["account_number"] = shared_account
    p3["bank_statement_fields"]["account_number"] = shared_account

    r1 = engine.evaluate(p1)
    r2 = engine.evaluate(p2)
    r3 = engine.evaluate(p3)

    results = [r1, r2, r3]

    assert any(
        v["rule_code"] == "VEL-002-CROSS-APP-ACCOUNT-COLLISION"
        for r in results
        for v in r["triggered_rules"]
    )