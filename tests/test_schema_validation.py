"""
Bad input must be stopped at the door with a 422 -- never reach the engine,
never produce a 500, never produce a fraud decision on garbage data.
"""
import pytest

from tests.conftest import EVALUATE_URL


def test_empty_body_rejected(client):
    assert client.post(EVALUATE_URL, json={}).status_code == 422


@pytest.mark.parametrize("missing_key", [
    "application_id", "submission_timestamp", "is_returning_customer",
    "form_data", "national_id_fields",
])
def test_missing_required_top_level_key_rejected(client, load_payload, missing_key):
    payload = load_payload("sample_new_to_bank_payload")
    del payload[missing_key]
    assert client.post(EVALUATE_URL, json=payload).status_code == 422


def test_empty_national_id_rejected(client, load_payload):
    payload = load_payload("sample_new_to_bank_payload")
    payload["national_id_fields"]["national_id"]["value"] = ""
    assert client.post(EVALUATE_URL, json=payload).status_code == 422


def test_wrong_type_for_requested_amount_rejected(client, load_payload):
    payload = load_payload("sample_new_to_bank_payload")
    payload["form_data"]["requested_amount"] = "not-a-number"
    assert client.post(EVALUATE_URL, json=payload).status_code == 422


def test_bad_timestamp_rejected(client, load_payload):
    payload = load_payload("sample_new_to_bank_payload")
    payload["submission_timestamp"] = "yesterday-ish"
    assert client.post(EVALUATE_URL, json=payload).status_code == 422


@pytest.mark.parametrize("bad_score", [-0.1, 1.5])
def test_document_quality_score_out_of_range_rejected(client, load_payload, bad_score):
    payload = load_payload("sample_new_to_bank_payload")
    payload["documents"][0]["overall_quality_score"] = bad_score
    assert client.post(EVALUATE_URL, json=payload).status_code == 422


def test_returning_customer_without_customer_id_rejected(client, load_payload):
    payload = load_payload("sample_returning_customer_payload")
    payload["internal_history"]["customer_id"] = None
    assert client.post(EVALUATE_URL, json=payload).status_code == 422


def test_unknown_extra_fields_are_tolerated(client, load_payload):
    """The engine ignores unknown keys, so the API must not reject them."""
    payload = load_payload("sample_new_to_bank_payload")
    payload["some_future_field"] = {"anything": 1}
    payload["form_data"]["another_new_field"] = "x"
    assert client.post(EVALUATE_URL, json=payload).status_code == 200


def test_minimal_form_data_accepted(client, load_payload):
    """The fraudulent sample only sends 4 form_data fields -- must validate."""
    resp = client.post(EVALUATE_URL, json=load_payload("sample_fraudulent_applicant_payload"))
    assert resp.status_code == 200, resp.text