"""
Request schema for POST /api/v1/fraud/evaluate.

Design notes:
- One schema covers BOTH new-to-bank and returning-customer applications.
  `is_returning_customer` is a discriminator; `internal_history` fields are
  all optional with cold-start-safe defaults (matches your
  "internal_history_missing=1" convention for new applicants).
- The nested *_fields groups (national_id_fields, salary_certificate_fields,
  bank_statement_fields, iscore_report_fields) are deliberately kept as
  permissive `Dict[str, Any]` rather than fully-typed models. The engine
  reads each individual key defensively via `.get(key, {}).get("value", default)`,
  so a strict schema would reject payloads the engine can safely handle, and
  would need updating every time a new field is read inside fraud_engine.py.
  What we DO validate strictly is the outer envelope (top-level required
  sections, document list shape, form_data, internal_history) since that
  shape is stable and a mistake there breaks every downstream check at once.
- `model_config = ConfigDict(extra="allow")` everywhere: the engine tolerates
  unknown/extra keys, so the API shouldn't reject a payload just because it
  carries a field the engine doesn't (yet) use.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DocumentItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    document_type: str
    overall_quality_score: float = Field(ge=0.0, le=1.0)
    is_tampered_suspected: bool = False
    extracted_fields: Dict[str, Any] = Field(default_factory=dict)


class DeviceTelemetry(BaseModel):
    model_config = ConfigDict(extra="allow")

    is_vpn_or_proxy: bool = False
    device_is_rooted_or_emulator: bool = False
    submission_hour_local: int = 12
    device_fingerprint_id: Optional[str] = None


class FormData(BaseModel):
    """Only the fields the fraud engine actually reads are required.
    The rest are optional: the fraudulent sample payload legitimately arrives
    with just these four, and a fraud check should never be blocked because a
    marketing/demographic field is missing."""
    model_config = ConfigDict(extra="allow")

    requested_amount: float
    tenure_months: int
    requested_annuity: float
    loan_purpose: str
    goods_price: float = 0.0
    family_status: Optional[str] = None
    children_count: int = 0
    family_members_count: int = 1
    housing_type: Optional[str] = None
    education_type: Optional[str] = None
    owns_car: bool = False
    owns_realty: bool = False
    branch_id: Optional[int] = None
    # Read by Layer 3 entity-velocity checks; optional since not every channel collects it.
    mobile_phone: Optional[str] = None


class AggregatedMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    prev_app_count: int = 0
    prev_approved_count: int = 0
    prev_refused_count: int = 0
    prev_approved_ratio: float = 0.0
    prev_avg_credit: float = 0.0
    inst_payment_count: int = 0
    inst_late_count: int = 0
    inst_severe_late_count: int = 0
    inst_late_ratio: float = 0.0
    inst_avg_days_late: float = 0.0
    inst_max_days_late: int = 0
    pos_record_count: int = 0
    pos_avg_dpd: float = 0.0
    cc_avg_balance: float = 0.0
    cc_balance_to_limit_ratio: float = 0.0


class InternalHistory(BaseModel):
    """Cold-start-safe: every field defaults to the "brand new applicant" value,
    so a new-to-bank payload doesn't need to send this block at all."""
    model_config = ConfigDict(extra="allow")

    internal_history_missing: int = 1
    customer_id: Optional[str] = None
    kyc_status: str = "not_applicable"
    customer_since_date: Optional[str] = None
    relationship_tenure_months: int = 0
    bank_accounts: List[Dict[str, Any]] = Field(default_factory=list)
    previous_bank_loans: List[Dict[str, Any]] = Field(default_factory=list)
    aggregated_metrics: AggregatedMetrics = Field(default_factory=AggregatedMetrics)


class ApplicationPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    application_id: str
    applicant_type: str = "individual"
    submission_timestamp: datetime
    is_returning_customer: bool

    documents: List[DocumentItem] = Field(default_factory=list)

    # Kept permissive on purpose -- see module docstring.
    # REQUIRED (no default): pydantic v2 skips validators on default values, so a
    # default here would let a payload with no national_id_fields slip past the check below.
    national_id_fields: Dict[str, Any]
    salary_certificate_fields: Dict[str, Any] = Field(default_factory=dict)
    bank_statement_fields: Dict[str, Any] = Field(default_factory=dict)
    iscore_report_fields: Dict[str, Any] = Field(default_factory=dict)

    device_telemetry: DeviceTelemetry = Field(default_factory=DeviceTelemetry)
    form_data: FormData
    internal_history: InternalHistory = Field(default_factory=InternalHistory)
    consistency_checks: Dict[str, Any] = Field(default_factory=dict)
    excluded_features_note: Optional[str] = None

    @field_validator("national_id_fields")
    @classmethod
    def _require_national_id(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not v.get("national_id", {}).get("value"):
            raise ValueError("national_id_fields.national_id.value is required for identity verification (Layer 1).")
        return v

    @model_validator(mode="after")
    def _returning_customer_must_have_customer_id(self) -> "ApplicationPayload":
        if self.is_returning_customer and not self.internal_history.customer_id:
            raise ValueError(
                "is_returning_customer=true requires internal_history.customer_id to be set."
            )
        return self