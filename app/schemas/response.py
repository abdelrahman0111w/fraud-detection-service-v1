"""
Response schema for POST /api/v1/fraud/evaluate.

This mirrors, field-for-field, the dict returned by
CreditFraudEngine.evaluate() in fraud_engine.py (see the `return {...}`
block at the end of that method). Keeping it 1:1 means:
  - FastAPI's response_model validation will immediately fail loudly if the
    engine's output shape ever drifts (e.g. someone renames a key inside
    fraud_engine.py) instead of silently returning something inconsistent.
  - The auto-generated OpenAPI docs at /docs are accurate for whoever
    consumes this API downstream.
"""
from typing import Any, List, Literal

from pydantic import BaseModel, ConfigDict


RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
RecommendedAction = Literal[
    "PROCEED_TO_CREDIT_EVALUATION",
    "REQUEST_ADDITIONAL_VERIFICATION_DOCUMENTS",
    "FLAG_FOR_MANUAL_FRAUD_INVESTIGATION",
    "REJECT_SUSPECTED_FRAUD",
]


class FraudRuleViolationSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    rule_code: str
    rule_name_en: str
    rule_name_ar: str
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    description_en: str
    description_ar: str
    observed_value: Any
    threshold_value: Any
    weight: float


class AnomalySignalSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    anomaly_name: str
    anomaly_score: float
    detected: bool
    explanation_en: str
    explanation_ar: str


class MetricsSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    income_mismatch_ratio: float
    employer_similarity_score: float
    inflow_uniformity_score: float
    benford_law_violation: bool
    terminal_digit_violation: bool
    adversarial_second_order_signal: bool
    isolation_forest_anomaly_score: float
    gradient_boost_fraud_probability: float
    entity_collisions_count: int
    total_violations_count: int
    critical_violations_count: int
    detected_anomalies_count: int
    model_engine_status: Literal["ONLINE_CALIBRATED_ARTIFACTS", "HEURISTIC_FALLBACK"]


class DownstreamRiskFeederSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    credibility_discount_factor: float
    declared_salary: float
    risk_adjusted_salary: float
    haircut_percentage: float
    proceed_with_credit_model: bool


class ReasonCodeSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: str
    title_en: str
    title_ar: str
    severity: str
    reason_en: str
    reason_ar: str


class ExplainableAiSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    regulatory_reason_codes: List[ReasonCodeSchema]
    executive_summary_ar: str
    executive_summary_en: str


class FraudAssessment(BaseModel):
    """Top-level response body for POST /api/v1/fraud/evaluate."""
    model_config = ConfigDict(extra="allow")

    fraud_risk_score: float
    fraud_risk_level: RiskLevel
    recommended_action: RecommendedAction
    action_ar: str
    verification_checklist: dict[str, bool]
    behavioral_anomalies: List[AnomalySignalSchema]
    metrics: MetricsSchema
    downstream_risk_feeder: DownstreamRiskFeederSchema
    explainable_ai: ExplainableAiSchema
    triggered_rules: List[FraudRuleViolationSchema]


class EnrichedApplication(BaseModel):
    """
    Response for POST /api/v1/application/enrich: the original application
    payload, unchanged, with a `fraud_assessment` block merged in. The rest of
    the payload's shape is the caller's own (extra="allow"), since it's just
    an echo of what they sent -- only `fraud_assessment` is a field this
    service actually produces and validates.
    """
    model_config = ConfigDict(extra="allow")

    fraud_assessment: FraudAssessment