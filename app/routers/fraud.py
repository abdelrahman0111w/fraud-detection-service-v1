import logging

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_engine
from app.schemas.request import ApplicationPayload
from app.schemas.response import EnrichedApplication, FraudAssessment

logger = logging.getLogger("fraud_api")

router = APIRouter(prefix="/api/v1/fraud", tags=["fraud"])
enrich_router = APIRouter(prefix="/api/v1/application", tags=["application"])


@router.post(
    "/evaluate",
    response_model=FraudAssessment,
    summary="Run the 5-layer fraud assessment on a loan application",
)
def evaluate_application(payload: ApplicationPayload, engine=Depends(get_engine)) -> FraudAssessment:
    """
    Validates the application payload, runs it through the 5-layer
    CreditFraudEngine, and returns ONLY the fraud_assessment block
    (risk_level, fraud_risk_score, recommended_action, reason codes,
    risk-adjusted salary) -- not the enriched full application.
    """
    # model_dump(mode="json") converts datetimes/nested models back into the
    # plain dict shape the engine's application.get(...) calls expect.
    application_dict = payload.model_dump(mode="json")

    try:
        assessment = engine.evaluate(application_dict)
    except Exception:
        # Never let an unexpected engine-internal error leak stack traces to
        # a caller, or silently return a partial/garbage assessment on a
        # real lending decision -- surface it as a clean 500 and log the
        # full detail server-side for investigation.
        logger.exception(
            "fraud_engine.evaluate() raised for application_id=%s",
            payload.application_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Fraud evaluation failed. This has been logged for investigation.",
        )

    return assessment


@enrich_router.post(
    "/enrich",
    response_model=EnrichedApplication,
    summary="Enrich an application payload with the full fraud_assessment block",
)
def enrich_application(payload: ApplicationPayload, engine=Depends(get_engine)) -> EnrichedApplication:
    """
    Validates the application payload, runs it through the fraud engine, and
    returns the ORIGINAL payload with a `fraud_assessment` block merged in --
    useful for a downstream credit-risk service that wants the full context,
    not just the decision (unlike /api/v1/fraud/evaluate, which returns the
    assessment alone).
    """
    application_dict = payload.model_dump(mode="json")

    try:
        enriched = engine.enrich_payload(application_dict)
    except Exception:
        logger.exception(
            "fraud_engine.enrich_payload() raised for application_id=%s",
            payload.application_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Application enrichment failed. This has been logged for investigation.",
        )

    return enriched