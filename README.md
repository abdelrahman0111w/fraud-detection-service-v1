# Fraud Detection Service

A production FastAPI service wrapping CrediX's 5-layer fraud detection engine
(`fraud_engine.py`). It validates a loan application payload, runs it through
deterministic rules, forensic anomaly detection, a 48-hour entity-velocity
store, and a dual-engine ML model (Isolation Forest + HistGradientBoosting),
then returns a bilingual (Arabic/English) risk assessment.

This service is scoped to **fraud detection only** — it does not perform
credit-risk/default-probability scoring.

---

## Architecture

```
Caller
  │  POST /api/v1/fraud/evaluate  or  POST /api/v1/application/enrich
  │  header: X-API-Key
  ▼
FastAPI (app/main.py)
  ├─ AuditLoggingMiddleware    → writes one JSON line per request to audit_log.jsonl
  ├─ verify_api_key dependency → 401 on missing/invalid key
  ├─ ApplicationPayload schema → validates the request body (422 on bad input)
  ▼
CreditFraudEngine.evaluate() / .enrich_payload()   (fraud_engine.py)
  ├─ Layer 1: Deterministic rules (identity, income, employer, documents, bureau, banking, device)
  ├─ Layer 2: Forensic anomalies (Benford's Law, terminal-digit, threshold gaming)
  ├─ Layer 3: SQLite entity-velocity store (48h phone/account/device collision detection)
  ├─ Layer 4: ML inference (Isolation Forest + HistGradientBoosting, pre-trained)
  └─ Layer 5: Hybrid decision fusion + bilingual explainability + income haircut
  ▼
FraudAssessment schema → validates the response shape before it goes out
```

---

## Project layout

```
fraud-detection-service-v1/
├── fraud_engine.py              # The 5-layer engine (inference only, no training)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml           # Local Docker Compose setup
├── railway.json                 # Railway deployment config
├── pytest.ini
├── .env                         # Local secrets (FRAUD_API_KEY) — never committed
│
├── app/
│   ├── main.py                  # FastAPI app, startup/health/version
│   ├── config.py                # Settings (env vars, see below)
│   ├── dependencies.py          # Engine singleton, loaded once at startup
│   ├── middleware/
│   │   ├── auth.py              # X-API-Key check
│   │   └── logging.py           # Append-only audit log middleware
│   ├── routers/
│   │   └── fraud.py             # /evaluate and /enrich endpoints
│   └── schemas/
│       ├── request.py           # ApplicationPayload (input validation)
│       └── response.py          # FraudAssessment / EnrichedApplication (output validation)
│
├── model/
│   ├── train_fraud.py           # Offline training script (produces the 6 artifacts below)
│   └── artifacts/fraud/
│       ├── isolation_forest_v2.joblib
│       ├── fraud_gradient_boost_v2.joblib
│       ├── scaler_v2.joblib
│       ├── mahalanobis_precision_v2.joblib
│       ├── feature_names.json
│       └── metrics.json
│
├── data/
│   └── fraud_training_data_25000.csv   # Training data (local only, not deployed)
│
└── tests/
    ├── conftest.py               # Shared fixtures (TestClient, isolated temp DB/audit log)
    ├── test_fraud_evaluate.py
    ├── test_application_enrich.py
    ├── test_new_customer_payload.py
    ├── test_returning_customer_payload.py
    ├── test_fraudulent_payload.py
    ├── test_schema_validation.py
    ├── test_auth_and_audit.py
    ├── test_engine_regression.py
    └── fixtures/
        ├── sample_new_to_bank_payload.json
        ├── sample_returning_customer_payload.json
        └── sample_fraudulent_applicant_payload.json
```

---

## Local setup

```bash
conda create -n fraud-detection-service-v1 python=3.12
conda activate fraud-detection-service-v1

pip install -r requirements.txt
```

Create a `.env` file in the project root (never commit this):
```
FRAUD_API_KEY=<a long random string — generate with: python -c "import secrets; print(secrets.token_urlsafe(32))">
```
If `FRAUD_API_KEY` is unset, authentication is disabled (convenient for local dev, must be set before any shared/public deployment).

Run the service:
```bash
uvicorn app.main:app --reload --port 8000
```

Open interactive API docs at: `http://localhost:8000/docs`

Run the test suite:
```bash
pytest tests/ -v
```
54 tests covering schema validation, every Layer 1–3 rule (including the
caller-reported-consistency-check fallbacks), ML-layer agreement, auth, and
the audit trail.

---

## API reference

All `/api/v1/*` endpoints require the header `X-API-Key: <your key>`. A
missing or wrong key returns `401`. A malformed request body returns `422`
with field-level detail, before it ever reaches the fraud engine.

### `POST /api/v1/fraud/evaluate`
Runs the full 5-layer assessment and returns **only** the `fraud_assessment`
block — `fraud_risk_level`, `fraud_risk_score`, `recommended_action`,
bilingual reason codes, and the risk-adjusted salary.

### `POST /api/v1/application/enrich`
Same assessment, but returns the **original application payload unchanged**,
with a `fraud_assessment` block merged in. Useful for a downstream system
(e.g. a credit-risk model) that needs the full application context plus the
fraud decision in one call.

### `GET /health`
Returns `model_status: "production_artifacts"` when the trained ML artifacts
loaded successfully, or `"heuristic_fallback"` if they didn't (this should
never happen in production — alert on it if it does).

### `GET /version`
Returns the API version string.

### Decision outcomes

| `fraud_risk_level` | `recommended_action` |
|---|---|
| `LOW` | `PROCEED_TO_CREDIT_EVALUATION` |
| `MEDIUM` | `REQUEST_ADDITIONAL_VERIFICATION_DOCUMENTS` |
| `HIGH` | `FLAG_FOR_MANUAL_FRAUD_INVESTIGATION` |
| `CRITICAL` | `REJECT_SUSPECTED_FRAUD` |

---

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `FRAUD_API_KEY` | unset (auth disabled) | Shared-secret key required in `X-API-Key` |
| `FRAUD_ARTIFACT_DIR` | `model/artifacts/fraud` | Where the 6 trained artifacts are loaded from |
| `FRAUD_DB_PATH` | `fraud_registry.db` | SQLite entity-velocity store |
| `FRAUD_AUDIT_LOG_PATH` | `audit_log.jsonl` | Append-only audit log (one JSON line per request) |

---

## Docker (local)

```bash
# .env must contain FRAUD_API_KEY, or compose will refuse to start
docker compose up --build
```
Mounts a named volume at `/srv/data` so `fraud_registry.db` and
`audit_log.jsonl` survive container restarts.

---

## Deploying to Railway

1. Push this repo to GitHub (`.env`, `fraud_registry.db`, and
   `audit_log.jsonl` are already excluded via `.gitignore` — model artifacts
   under `model/artifacts/fraud/` **must** be committed, since Railway's
   build needs them).
2. Railway dashboard → **New Project** → **Deploy from GitHub repo**.
   `railway.json` tells Railway to build from the `Dockerfile`.
3. Service → **Variables** → set `FRAUD_API_KEY` (generate a fresh one —
   never reuse a key that's appeared in chat logs, docs, or screenshots).
4. Service → **Settings** → **Volumes** → add a volume mounted at `/srv/data`.
5. Service → **Settings** → **Networking** → **Generate Domain** for a public
   HTTPS URL (TLS is automatic on Railway).
6. Every `git push` to the linked branch auto-deploys.

---

## Retraining the models

```bash
python model/train_fraud.py --data data/fraud_training_data_25000.csv --out model/artifacts/fraud
```
Always retrain inside the **same environment** (same scikit-learn version)
that will run the API — a version mismatch between training and serving
produces `InconsistentVersionWarning` and is a model-risk concern, not just
a cosmetic warning. After retraining, re-run `pytest tests/ -v` and review
the golden-decision tests before redeploying.

---

## Known limitations

- **Benford's Law / terminal-digit forensics are implemented but dormant.**
  They require `bank_statement_fields.sample_transaction_amounts` (an
  itemized transaction list), which the current document-parsing pipeline
  doesn't supply — it extracts only summary statistics. All other Layer 1–5
  checks are fully active.
- **SQLite entity-velocity store is single-instance.** Fine for one running
  replica; if this service is ever scaled to multiple instances behind a
  load balancer, the store needs to move to Postgres or Redis so all
  instances see the same 48-hour collision data.
- Model governance / regulatory sign-off for using this in real lending
  decisions is outside this service's scope and is an organizational
  decision, not a technical one.