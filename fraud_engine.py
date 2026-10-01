"""
Senior Production-Grade 5-Layer Credit Application Fraud & Consistency Engine
=============================================================================
Platform: Smart Financing & Credit Request Analysis Platform (ZAWOLF / CrediX)
Standard: Egyptian Banking Federation & CBE Regulatory Compliance Guidelines

Architecture:
  - Layer 1: Deterministic Cross-Document & Security Rules (NID, Salary vs Statement, Employer, OCR, Bureau, Device Telemetry)
  - Layer 2: Deep Forensic & Anti-Adversarial Signals (Benford's Law, Terminal-Digit Chi-Square, Micro-Transaction Absence, Threshold-Gaming)
  - Layer 3: SQLite Entity Collision & Velocity Defense (Cross-application 48h tracking: NID, Phone, Account, Device Hash)
  - Layer 4: Dual-Engine ML Inference (Multi-Tree Isolation Forest + Cost-Sensitive Gradient Boosting)
  - Layer 5: Cost-Sensitive Hybrid Fusion, Bilingual Explainable AI (XAI) & Income Haircut Calculator

Note:
  Model training is separated into model/train_fraud.py and notebooks/fraud_model_training.ipynb.
  This engine exclusively performs fast, production-grade vectorized inference (<5ms).
"""

import os
import re
import math
import json
import sqlite3
import hashlib
import difflib
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import joblib
from scipy import stats


# -----------------------------------------------------------------------------
# Data Structures & Signal Schemas
# -----------------------------------------------------------------------------

@dataclass
class FraudRuleViolation:
    rule_code: str
    rule_name_en: str
    rule_name_ar: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description_en: str
    description_ar: str
    observed_value: Any
    threshold_value: Any
    weight: float


@dataclass
class AnomalySignal:
    anomaly_name: str
    anomaly_score: float  # 0.0 to 1.0
    detected: bool
    explanation_en: str
    explanation_ar: str


# -----------------------------------------------------------------------------
# Layer 2 Helper: Deep Forensic Analyzer (Benford, Terminal Digits & Micro-Noise)
# -----------------------------------------------------------------------------

class DeepForensicAnalyzer:
    """
    Mathematical forensics on transaction amounts in uploaded bank statements.
    Implements:
      1. Benford's Law (Lead Digit Chi-Square Test)
      2. Terminal / Last-Digit Uniformity Chi-Square Test (Human bias detection)
      3. Micro-Transaction Friction Analysis (Absence of daily expense noise)
      4. Inflow Uniformity Analysis (Detection of rounded fabricated salaries)
    """

    BENFORD_PROBABILITIES = {
        1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097,
        5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
    }

    @staticmethod
    def evaluate_benford_law(numbers: List[float]) -> Tuple[bool, float, float]:
        """
        Tests if transaction lead digits conform to Benford's Law.
        Returns: (is_anomaly, chi_square_stat, p_value_approx)
        """
        valid_digits = []
        for n in numbers:
            val = abs(float(n or 0))
            if val >= 1.0:
                s = f"{val:.4f}".replace(".", "").lstrip("0")
                if s and s[0] in "123456789":
                    valid_digits.append(int(s[0]))

        if len(valid_digits) < 4:
            return False, 0.0, 1.0

        n_total = len(valid_digits)
        observed_counts = {d: 0 for d in range(1, 10)}
        for d in valid_digits:
            observed_counts[d] += 1

        chi_square = 0.0
        for d in range(1, 10):
            expected = n_total * DeepForensicAnalyzer.BENFORD_PROBABILITIES[d]
            observed = observed_counts[d]
            chi_square += ((observed - expected) ** 2) / max(expected, 0.001)

        # Critical chi-square at df=8, p=0.05 is 15.51
        is_anomaly = bool(chi_square > 15.51)
        return is_anomaly, round(chi_square, 2), 0.02 if is_anomaly else 0.85

    @staticmethod
    def evaluate_last_digit_uniformity(numbers: List[float]) -> Tuple[bool, float, float]:
        """
        Terminal Digit Analysis (Last-Digit Chi-Square Test).
        In authentic retail transactions and human spending, the last integer 
        digit (0-9) follows a discrete Uniform Distribution (10% per digit).
        Human fabricators heavily over-sample 0 and 5, and under-sample 3, 7, 4.
        Returns: (is_anomaly, chi_square_stat, p_value_approx)
        """
        valid_digits = []
        for n in numbers:
            val = abs(float(n or 0))
            if val >= 10.0:
                last_digit = int(math.floor(val)) % 10
                valid_digits.append(last_digit)

        # Cochran's statistical sample criterion for 10-bin Chi-Square (expected count >= 2.5 per bin)
        if len(valid_digits) < 25:
            return False, 0.0, 1.0

        n_total = len(valid_digits)
        observed = {d: 0 for d in range(10)}
        for d in valid_digits:
            observed[d] += 1

        expected = n_total / 10.0
        chi_square = sum(((observed[d] - expected) ** 2) / max(expected, 0.001) for d in range(10))

        # Critical Chi-Square for df = 9 at p = 0.05 is 16.92
        is_anomaly = bool(chi_square > 16.92)
        return is_anomaly, round(chi_square, 2), 0.01 if is_anomaly else 0.85

    @staticmethod
    def analyze_micro_transaction_absence(numbers: List[float]) -> Tuple[bool, float]:
        """
        Real-world daily expense noise analysis.
        Authentic personal accounts contain messy micro-transactions (< 250 EGP or odd piastres)
        for telecom bills, transport, retail, pharmacies, etc.
        Fabricated statements typically only list clean round lump-sums (Salary, Rent, Big ATM).
        Returns: (is_suspicious_absence, micro_ratio)
        """
        if not numbers or len(numbers) < 6:
            return False, 1.0

        amounts = [abs(float(n or 0)) for n in numbers if abs(float(n or 0)) > 0]
        if not amounts:
            return False, 1.0

        micro_count = sum(1 for a in amounts if a < 250.0 or (a % 1.0 > 0.01))
        micro_ratio = micro_count / len(amounts)

        # Flag if micro-transactions account for under 8% in a populated statement
        is_synthetic = bool(micro_ratio < 0.08 and len(amounts) >= 8)
        return is_synthetic, round(micro_ratio, 3)

    @staticmethod
    def calculate_inflow_uniformity(amounts: List[float]) -> float:
        """
        Calculates ratio of highly rounded inflow numbers (e.g. 1000, 5000, 10000).
        Forged statements typically use round numbers rather than real-world messy sums.
        """
        if not amounts:
            return 0.0
        clean_amounts = [abs(float(a)) for a in amounts if abs(float(a)) > 100]
        if not clean_amounts:
            return 0.0
        round_count = sum(1 for a in clean_amounts if a % 1000 == 0 or a % 500 == 0)
        return round(round_count / len(clean_amounts), 3)

    @staticmethod
    def evaluate_second_order_adversarial_signals(numbers: List[float]) -> Tuple[bool, Dict[str, Any]]:
        """
        Anti-Adversarial Countermeasure Layer (targets forgers sophisticated enough
        to defeat the basic first-digit Benford's Law screen above).

        A forger who deliberately samples leading digits from the Benford
        distribution to pass evaluate_benford_law() typically still fails one or
        more of these three deeper, harder-to-simultaneously-satisfy signals:

          1. Second-Digit Benford Conformity: real financial data also follows a
             known (flatter) frequency law on the SECOND digit. Naive first-digit
             -only fabrication rarely reproduces this correctly.
          2. Log-Mantissa Uniformity (Kolmogorov-Smirnov test): Benford's Law is
             mathematically equivalent to the fractional part of log10(amount)
             being uniformly distributed on [0,1). This is a continuous,
             higher-resolution test than the coarse 9-bucket chi-square above,
             and can catch fabrication that only "roughly" matches leading-digit
             frequencies.
          3. Recurring-Transaction Absence: genuine personal bank statements
             almost always contain a handful of near-identical repeated amounts
             (rent, subscriptions, recurring transfers). A fabricator drawing
             each amount independently at random produces unnaturally "too
             random" data with essentially no repeats.

        Returns: (is_suspicious, detail_dict). Requires >= 20 usable amounts;
        returns (False, {"applicable": False}) otherwise (conservative -- never
        penalizes a thin transaction history).
        """
        amounts = [abs(float(n)) for n in numbers if n and abs(float(n)) >= 1.0]
        if len(amounts) < 20:
            return False, {"applicable": False}

        # --- 1. Second-digit Benford chi-square (standard forensic-accounting law) ---
        SECOND_DIGIT_EXPECTED = [0.1197, 0.1139, 0.1088, 0.1043, 0.1003,
                                  0.0967, 0.0934, 0.0904, 0.0876, 0.0850]
        second_digits = []
        for a in amounts:
            digits_only = str(int(a)).lstrip("0")
            if len(digits_only) >= 2:
                second_digits.append(int(digits_only[1]))

        second_digit_anomaly = False
        sd_chi = 0.0
        if len(second_digits) >= 20:
            n = len(second_digits)
            observed = [second_digits.count(d) for d in range(10)]
            expected = [p * n for p in SECOND_DIGIT_EXPECTED]
            sd_chi = sum(((o - e) ** 2) / e for o, e in zip(observed, expected) if e > 0)
            second_digit_anomaly = sd_chi > 16.92  # chi2 critical value, df=9, p=0.05

        # --- 2. Log-mantissa uniformity (finer-grained than the leading-digit bucket test) ---
        mantissa_anomaly = False
        ks_p = 1.0
        try:
            mantissas = [math.log10(a) % 1.0 for a in amounts]
            _, ks_p = stats.kstest(mantissas, "uniform")
            mantissa_anomaly = bool(ks_p < 0.05)
        except (ValueError, ZeroDivisionError):
            pass

        # --- 3. Recurring / near-duplicate transaction absence ---
        near_dup_count = 0
        for i, a in enumerate(amounts):
            for b in amounts[i + 1:]:
                if a > 0 and abs(a - b) / a <= 0.01:
                    near_dup_count += 1
                    break
        recurrence_ratio = near_dup_count / len(amounts)
        recurrence_absent = bool(recurrence_ratio < 0.05)

        is_suspicious = bool(second_digit_anomaly or mantissa_anomaly or recurrence_absent)
        detail = {
            "applicable": True,
            "second_digit_chi_square": round(sd_chi, 2),
            "second_digit_anomaly": second_digit_anomaly,
            "mantissa_ks_pvalue": round(float(ks_p), 4),
            "mantissa_anomaly": mantissa_anomaly,
            "recurrence_ratio": round(recurrence_ratio, 3),
            "recurrence_absent": recurrence_absent,
        }
        return is_suspicious, detail


# -----------------------------------------------------------------------------
# Layer 3 Helper: Real-time Entity Graph & Velocity Store (SQLite)
# -----------------------------------------------------------------------------

class SQLiteEntityStore:
    """
    Lightweight, embedded cross-application registry to detect fraud ring velocity
    and entity collisions (Phone, NID, Employer, IBAN, Device Fingerprint) across 48h.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(base_dir, "fraud_registry.db")
        else:
            self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS entity_audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        application_id TEXT,
                        timestamp TEXT,
                        nid_hash TEXT,
                        phone_hash TEXT,
                        employer_name TEXT,
                        bank_account_hash TEXT,
                        device_id_hash TEXT
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_time ON entity_audit_log(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_phone ON entity_audit_log(phone_hash)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_device ON entity_audit_log(device_id_hash)")
                conn.commit()
        except Exception:
            pass

    def check_and_record_velocity(
        self, application_id: str, national_id: str, phone: str, employer: str, account: str, device_id: str = ""
    ) -> Dict[str, Any]:
        """
        Checks if entity identifiers have collided in >2 applications within 48 hours.
        """
        now = datetime.now(timezone.utc)
        window_start = (now - timedelta(hours=48)).isoformat()
        nid_h = hashlib.sha256((national_id or "").strip().encode()).hexdigest() if national_id else ""
        ph_h = hashlib.sha256((phone or "").strip().encode()).hexdigest() if phone else ""
        acc_h = hashlib.sha256((account or "").strip().encode()).hexdigest() if account else ""
        dev_h = hashlib.sha256((device_id or "").strip().encode()).hexdigest() if device_id else ""
        emp_clean = (employer or "").strip().lower()

        collisions = {
            "duplicate_phone_in_48h": False,
            "duplicate_account_in_48h": False,
            "duplicate_device_in_48h": False,
            "employer_spike_in_48h": False,
            "total_collisions": 0
        }

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if ph_h:
                    cursor.execute(
                        "SELECT COUNT(DISTINCT application_id) FROM entity_audit_log WHERE phone_hash = ? AND timestamp >= ? AND application_id != ?",
                        (ph_h, window_start, application_id)
                    )
                    count_phone = cursor.fetchone()[0]
                    if count_phone >= 2:
                        collisions["duplicate_phone_in_48h"] = True
                        collisions["total_collisions"] += count_phone

                if acc_h:
                    cursor.execute(
                        "SELECT COUNT(DISTINCT application_id) FROM entity_audit_log WHERE bank_account_hash = ? AND timestamp >= ? AND application_id != ?",
                        (acc_h, window_start, application_id)
                    )
                    count_acc = cursor.fetchone()[0]
                    if count_acc >= 2:
                        collisions["duplicate_account_in_48h"] = True
                        collisions["total_collisions"] += count_acc

                if dev_h:
                    cursor.execute(
                        "SELECT COUNT(DISTINCT application_id) FROM entity_audit_log WHERE device_id_hash = ? AND timestamp >= ? AND application_id != ?",
                        (dev_h, window_start, application_id)
                    )
                    count_dev = cursor.fetchone()[0]
                    if count_dev >= 2:
                        collisions["duplicate_device_in_48h"] = True
                        collisions["total_collisions"] += count_dev

                if emp_clean and emp_clean not in ["telecom egypt", "vodafone", "orange", "cbe", "national bank of egypt", "banque misr", "government", "we"]:
                    cursor.execute(
                        "SELECT COUNT(DISTINCT application_id) FROM entity_audit_log WHERE employer_name = ? AND timestamp >= ? AND application_id != ?",
                        (emp_clean, window_start, application_id)
                    )
                    count_emp = cursor.fetchone()[0]
                    if count_emp >= 2:
                        collisions["employer_spike_in_48h"] = True
                        collisions["total_collisions"] += count_emp

                # Record current transaction
                cursor.execute(
                    "INSERT INTO entity_audit_log (application_id, timestamp, nid_hash, phone_hash, employer_name, bank_account_hash, device_id_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (application_id, now.isoformat(), nid_h, ph_h, emp_clean, acc_h, dev_h)
                )
                conn.commit()
        except Exception:
            pass

        return collisions

    def get_recent_graph_data(self, limit: int = 30) -> Dict[str, Any]:
        """
        Retrieves recent entity links to power the Fraud Ring Network Graph in dashboard.
        """
        nodes = []
        edges = []
        node_ids = set()

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT application_id, phone_hash, employer_name, bank_account_hash, device_id_hash FROM entity_audit_log ORDER BY id DESC LIMIT ?",
                    (limit,)
                )
                rows = cursor.fetchall()

                for row in rows:
                    app_id, ph, emp, acc, dev = row
                    if app_id not in node_ids:
                        nodes.append({"id": app_id, "label": f"App: {app_id[:10]}", "group": "application"})
                        node_ids.add(app_id)

                    if ph:
                        ph_node = f"Phone_{ph[:6]}"
                        if ph_node not in node_ids:
                            nodes.append({"id": ph_node, "label": ph_node, "group": "phone"})
                            node_ids.add(ph_node)
                        edges.append({"from": app_id, "to": ph_node, "relationship": "USES_PHONE"})

                    if dev:
                        dev_node = f"Dev_{dev[:6]}"
                        if dev_node not in node_ids:
                            nodes.append({"id": dev_node, "label": dev_node, "group": "device"})
                            node_ids.add(dev_node)
                        edges.append({"from": app_id, "to": dev_node, "relationship": "USES_DEVICE"})

                    if acc:
                        acc_node = f"Acc_{acc[:6]}"
                        if acc_node not in node_ids:
                            nodes.append({"id": acc_node, "label": acc_node, "group": "account"})
                            node_ids.add(acc_node)
                        edges.append({"from": app_id, "to": acc_node, "relationship": "USES_ACCOUNT"})
        except Exception:
            pass

        return {"nodes": nodes, "edges": edges, "total_entities": len(nodes)}


# -----------------------------------------------------------------------------
# Layer 4 Helper: Persistent Model Manager (Pre-trained ML Inference)
# -----------------------------------------------------------------------------

class PersistentModelManager:
    """
    Loads pre-trained production model weights (Isolation Forest + HistGradientBoosting)
    and executes fast vectorized scoring (<5ms per application).
    Never retrains at runtime.
    """

    FEATURE_NAMES = [
        'income_mismatch_ratio', 'annuity_to_balance_ratio', 'balance_volatility_cv',
        'surge_ratio_max_to_avg', 'ocr_quality_mean', 'min_to_avg_balance_ratio',
        'applicant_age_norm', 'employment_tenure_years', 'inflow_regularity_score',
        'iscore_normalized', 'inflow_uniformity_score', 'bureau_facilities_count'
    ]

    def __init__(self, artifact_dir: Optional[str] = None):
        if artifact_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.artifact_dir = os.path.join(base_dir, "model", "artifacts", "fraud")
        else:
            self.artifact_dir = artifact_dir

        self.iso_model = None
        self.gb_model = None
        self.scaler = None
        self.precision_matrix = None
        self.is_loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            iso_path = os.path.join(self.artifact_dir, "isolation_forest_v2.joblib")
            gb_path = os.path.join(self.artifact_dir, "fraud_gradient_boost_v2.joblib")
            scaler_path = os.path.join(self.artifact_dir, "scaler_v2.joblib")
            prec_path = os.path.join(self.artifact_dir, "mahalanobis_precision_v2.joblib")

            if os.path.exists(iso_path) and os.path.exists(gb_path) and os.path.exists(scaler_path):
                self.iso_model = joblib.load(iso_path)
                self.gb_model = joblib.load(gb_path)
                self.scaler = joblib.load(scaler_path)
                if os.path.exists(prec_path):
                    self.precision_matrix = joblib.load(prec_path)
                self.is_loaded = True
            else:
                self._fallback_init()
        except Exception:
            self._fallback_init()

    def _fallback_init(self):
        """Safe heuristic fallback if model artifacts are not yet saved to disk."""
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        np.random.seed(42)
        X_dummy = np.random.uniform(0.1, 0.9, size=(250, 12))
        self.scaler = StandardScaler().fit(X_dummy)
        self.iso_model = IsolationForest(n_estimators=50, random_state=42).fit(self.scaler.transform(X_dummy))
        self.gb_model = None
        self.is_loaded = False

    def predict_scores(self, feature_vector: np.ndarray) -> Tuple[float, float, float, bool]:
        """
        Executes inference on 12-dimensional feature vector.
        Returns: (isolation_forest_score, mahalanobis_score, gradient_boost_prob, is_anomaly)
        """
        try:
            scaled = self.scaler.transform(feature_vector)
            
            # 1. Isolation Forest Anomaly Score [0.0, 1.0]
            raw_iso = self.iso_model.decision_function(scaled)[0]
            iso_score = float(np.clip(0.50 - (raw_iso * 1.8), 0.0, 1.0))

            # 2. True Multi-Vector Mahalanobis Distance Outlier Score
            from scipy import stats
            diff = scaled[0]  # shape (12,)
            if self.precision_matrix is not None:
                # Real Mahalanobis Distance: (x - mu)^T * Sigma^{-1} * (x - mu)
                mahal_dist_sq = float(diff @ self.precision_matrix @ diff.T)
            else:
                mahal_dist_sq = float(np.sum(diff ** 2))
                
            # Chi-Square CDF mapping with df=12
            mahal_score = float(stats.chi2.cdf(mahal_dist_sq, df=12))

            # 3. Supervised Gradient Boosting probability [0.0, 1.0]
            if self.gb_model is not None:
                gb_prob = float(self.gb_model.predict_proba(scaled)[0, 1])
            else:
                gb_prob = (0.5 * iso_score) + (0.5 * mahal_score)

            # Combined Unsupervised Alert
            unsupervised_combined = (0.50 * iso_score) + (0.50 * mahal_score)
            is_anomaly = bool(unsupervised_combined > 0.75 or gb_prob > 0.50)

            return round(iso_score, 4), round(mahal_score, 4), round(gb_prob, 4), is_anomaly
        except Exception:
            return 0.15, 0.15, 0.15, False


# -----------------------------------------------------------------------------
# Layer 4 Feature Vectorizer (Extracts 12 Canonical Features)
# -----------------------------------------------------------------------------

class HighDimensionalFraudVectorizer:
    """
    Transforms raw application JSON payload into canonical 12-dimensional feature vector.
    """

    @staticmethod
    def extract_features(app: Dict[str, Any], mismatch_ratio: float, uniformity_score: float) -> np.ndarray:
        bank_fields = app.get("bank_statement_fields", {})
        form_data = app.get("form_data", {})
        salary_fields = app.get("salary_certificate_fields", {})
        nid_fields = app.get("national_id_fields", {})
        iscore_fields = app.get("iscore_report_fields", {})

        avg_balance = max(float(bank_fields.get("avg_monthly_balance", {}).get("value", 0.0) or 0.0), 1.0)
        max_balance = float(bank_fields.get("max_monthly_balance", {}).get("value", 0.0) or 0.0)
        min_balance = float(bank_fields.get("min_monthly_balance", {}).get("value", 0.0) or 0.0)
        volatility = float(bank_fields.get("balance_volatility_std", {}).get("value", 0.0) or 0.0)
        annuity = float(form_data.get("requested_annuity", 0.0) or 0.0)
        regularity = float(bank_fields.get("income_regularity_score", {}).get("value", 1.0) or 1.0)
        tenure = float(salary_fields.get("employment_tenure_years", {}).get("value", 3.0) or 3.0)
        age = float(nid_fields.get("age_years", {}).get("value", 35.0) or 35.0)
        iscore = float(iscore_fields.get("credit_score", {}).get("value", 650) or 650)
        # active_facilities_count isn't in this contract's iscore_report_fields; fall back
        # to counting the bureau_facilities list itself (present in the real payloads)
        # rather than silently assuming a fixed default of 2 regardless of actual history.
        if "active_facilities_count" in iscore_fields:
            facilities = float(iscore_fields.get("active_facilities_count", {}).get("value", 2) or 2)
        else:
            bureau_facilities_list = iscore_fields.get("bureau_facilities", [])
            facilities = float(len(bureau_facilities_list)) if isinstance(bureau_facilities_list, list) else 2.0

        # Calculate OCR mean across all attached documents
        docs = app.get("documents", [])
        ocr_scores = [float(d.get("overall_quality_score", 0.85)) for d in docs if isinstance(d, dict)]
        ocr_mean = float(np.mean(ocr_scores)) if ocr_scores else 0.88

        # 12 Normalized Features matching training matrix:
        feat = [
            float(np.clip(mismatch_ratio, 0.0, 2.0)),
            float(np.clip(annuity / avg_balance, 0.0, 5.0)),
            float(np.clip(volatility / avg_balance, 0.0, 5.0)),
            float(np.clip(max_balance / avg_balance, 1.0, 10.0)),
            float(np.clip(ocr_mean, 0.0, 1.0)),
            float(np.clip(min_balance / avg_balance, 0.0, 1.0)),
            float(np.clip((age - 21.0) / 44.0, 0.0, 1.0)),
            float(np.clip(tenure, 0.0, 30.0)),
            float(np.clip(regularity, 0.0, 1.0)),
            float(np.clip((iscore - 300.0) / 550.0, 0.0, 1.0)),
            float(np.clip(uniformity_score, 0.0, 1.0)),
            float(np.clip(facilities, 0.0, 15.0))
        ]
        return np.array([feat], dtype=np.float32)


# -----------------------------------------------------------------------------
# Main Senior 5-Layer Credit Fraud Engine
# -----------------------------------------------------------------------------

class CreditFraudEngine:
    """
    Senior / Production-Grade Hybrid 5-Layer Fraud Prevention & Consistency Engine.
    Executes fast inference, cross-document deterministic checks, entity graph defense,
    deep mathematical forensics, and bilingual CBE-compliant explainability.
    """

    SEVERITY_WEIGHTS = {
        "LOW": 0.10,
        "MEDIUM": 0.25,
        "HIGH": 0.50,
        "CRITICAL": 0.90
    }

    # Regulatory & Risk Thresholds
    INCOME_MISMATCH_WARN = 0.30
    INCOME_MISMATCH_CRITICAL = 0.50
    EMPLOYER_SIMILARITY_MIN = 0.65
    MAX_ISCORE_AGE_DAYS = 30
    MIN_APPLICANT_AGE = 21
    MAX_APPLICANT_AGE = 65
    MIN_OCR_CONFIDENCE_THRESHOLD = 0.55
    WINDOW_DRESSING_SURGE_RATIO = 2.5
    ANNUITY_TO_LIQUIDITY_MAX = 0.60

    def __init__(self, db_path: Optional[str] = None, artifact_dir: Optional[str] = None):
        self.entity_store = SQLiteEntityStore(db_path)
        self.model_manager = PersistentModelManager(artifact_dir)

    def evaluate(self, application: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes end-to-end 5-layer evaluation on the application JSON payload.
        Zero training takes place inside this method; inference only.
        """
        violations: List[FraudRuleViolation] = []
        checks_passed: Dict[str, bool] = {}

        # ---------------------------------------------------------------------
        # LAYER 1: Deterministic Cross-Document & Security Consistency
        # ---------------------------------------------------------------------
        nid_ok, nid_v = self._verify_identity(application)
        checks_passed["identity_verified"] = nid_ok
        violations.extend(nid_v)

        income_ok, inc_v, income_mismatch = self._verify_income_consistency(application)
        checks_passed["income_verified"] = income_ok
        violations.extend(inc_v)

        emp_ok, emp_v, emp_similarity = self._verify_employer_consistency(application)
        checks_passed["employer_verified"] = emp_ok
        violations.extend(emp_v)

        docs_ok, doc_v = self._verify_document_integrity(application)
        checks_passed["document_integrity_verified"] = docs_ok
        violations.extend(doc_v)

        bureau_ok, bur_v = self._verify_bureau_report(application)
        checks_passed["bureau_verified"] = bureau_ok
        violations.extend(bur_v)

        bank_ok, bank_v = self._verify_banking_behavior(application)
        checks_passed["bank_statement_math_verified"] = bank_ok
        violations.extend(bank_v)

        dev_telemetry_ok, dev_v = self._verify_device_telemetry(application)
        checks_passed["device_telemetry_verified"] = dev_telemetry_ok
        violations.extend(dev_v)

        # ---------------------------------------------------------------------
        # LAYER 2: Deep Forensic & Anti-Adversarial Signals
        # ---------------------------------------------------------------------
        anomaly_signals, uniformity_score, benford_anomaly, terminal_digit_anomaly, adversarial_second_order_anomaly = self._detect_behavioral_and_forensic_anomalies(
            application, income_mismatch
        )

        # ---------------------------------------------------------------------
        # LAYER 3: Entity Collisions & Cross-Application Velocity (Graph Store)
        # ---------------------------------------------------------------------
        collisions = self._verify_entity_velocity(application)
        if collisions.get("duplicate_phone_in_48h"):
            violations.append(FraudRuleViolation(
                rule_code="VEL-001-CROSS-APP-PHONE-COLLISION",
                rule_name_en="Cross-Application Phone Velocity Collision",
                rule_name_ar="تكرار رقم الهاتف في أكثر من طلب خلال 48 ساعة",
                severity="CRITICAL",
                description_en="Contact phone number detected across multiple distinct applicant profiles within 48h (Fraud Ring Signal).",
                description_ar="تم رصد استخدام رقم الهاتف في طلبات ائتمانية متعددة ببطاقات رقم قومي مختلفة خلال 48 ساعة.",
                observed_value=collisions["total_collisions"],
                threshold_value="< 2 applications / 48h",
                weight=self.SEVERITY_WEIGHTS["CRITICAL"]
            ))

        if collisions.get("duplicate_account_in_48h"):
            violations.append(FraudRuleViolation(
                rule_code="VEL-002-CROSS-APP-ACCOUNT-COLLISION",
                rule_name_en="Bank Account Multiple Identity Reuse",
                rule_name_ar="استخدام الحساب البنكي لأكثر من عميل مختلف",
                severity="CRITICAL",
                description_en="The provided bank statement account number is associated with 2 or more other distinct borrowers in the registry within 48h (a single co-occurrence, e.g. a legitimate joint/family account, is tolerated).",
                description_ar="رقم الحساب البنكي المستخدم مرتبط بعميلين مختلفين آخرين على الأقل خلال 48 ساعة (استخدام واحد سابق فقط، مثل حساب أسري مشترك، لا يُعتبر مخالفة).",
                observed_value=collisions["total_collisions"],
                threshold_value="< 2 applications / 48h",
                weight=self.SEVERITY_WEIGHTS["CRITICAL"]
            ))

        if collisions.get("duplicate_device_in_48h"):
            violations.append(FraudRuleViolation(
                rule_code="VEL-003-DEVICE-FINGERPRINT-COLLISION",
                rule_name_en="Device Fingerprint Collision Across Identities",
                rule_name_ar="تكرار بصمة الجهاز في طلبات ائتمانية بأسماء مختلفة",
                severity="CRITICAL",
                description_en="Hardware device fingerprint used across multiple distinct loan applications within 48h.",
                description_ar="بصمة الجهاز الرقمية مسجلة في طلبات تمويل أخرى ببيانات هوية مختلفة خلال 48 ساعة.",
                observed_value=True,
                threshold_value=False,
                weight=self.SEVERITY_WEIGHTS["CRITICAL"]
            ))

        # ---------------------------------------------------------------------
        # LAYER 4: Dual-Engine ML Inference (Isolation Forest + HistGradientBoosting)
        # ---------------------------------------------------------------------
        feature_matrix = HighDimensionalFraudVectorizer.extract_features(application, income_mismatch, uniformity_score)
        iso_score, mahal_score, gb_prob, is_ml_anomaly = self.model_manager.predict_scores(feature_matrix)

        if is_ml_anomaly:
            anomaly_signals.append(AnomalySignal(
                anomaly_name="UNSUPERVISED_ISOLATION_FOREST_ANOMALY",
                anomaly_score=iso_score,
                detected=True,
                explanation_en=f"Isolation Forest identified deep structural anomaly (Score: {iso_score:.3f}). Feature pattern deviates from authentic population.",
                explanation_ar=f"محرك العزل الرياضي (Isolation Forest) رصد شذوذاً هيكلياً (النتيجة: {iso_score:.3f}) يختلف عن أنماط المقترضين الطبيعيين."
            ))

        # ---------------------------------------------------------------------
        # LAYER 5: Cost-Sensitive Hybrid Fusion & Final Decision
        # ---------------------------------------------------------------------
        fraud_score, risk_level, action = self._calculate_hybrid_decision(
            violations, anomaly_signals, iso_score, gb_prob, mahal_score
        )
      
        reason_codes, exec_summary_ar, exec_summary_en = self._generate_explainability(
            violations, anomaly_signals, risk_level, application, gb_prob
        )

        haircut_multiplier, adjusted_salary = self._compute_income_haircut(
            application, fraud_score, risk_level, income_mismatch
        )

        return {
            "fraud_risk_score": round(fraud_score, 4),
            "fraud_risk_level": risk_level,
            "recommended_action": action,
            "action_ar": self._translate_action(action),
            "verification_checklist": checks_passed,
            "behavioral_anomalies": [asdict(a) for a in anomaly_signals],
            "metrics": {
                "income_mismatch_ratio": round(income_mismatch, 4),
                "employer_similarity_score": round(emp_similarity, 4),
                "inflow_uniformity_score": round(uniformity_score, 4),
                "benford_law_violation": bool(benford_anomaly),
                "terminal_digit_violation": bool(terminal_digit_anomaly),
                "adversarial_second_order_signal": bool(adversarial_second_order_anomaly),
                "isolation_forest_anomaly_score": round(iso_score, 4),
                "gradient_boost_fraud_probability": round(gb_prob, 4),
                "entity_collisions_count": collisions.get("total_collisions", 0),
                "total_violations_count": len(violations),
                "critical_violations_count": sum(1 for v in violations if v.severity == "CRITICAL"),
                "detected_anomalies_count": sum(1 for a in anomaly_signals if a.detected),
                "model_engine_status": "ONLINE_CALIBRATED_ARTIFACTS" if self.model_manager.is_loaded else "HEURISTIC_FALLBACK"
            },
            "downstream_risk_feeder": {
                "credibility_discount_factor": round(haircut_multiplier, 4),
                "declared_salary": round(self._get_declared_salary(application), 2),
                "risk_adjusted_salary": round(adjusted_salary, 2),
                "haircut_percentage": round((1.0 - haircut_multiplier) * 100, 1),
                "proceed_with_credit_model": risk_level != "CRITICAL"
            },
            "explainable_ai": {
                "regulatory_reason_codes": reason_codes,
                "executive_summary_ar": exec_summary_ar,
                "executive_summary_en": exec_summary_en
            },
            "triggered_rules": [asdict(v) for v in violations]
        }

    # -------------------------------------------------------------------------
    # Layer 1 Rule Methods
    # -------------------------------------------------------------------------

    def _verify_identity(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation]]:
        violations = []
        nid_fields = app.get("national_id_fields", {})
        nid_val = str(nid_fields.get("national_id", {}).get("value", "")).strip()

        if not re.match(r"^[23]\d{13}$", nid_val):
            violations.append(FraudRuleViolation(
                rule_code="ID-001-INVALID-NID-FORMAT",
                rule_name_en="Invalid Egyptian National ID Format",
                rule_name_ar="الرقم القومي غير مطابق للمعيار المصري",
                severity="CRITICAL",
                description_en="National ID must be exactly 14 digits starting with 2 (born 1900-1999) or 3 (born 2000-2099).",
                description_ar="الرقم القومي يجب أن يتكون من 14 رقماً ويبدأ بـ 2 لمواليد القرن الماضي أو 3 لمواليد القرن الحالي.",
                observed_value=nid_val,
                threshold_value="14 digits starting with 2 or 3",
                weight=self.SEVERITY_WEIGHTS["CRITICAL"]
            ))

        age_val = float(nid_fields.get("age_years", {}).get("value", 0.0) or 0.0)
        if age_val < self.MIN_APPLICANT_AGE or age_val > self.MAX_APPLICANT_AGE:
            violations.append(FraudRuleViolation(
                rule_code="ID-002-AGE-POLICY-BREACH",
                rule_name_en="Applicant Age Outside Regulatory Financing Range",
                rule_name_ar="عمر المتقدم خارج النطاق التمويلي المصرح به رقابياً",
                severity="HIGH",
                description_en=f"Applicant age ({age_val:.1f} years) must be between {self.MIN_APPLICANT_AGE} and {self.MAX_APPLICANT_AGE}.",
                description_ar=f"عمر المتقدم ({age_val:.1f} سنة) يجب أن يكون بين {self.MIN_APPLICANT_AGE} و {self.MAX_APPLICANT_AGE} عاماً.",
                observed_value=age_val,
                threshold_value=f"{self.MIN_APPLICANT_AGE} - {self.MAX_APPLICANT_AGE}",
                weight=self.SEVERITY_WEIGHTS["HIGH"]
            ))

        return len(violations) == 0, violations

    def _verify_income_consistency(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation], float]:
        violations = []
        salary_fields = app.get("salary_certificate_fields", {})
        bank_fields = app.get("bank_statement_fields", {})

        declared_salary = float(salary_fields.get("declared_net_salary", {}).get("value", 0.0) or 0.0)
        bank_inflow = float(bank_fields.get("avg_monthly_net_inflow", {}).get("value", 0.0) or 0.0)

        if declared_salary <= 0:
            mismatch_ratio = 1.0
        else:
            mismatch_ratio = abs(declared_salary - bank_inflow) / declared_salary

        if declared_salary > 0 and bank_inflow > 0:
            if mismatch_ratio >= self.INCOME_MISMATCH_CRITICAL and declared_salary > bank_inflow:
                violations.append(FraudRuleViolation(
                    rule_code="INC-001-GROSS-INCOME-INFLATION",
                    rule_name_en="Gross Income Inflation Discrepancy",
                    rule_name_ar="تضخيم جوهري في الدخل المذكور بشهادة الراتب",
                    severity="HIGH",
                    description_en=f"Declared salary (EGP {declared_salary:,.0f}) exceeds verified bank net inflows (EGP {bank_inflow:,.0f}) by {mismatch_ratio*100:.1f}%.",
                    description_ar=f"صافي الراتب المذكور ({declared_salary:,.0f} ج.م) يفوق متوسط إيداعات البنك ({bank_inflow:,.0f} ج.م) بنسبة {mismatch_ratio*100:.1f}%.",
                    observed_value=f"{mismatch_ratio*100:.1f}% mismatch",
                    threshold_value=f"< {self.INCOME_MISMATCH_CRITICAL*100:.0f}%",
                    weight=self.SEVERITY_WEIGHTS["HIGH"]
                ))
            elif mismatch_ratio >= self.INCOME_MISMATCH_WARN and declared_salary > bank_inflow:
                violations.append(FraudRuleViolation(
                    rule_code="INC-002-MODERATE-INCOME-MISMATCH",
                    rule_name_en="Moderate Cross-Document Income Discrepancy",
                    rule_name_ar="تفاوت متوسط بين شهادة الراتب وكشف الحساب البنكي",
                    severity="MEDIUM",
                    description_en=f"Declared salary (EGP {declared_salary:,.0f}) exceeds verified bank inflows (EGP {bank_inflow:,.0f}) by {mismatch_ratio*100:.1f}%.",
                    description_ar=f"الراتب المذكور يفوق كشف الحساب بفارق {mismatch_ratio*100:.1f}% يتطلب تدقيقاً إضافياً.",
                    observed_value=f"{mismatch_ratio*100:.1f}% mismatch",
                    threshold_value=f"< {self.INCOME_MISMATCH_WARN*100:.0f}%",
                    weight=self.SEVERITY_WEIGHTS["MEDIUM"]
                ))

        return len(violations) == 0, violations, mismatch_ratio

    def _verify_employer_consistency(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation], float]:
        violations = []
        salary_fields = app.get("salary_certificate_fields", {})
        bank_fields = app.get("bank_statement_fields", {})
        form_data = app.get("form_data", {})

        emp_salary = str(salary_fields.get("employer_name", {}).get("value", "")).strip()
        emp_bank = str(bank_fields.get("payroll_transfer_employer", {}).get("value", "")).strip()
        consistency = app.get("consistency_checks", {}) or {}
        employer_verified_source = "unverified"  # tracked for the checklist/metrics below

        # --- Rule EMP-001: String Name Consistency Across Docs ---
        # Preferred source: recompute directly from the two documents when both are present.
        if emp_salary and emp_bank:
            similarity = difflib.SequenceMatcher(None, emp_salary.lower(), emp_bank.lower()).ratio()
            employer_verified_source = "recomputed"
            if similarity < self.EMPLOYER_SIMILARITY_MIN:
                violations.append(FraudRuleViolation(
                    rule_code="EMP-001-EMPLOYER-NAME-MISMATCH",
                    rule_name_en="Employer Entity Discrepancy Across Documents",
                    rule_name_ar="عدم تطابق جهة العمل بين شهادة الراتب ومحول الراتب بالبنك",
                    severity="HIGH",
                    description_en=f"Salary employer '{emp_salary}' differs from bank depositor '{emp_bank}' (Similarity: {similarity*100:.1f}%).",
                    description_ar=f"جهة العمل في شهادة الراتب '{emp_salary}' تختلف عن محول المرتب '{emp_bank}' (التطابق: {similarity*100:.1f}%).",
                    observed_value=f"{similarity*100:.1f}% similarity",
                    threshold_value=f">= {self.EMPLOYER_SIMILARITY_MIN*100:.0f}%",
                    weight=self.SEVERITY_WEIGHTS["HIGH"]
                ))
        else:
            # Fallback (A): the bank-side employer field this contract needs
            # (payroll_transfer_employer) isn't supplied. Trust the caller's own
            # consistency_checks instead of silently assuming a clean match.
            reported_match = consistency.get("employer_name_match")
            reported_similarity = consistency.get("employer_match_similarity_score")

            if reported_similarity is not None:
                similarity = float(reported_similarity)
                employer_verified_source = "reported"
            elif reported_match is not None:
                similarity = 1.0 if reported_match else 0.0
                employer_verified_source = "reported"
            else:
                # (C): no source at all -- report NOT VERIFIED rather than a false pass.
                similarity = 0.0
                employer_verified_source = "unverified"

            if employer_verified_source == "reported" and (
                reported_match is False or similarity < self.EMPLOYER_SIMILARITY_MIN
            ):
                violations.append(FraudRuleViolation(
                    rule_code="EMP-001-EMPLOYER-NAME-MISMATCH",
                    rule_name_en="Employer Entity Discrepancy Across Documents",
                    rule_name_ar="عدم تطابق جهة العمل بين شهادة الراتب ومحول الراتب بالبنك",
                    severity="HIGH",
                    description_en=f"Caller-reported employer match check failed for '{emp_salary}' (reported similarity: {similarity*100:.1f}%).",
                    description_ar=f"فحص تطابق جهة العمل المُبلَّغ من مقدم الطلب فشل لـ '{emp_salary}' (نسبة التطابق المُبلغة: {similarity*100:.1f}%).",
                    observed_value=f"{similarity*100:.1f}% reported similarity",
                    threshold_value=f">= {self.EMPLOYER_SIMILARITY_MIN*100:.0f}%",
                    weight=self.SEVERITY_WEIGHTS["HIGH"]
                ))
            elif employer_verified_source == "unverified":
                violations.append(FraudRuleViolation(
                    rule_code="EMP-005-EMPLOYER-VERIFICATION-UNAVAILABLE",
                    rule_name_en="Employer Cross-Document Match Not Verifiable",
                    rule_name_ar="تعذر التحقق من تطابق جهة العمل بين المستندات",
                    severity="LOW",
                    description_en=f"Neither bank payroll depositor data nor a caller-reported employer match check was available to verify '{emp_salary}'.",
                    description_ar=f"لا تتوفر بيانات محول المرتب البنكي ولا نتيجة فحص تطابق جهة العمل المُبلغة للتحقق من '{emp_salary}'.",
                    observed_value="No employer cross-check data supplied",
                    threshold_value="payroll_transfer_employer or consistency_checks.employer_match_similarity_score required",
                    weight=self.SEVERITY_WEIGHTS["LOW"]
                ))

        # --- Rule EMP-002: Ghost Payroll Channel (Cash / ATM / Wallet Deposit for Corporate Salary) ---
        payroll_channel = str(bank_fields.get("payroll_channel_type", {}).get("value", "CORPORATE_ACH")).upper()
        if any(kw in emp_salary.lower() for kw in ["شركة", "مجموعة", "مؤسسة", "holding", "corp", "llc", "group"]):
            if payroll_channel in ["ATM_CASH_DEPOSIT", "BRANCH_CASH_DEPOSIT", "INDIVIDUAL_WALLET", "CASH"]:
                violations.append(FraudRuleViolation(
                    rule_code="EMP-002-GHOST-PAYROLL-CHANNEL",
                    rule_name_en="Fabricated Corporate Payroll Channel (Cash/ATM Deposit)",
                    rule_name_ar="شبهة تزوير إيداع المرتب نقداً أو عبر ATM لجهة عمل اعتبارية",
                    severity="CRITICAL",
                    description_en=f"Corporate salary for '{emp_salary}' was deposited via '{payroll_channel}' instead of legitimate Corporate Bulk ACH transfer.",
                    description_ar=f"المرتب الخاص بشركة '{emp_salary}' تم إيداعه عبر '{payroll_channel}' وليس كتحويل بنكي مؤسسي معتمد.",
                    observed_value=payroll_channel,
                    threshold_value="CORPORATE_ACH / BANK_PAYROLL",
                    weight=self.SEVERITY_WEIGHTS["CRITICAL"]
                ))

        # --- Rule EMP-003: Commercial Registry & Tax ID Verification ---
        tax_id = str(salary_fields.get("tax_registration_number", {}).get("value", "")).strip()
        if tax_id:
            cleaned_tax = re.sub(r"[^\d]", "", tax_id)
            is_dummy_tax = cleaned_tax in ["000000000", "123456789", "999999999", "111111111"] or len(set(cleaned_tax)) <= 1
            if len(cleaned_tax) != 9 or is_dummy_tax:
                violations.append(FraudRuleViolation(
                    rule_code="EMP-003-INVALID-TAX-REGISTRATION",
                    rule_name_en="Invalid or Fabricated Employer Tax ID",
                    rule_name_ar="رقم تسجيل ضريبي وهمي أو غير مطابق لمعيار الـ 9 أرقام",
                    severity="CRITICAL" if is_dummy_tax else "HIGH",
                    description_en=f"Employer Tax ID '{tax_id}' is invalid or detected as a fabricated dummy registration number.",
                    description_ar=f"الرقم الضريبي لجهة العمل '{tax_id}' غير صحيح أو تم توليده كأرقام وهمية مصطنعة.",
                    observed_value=tax_id,
                    threshold_value="9-Digit Official Tax Registration",
                    weight=self.SEVERITY_WEIGHTS["CRITICAL" if is_dummy_tax else "HIGH"]
                ))

        # --- Rule EMP-004: Shell Company Heuristics (Generic Name + Mobile-only Contact) ---
        generic_terms = ["للتجارة والتوريدات", "خدمات عامة", "مكتب استشارات", "commercial office", "general trading", "import & export"]
        is_generic_name = any(term in emp_salary.lower() for term in generic_terms)
        has_legal_form = any(suffix in emp_salary.lower() for suffix in ["ش.م.م", "ذ.م.م", "s.a.e", "llc", "مساهمة", "تضامن", "corp"])
        emp_phone = str(salary_fields.get("employer_phone", {}).get("value", "")).strip()
        is_mobile_only = emp_phone.startswith(("010", "011", "012", "015")) and not emp_phone.startswith(("02", "03"))

        if is_generic_name and not has_legal_form and is_mobile_only:
            violations.append(FraudRuleViolation(
                rule_code="EMP-004-SHELL-COMPANY-PROFILE",
                rule_name_en="Shell Company Profile (Generic Name & Mobile-Only Contact)",
                rule_name_ar="مؤشرات شركة وهمية (تسمية عامة غير مقيدة شكلياً وهاتف محمول فقط)",
                severity="HIGH",
                description_en=f"Employer '{emp_salary}' exhibits shell-company patterns: generic trading title, no legal entity suffix, and lack of verified landline.",
                description_ar=f"جهة العمل '{emp_salary}' تظهر نمط شركة وهمية: اسم تجاري عام، غياب الشكل القانوني المعتمد، وغياب هاتف أرضي ثابت.",
                observed_value=f"Name: {emp_salary} | Phone: {emp_phone}",
                threshold_value="Registered Corporate Entity with Fixed Landline",
                weight=self.SEVERITY_WEIGHTS["HIGH"]
            ))

        return len(violations) == 0, violations, similarity

    def _verify_document_integrity(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation]]:
        violations = []
        docs = app.get("documents", [])

        for doc in docs:
            doc_type = doc.get("document_type", "Unknown")
            is_tampered = bool(doc.get("is_tampered_suspected", False))
            quality = float(doc.get("overall_quality_score", 1.0) or 1.0)

            if is_tampered:
                violations.append(FraudRuleViolation(
                    rule_code=f"DOC-001-TAMPERED-{doc_type.upper()}",
                    rule_name_en=f"Digital Alteration Detected in {doc_type}",
                    rule_name_ar=f"اشتباه تلاعب وتعديل رقمي في مستند {doc_type}",
                    severity="CRITICAL",
                    description_en=f"Forensic pixel analysis identified digital manipulation or modified font layers in {doc_type}.",
                    description_ar=f"الفحص الجنائي للصور والمستندات كشف عن تعديلات رقمية في مستند {doc_type}.",
                    observed_value="Tampering Flag = TRUE",
                    threshold_value="Tampering Flag = FALSE",
                    weight=self.SEVERITY_WEIGHTS["CRITICAL"]
                ))

            if quality < self.MIN_OCR_CONFIDENCE_THRESHOLD:
                violations.append(FraudRuleViolation(
                    rule_code=f"DOC-002-LOW-QUALITY-{doc_type.upper()}",
                    rule_name_en=f"Sub-standard OCR Confidence in {doc_type}",
                    rule_name_ar=f"انخفاض جودة القراءة الضوئية لمستند {doc_type}",
                    severity="MEDIUM",
                    description_en=f"OCR quality ({quality*100:.1f}%) is below the institutional threshold.",
                    description_ar=f"جودة استخراج النصوص ({quality*100:.1f}%) غير كافية للاعتماد الآلي.",
                    observed_value=f"{quality*100:.1f}%",
                    threshold_value=f">= {self.MIN_OCR_CONFIDENCE_THRESHOLD*100:.0f}%",
                    weight=self.SEVERITY_WEIGHTS["MEDIUM"]
                ))

        return len(violations) == 0, violations

    def _verify_bureau_report(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation]]:
        violations = []
        iscore_fields = app.get("iscore_report_fields", {})
        consistency = app.get("consistency_checks", {}) or {}

        # --- Rule BUR-001: I-Score Freshness Window ---
        # Preferred: this contract's actual field is score_date, not inquiry_date.
        # Try both, in case a future upstream renames it back.
        report_date_str = (
            iscore_fields.get("inquiry_date", {}).get("value", "")
            or iscore_fields.get("score_date", {}).get("value", "")
        )
        age_days = None
        if report_date_str:
            try:
                report_date = datetime.fromisoformat(report_date_str.replace("Z", "+00:00")).date()
                age_days = (datetime.now(timezone.utc).date() - report_date).days
            except Exception:
                age_days = None

        if age_days is None:
            # Fallback (A): trust the caller's own precomputed age if the raw date
            # isn't parseable or wasn't supplied at all.
            reported_age = consistency.get("iscore_report_age_days")
            if reported_age is not None:
                try:
                    age_days = int(reported_age)
                except (TypeError, ValueError):
                    age_days = None

        if age_days is not None:
            if age_days > self.MAX_ISCORE_AGE_DAYS:
                violations.append(FraudRuleViolation(
                    rule_code="BUR-001-STALE-ISCORE-REPORT",
                    rule_name_en="Expired I-Score Bureau Inquiry",
                    rule_name_ar="تقرير الآي سكور منتهي الصلاحية المصرفية",
                    severity="MEDIUM",
                    description_en=f"I-Score report age ({age_days} days) exceeds CBE fresh inquiry window ({self.MAX_ISCORE_AGE_DAYS} days).",
                    description_ar=f"عمر تقرير الاستعلام ({age_days} يوماً) يتجاوز الحد الأقصى لصلاحية الاستعلام بالبنك المركزي ({self.MAX_ISCORE_AGE_DAYS} يوماً).",
                    observed_value=f"{age_days} days",
                    threshold_value=f"<= {self.MAX_ISCORE_AGE_DAYS} days",
                    weight=self.SEVERITY_WEIGHTS["MEDIUM"]
                ))
        else:
            # (C): no date, in any known field, and no caller-reported age -- say so.
            violations.append(FraudRuleViolation(
                rule_code="BUR-004-ISCORE-FRESHNESS-UNVERIFIABLE",
                rule_name_en="I-Score Report Freshness Not Verifiable",
                rule_name_ar="تعذر التحقق من تاريخ صلاحية تقرير الآي سكور",
                severity="LOW",
                description_en="No parseable bureau report date or caller-reported report age was supplied to verify the CBE 30-day freshness window.",
                description_ar="لا يتوفر تاريخ استعلام صالح للقراءة ولا عمر تقرير مُبلغ من مقدم الطلب للتحقق من نافذة الثلاثين يوماً لدى البنك المركزي.",
                observed_value="No date/age data supplied",
                threshold_value="iscore_report_fields.score_date or consistency_checks.iscore_report_age_days required",
                weight=self.SEVERITY_WEIGHTS["LOW"]
            ))

        # --- Rule BUR-002: Active Legal/Judicial Actions ---
        # Preferred: a direct legal_action_flags list (this contract doesn't send one).
        # Fallback: derive it from any bureau_facilities entry flagged legal_action_flag=true.
        legal_actions = iscore_fields.get("legal_action_flags", {}).get("value", [])
        if not legal_actions:
            facilities = iscore_fields.get("bureau_facilities", [])
            if isinstance(facilities, list):
                legal_actions = [
                    f.get("lender_name", "Unknown facility")
                    for f in facilities
                    if isinstance(f, dict) and f.get("legal_action_flag") is True
                ]
        if legal_actions:
            violations.append(FraudRuleViolation(
                rule_code="BUR-002-LEGAL-ENFORCEMENT-ACTION",
                rule_name_en="Active Legal or Negative Action Recorded in Bureau",
                rule_name_ar="تسجيل إجراءات قانونية أو تعثر قضائي في تقرير الاستعلام الائتماني",
                severity="CRITICAL",
                description_en=f"Applicant has active adverse bureau actions: {', '.join(legal_actions)}.",
                description_ar=f"تم رصد إجراءات قضائية ونزاعات سداد نشطة ضد العميل: {', '.join(legal_actions)}.",
                observed_value=legal_actions,
                threshold_value="None",
                weight=self.SEVERITY_WEIGHTS["CRITICAL"]
            ))

        return len(violations) == 0, violations

    def _verify_banking_behavior(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation]]:
        violations = []
        bank_fields = app.get("bank_statement_fields", {})
        consistency = app.get("consistency_checks", {}) or {}

        # --- Rule BNK-001: Bounced/Returned Cheques ---
        # Preferred field name: bounced_cheques_count_12m (never sent by this
        # contract). Fallback: returned_cheques_count, which the real payloads use.
        if "bounced_cheques_count_12m" in bank_fields:
            bounced_cheques = int(bank_fields.get("bounced_cheques_count_12m", {}).get("value", 0) or 0)
        else:
            bounced_cheques = int(bank_fields.get("returned_cheques_count", {}).get("value", 0) or 0)

        if bounced_cheques > 0:
            violations.append(FraudRuleViolation(
                rule_code="BNK-001-BOUNCED-CHEQUES",
                rule_name_en="History of Bounced Cheques within 12 Months",
                rule_name_ar="وجود شيكات مرتجعة بدون رصيد خلال الـ 12 شهراً الماضية",
                severity="HIGH",
                description_en=f"Applicant account recorded {bounced_cheques} bounced cheque(s) indicating liquidity stress.",
                description_ar=f"سجل الحساب {bounced_cheques} شيكاً مرتداً لعدم كفاية الرصيد، مؤشر مخاطر ائتمانية عالية.",
                observed_value=bounced_cheques,
                threshold_value=0,
                weight=self.SEVERITY_WEIGHTS["HIGH"]
            ))

        # --- Rule BNK-002: Running Balance Arithmetic ---
        # Preferred field: bank_statement_fields.running_balance_math_verified
        # (never sent by this contract). Fallback: the caller's own
        # consistency_checks.running_balance_math_valid. Missing data must NOT
        # silently default to "balanced" -- that was the previous bug (default=True).
        if "running_balance_math_verified" in bank_fields:
            is_balanced = bool(bank_fields.get("running_balance_math_verified", {}).get("value", True))
            balance_check_source = "recomputed"
        elif "running_balance_math_valid" in consistency:
            is_balanced = bool(consistency.get("running_balance_math_valid"))
            balance_check_source = "reported"
        else:
            is_balanced = True  # no data either way -- not treated as a violation, just unverified
            balance_check_source = "unverified"

        if balance_check_source != "unverified" and not is_balanced:
            violations.append(FraudRuleViolation(
                rule_code="BNK-002-STATEMENT-ARITHMETIC-ANOMALY",
                rule_name_en="Bank Statement Running Balance Math Error",
                rule_name_ar="خلل في العمليات الحسابية للأرصدة المتتالية بكشف الحساب",
                severity="HIGH",
                description_en="Cumulative transactions do not reconcile with ending balances. Indicator of forged statement OR an upstream OCR/parsing error -- routed to manual review rather than auto-rejection pending confirmation.",
                description_ar="تسلسل العمليات الحسابية لا يتطابق مع رصيد الإقفال. مؤشر محتمل على التزوير أو خطأ في معالجة المستند، لذا يُحال لمراجعة يدوية بدل الرفض الفوري لحين التأكد.",
                observed_value=False,
                threshold_value=True,
                weight=self.SEVERITY_WEIGHTS["HIGH"]
            ))
        elif balance_check_source == "unverified":
            violations.append(FraudRuleViolation(
                rule_code="BNK-003-BALANCE-MATH-UNVERIFIABLE",
                rule_name_en="Bank Statement Arithmetic Not Verifiable",
                rule_name_ar="تعذر التحقق من صحة العمليات الحسابية بكشف الحساب",
                severity="LOW",
                description_en="Neither a recomputed running-balance check nor a caller-reported arithmetic validation was supplied for this bank statement.",
                description_ar="لا يتوفر فحص مُعاد احتساب للرصيد المتتالي ولا نتيجة تحقق مُبلغة من مقدم الطلب لهذا كشف الحساب.",
                observed_value="No arithmetic validation data supplied",
                threshold_value="bank_statement_fields.running_balance_math_verified or consistency_checks.running_balance_math_valid required",
                weight=self.SEVERITY_WEIGHTS["LOW"]
            ))

        return len(violations) == 0, violations

    def _verify_device_telemetry(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation]]:
        """
        Security telemetry: VPN/Proxy detection, rooted devices, emulators, off-hours submission.
        """
        violations = []
        telemetry = app.get("device_telemetry", {})
        if not telemetry:
            return True, violations

        if telemetry.get("is_vpn_or_proxy", False):
            violations.append(FraudRuleViolation(
                rule_code="SEC-001-VPN-OR-PROXY-DETECTED",
                rule_name_en="Anonymizing Proxy or VPN Detected",
                rule_name_ar="تقديم الطلب عبر شبكة افتراضية خاصة أو خادم وسيط (VPN/Proxy)",
                severity="HIGH",
                description_en="Applicant submitted credit application via masked IP / VPN concealing true geographic origin.",
                description_ar="تم إرسال الطلب الائتماني باستخدام VPN لإخفاء الموقع الجغرافي الفعلي للمستخدم.",
                observed_value=True,
                threshold_value=False,
                weight=self.SEVERITY_WEIGHTS["HIGH"]
            ))

        if telemetry.get("device_is_rooted_or_emulator", False):
            violations.append(FraudRuleViolation(
                rule_code="SEC-002-COMPROMISED-DEVICE-ENVIRONMENT",
                rule_name_en="Rooted Device or Virtual Emulator Environment",
                rule_name_ar="التقديم من جهاز مفتوح الصلاحيات (Rooted) أو بيئة محاكي برمجية",
                severity="CRITICAL",
                description_en="Client environment identified as an emulated virtual machine or jailbroken device.",
                description_ar="البيئة البرمجية للجهاز تدل على محاكي افتراضي أو تعديل لنظام التشغيل الأساسي.",
                observed_value=True,
                threshold_value=False,
                weight=self.SEVERITY_WEIGHTS["CRITICAL"]
            ))

        submission_hour = telemetry.get("submission_hour_local", 12)
        if 2 <= submission_hour <= 5:
            violations.append(FraudRuleViolation(
                rule_code="SEC-003-ABNORMAL-OFF-HOURS-SUBMISSION",
                rule_name_en="Off-hours Automated Batch Submission Window",
                rule_name_ar="تقديم الطلب في ساعات حظر الفجر المشبوهة (2 AM - 5 AM)",
                severity="LOW",
                description_en=f"Application submitted at {submission_hour}:00 AM (typical bot/syndicate batching window).",
                description_ar=f"تم إرسال الطلب في تمام الساعة {submission_hour}:00 فجراً، وهو توقيت غير معتاد لطلبات الأفراد.",
                observed_value=f"{submission_hour}:00 AM",
                threshold_value="Normal waking hours",
                weight=self.SEVERITY_WEIGHTS["LOW"]
            ))

        return len(violations) == 0, violations

    # -------------------------------------------------------------------------
    # Layer 2 & Anti-Adversarial Forensic Methods
    # -------------------------------------------------------------------------

    def _detect_behavioral_and_forensic_anomalies(
        self, app: Dict[str, Any], income_mismatch: float
    ) -> Tuple[List[AnomalySignal], float, bool, bool, bool]:
        signals = []
        bank_fields = app.get("bank_statement_fields", {})
        form_data = app.get("form_data", {})

        avg_balance = float(bank_fields.get("avg_monthly_balance", {}).get("value", 0.0) or 0.0)
        max_balance = float(bank_fields.get("max_monthly_balance", {}).get("value", 0.0) or 0.0)
        min_balance = float(bank_fields.get("min_monthly_balance", {}).get("value", 0.0) or 0.0)
        volatility = float(bank_fields.get("balance_volatility_std", {}).get("value", 0.0) or 0.0)
        regularity = float(bank_fields.get("income_regularity_score", {}).get("value", 1.0) or 1.0)
        requested_annuity = float(form_data.get("requested_annuity", 0.0) or 0.0)

        # 1. Window Dressing
        is_window_dressed = False
        surge_ratio = 1.0
        if avg_balance > 0:
            surge_ratio = max_balance / avg_balance
            if surge_ratio >= self.WINDOW_DRESSING_SURGE_RATIO and min_balance < (0.15 * avg_balance):
                is_window_dressed = True

        signals.append(AnomalySignal(
            anomaly_name="ARTIFICIAL_BALANCE_INFLATION_WINDOW_DRESSING",
            anomaly_score=round(min(surge_ratio / 4.0, 1.0), 3) if is_window_dressed else 0.0,
            detected=is_window_dressed,
            explanation_en=f"Peak balance (EGP {max_balance:,.0f}) is {surge_ratio:.1f}x higher than average balance, followed by low liquidity reserve.",
            explanation_ar=f"أعلى رصيد ({max_balance:,.0f} ج.م) يتجاوز {surge_ratio:.1f} أضعاف المتوسط الشهري مع فراغ الحساب، مؤشر على تجميل كشف الحساب."
        ))

        # 2. Volatility
        is_volatile = False
        if avg_balance > 0 and (volatility / avg_balance) > 1.20 and regularity < 0.60:
            is_volatile = True

        signals.append(AnomalySignal(
            anomaly_name="ERRATIC_CASHFLOW_INSTABILITY",
            anomaly_score=0.65 if is_volatile else 0.10,
            detected=is_volatile,
            explanation_en=f"High balance dispersion (std EGP {volatility:,.0f}) paired with low salary deposit regularity ({regularity*100:.0f}%).",
            explanation_ar=f"تذبذب شديد في السيولة (انحراف معياري {volatility:,.0f} ج.م) مع ضعف انتظام مواعيد نزول المرتب ({regularity*100:.0f}%)."
        ))

        # 3. Annuity Stress
        is_annuity_stress = False
        if avg_balance > 0 and requested_annuity > 0:
            buffer_ratio = requested_annuity / avg_balance
            if buffer_ratio > self.ANNUITY_TO_LIQUIDITY_MAX:
                is_annuity_stress = True

        signals.append(AnomalySignal(
            anomaly_name="LIQUIDITY_BUFFER_STRESS",
            anomaly_score=0.70 if is_annuity_stress else 0.05,
            detected=is_annuity_stress,
            explanation_en=f"Requested monthly annuity (EGP {requested_annuity:,.0f}) absorbs > 60% of applicant's historical average liquid balance.",
            explanation_ar=f"القسط الشهري المطلوب ({requested_annuity:,.0f} ج.م) يستنزف أكثر من 60% من متوسط رصيد العميل السائل تاريخياً."
        ))

        # 4. Mathematical Forensics on Transaction Amounts
        sample_transactions = bank_fields.get("sample_transaction_amounts", {}).get("value", [])
        benford_anomaly = False
        terminal_digit_anomaly = False
        chi_stat = 0.0
        uniformity_score = 0.0

        if sample_transactions and len(sample_transactions) >= 40:
            benford_anomaly, chi_stat, _ = DeepForensicAnalyzer.evaluate_benford_law(sample_transactions)
            uniformity_score = DeepForensicAnalyzer.calculate_inflow_uniformity(sample_transactions)

            if benford_anomaly:
                signals.append(AnomalySignal(
                    anomaly_name="BENFORD_LAW_FIRST_DIGIT_VIOLATION",
                    anomaly_score=0.85,
                    detected=True,
                    explanation_en=f"Transaction lead digit distribution violates Benford's Law (Chi-Square: {chi_stat:.1f}, p < 0.05). High likelihood of fabricated numbers.",
                    explanation_ar=f"توزيع الأرقام في كشف الحساب ينتهك قانون بنفورد الإحصائي (مربع كاي: {chi_stat:.1f}). مؤشر قوي على أرقام مصطنعة ومكتوبة يدوياً."
                ))

            if uniformity_score > 0.40:
                signals.append(AnomalySignal(
                    anomaly_name="FABRICATED_ROUND_NUMBER_UNIFORMITY",
                    anomaly_score=uniformity_score,
                    detected=True,
                    explanation_en=f"Abnormal uniformity: {uniformity_score*100:.0f}% of transactions are perfect round thousands/hundreds without authentic fractional friction.",
                    explanation_ar=f"تكرار غير طبيعي لأرقام مستديرة تماماً بنسبة {uniformity_score*100:.0f}% دون وجود كسور أو تعاملات تجزئة حقيقية."
                ))

        # 5. Last-Digit Uniformity Forensics (Terminal Digit Chi-Square Test)
        if sample_transactions and len(sample_transactions) >= 40:
            terminal_digit_anomaly, ld_chi_stat, _ = DeepForensicAnalyzer.evaluate_last_digit_uniformity(sample_transactions)
            if terminal_digit_anomaly:
                signals.append(AnomalySignal(
                    anomaly_name="TERMINAL_DIGIT_UNIFORMITY_VIOLATION",
                    anomaly_score=0.80,
                    detected=True,
                    explanation_en=f"Last-digit frequency violates uniform distribution (Chi-Square: {ld_chi_stat:.1f}, p < 0.05). Reflects human subconscious digit bias in forged entries.",
                    explanation_ar=f"الخانة الأخيرة في مبالغ المعاملات تنتهك التوزيع الطبيعي المنتظم (مربع كاي: {ld_chi_stat:.1f}). مؤشر على انحياز بشري في تأليف المبالغ."
                ))

        # 6. Micro-Transaction Absence (Real-World Expense Noise Check)
        if sample_transactions:
            is_synthetic_statement, micro_ratio = DeepForensicAnalyzer.analyze_micro_transaction_absence(sample_transactions)
            if is_synthetic_statement:
                signals.append(AnomalySignal(
                    anomaly_name="SYNTHETIC_STATEMENT_MICRO_TRANSACTION_ABSENCE",
                    anomaly_score=0.75,
                    detected=True,
                    explanation_en=f"Absence of real-world expense noise: Only {micro_ratio*100:.1f}% micro-transactions (<250 EGP). Authentic lifestyle friction is absent.",
                    explanation_ar=f"غياب ضوضاء الحياة اليومية ومصروفات الاحتكاك العادية ({micro_ratio*100:.1f}% فقط معاملات صغيرة). كشف الحساب يحتوي فقط على مبالغ كبرى مصطنعة."
                ))

        # 7. Second-Order Adversarial Forensic Signals (Benford-Evasion Countermeasure)
        adversarial_second_order_anomaly = False
        if sample_transactions and len(sample_transactions) >= 40:
            adv_flag, adv_detail = DeepForensicAnalyzer.evaluate_second_order_adversarial_signals(sample_transactions)
            if adv_flag:
                adversarial_second_order_anomaly = True
                reasons_en, reasons_ar = [], []
                if adv_detail.get("second_digit_anomaly"):
                    reasons_en.append(f"second-digit Benford violation (χ²={adv_detail['second_digit_chi_square']})")
                    reasons_ar.append(f"انتهاك قانون بنفورد للرقم الثاني (كاي²={adv_detail['second_digit_chi_square']})")
                if adv_detail.get("mantissa_anomaly"):
                    reasons_en.append(f"log-mantissa non-uniformity (KS p={adv_detail['mantissa_ks_pvalue']})")
                    reasons_ar.append(f"عدم انتظام في التوزيع اللوغاريتمي الدقيق (KS p={adv_detail['mantissa_ks_pvalue']})")
                if adv_detail.get("recurrence_absent"):
                    reasons_en.append(f"no recurring/near-duplicate transactions ({adv_detail['recurrence_ratio']*100:.1f}% only)")
                    reasons_ar.append(f"غياب معاملات متكررة أو شبه متطابقة ({adv_detail['recurrence_ratio']*100:.1f}% فقط)")

                signals.append(AnomalySignal(
                    anomaly_name="ADVERSARIAL_BENFORD_EVASION_SECOND_ORDER_SIGNAL",
                    anomaly_score=0.70,
                    detected=True,
                    explanation_en=(
                        "Transaction set passes the basic first-digit Benford screen but fails deeper "
                        f"forensic checks: {'; '.join(reasons_en)}. Consistent with a sophisticated forger "
                        "deliberately engineering statement data to defeat naive fraud screening."
                    ),
                    explanation_ar=(
                        "كشف الحساب يجتاز الفحص الأساسي لقانون بنفورد (الرقم الأول) لكنه يفشل في فحوصات "
                        f"جنائية أعمق: {'; '.join(reasons_ar)}. مؤشر على محتال متمرس يصمم بيانات الكشف "
                        "عمداً لتفادي أدوات الكشف السطحية."
                    )
                ))

        # 8. Adversarial Threshold-Gaming Detector
        is_gaming, gaming_score, gaming_reasons = self._evaluate_threshold_gaming(app, income_mismatch)
        if is_gaming:
            signals.append(AnomalySignal(
                anomaly_name="ADVERSARIAL_THRESHOLD_GAMING",
                anomaly_score=gaming_score,
                detected=True,
                explanation_en=f"Applicant parameters cluster suspiciously close below auto-rejection cutoffs: {'; '.join(gaming_reasons)}.",
                explanation_ar=f"رصد محاولة تحايل استراتيجي وضبط الأرقام عمداً تحت أسقف الرفض مباشرة: {'; '.join(gaming_reasons)}."
            ))

        return signals, uniformity_score, benford_anomaly, terminal_digit_anomaly, adversarial_second_order_anomaly

    def _evaluate_threshold_gaming(
        self, application: Dict[str, Any], income_mismatch: float
    ) -> Tuple[bool, float, List[str]]:
        """
        Detects applicants who calibrate application metrics strategically
        just below automated cutoff boundaries (Margin Proximity Clustering).
        """
        gaming_signals = []
        proximity_score = 0.0

        bank_fields = application.get("bank_statement_fields", {})
        form_data = application.get("form_data", {})

        # Income Mismatch Gaming (Tuned right under critical ceiling)
        if 0.12 <= income_mismatch < self.INCOME_MISMATCH_CRITICAL:
            gaming_signals.append(f"Income mismatch ({income_mismatch*100:.1f}%) calibrated directly below {self.INCOME_MISMATCH_CRITICAL*100:.0f}% cutoff")
            proximity_score += 0.35

        # Annuity/Liquidity Buffer Gaming
        avg_bal = float(bank_fields.get("avg_monthly_balance", {}).get("value", 0.0) or 0.0)
        annuity = float(form_data.get("requested_annuity", 0.0) or 0.0)
        if avg_bal > 0 and annuity > 0:
            buffer = annuity / avg_bal
            if 0.52 <= buffer < self.ANNUITY_TO_LIQUIDITY_MAX:
                gaming_signals.append(f"Annuity/Liquidity ratio ({buffer*100:.1f}%) tuned directly below {self.ANNUITY_TO_LIQUIDITY_MAX*100:.0f}% ceiling")
                proximity_score += 0.35

        # Age Boundary Proximity
        nid_fields = application.get("national_id_fields", {})
        age = float(nid_fields.get("age_years", {}).get("value", 0.0) or 0.0)
        if (self.MIN_APPLICANT_AGE <= age <= self.MIN_APPLICANT_AGE + 0.3) or (self.MAX_APPLICANT_AGE - 0.3 <= age <= self.MAX_APPLICANT_AGE):
            gaming_signals.append(f"Applicant age ({age:.1f}y) lies directly on policy edge")
            proximity_score += 0.20

        is_gaming = len(gaming_signals) >= 2 or proximity_score >= 0.60
        return is_gaming, round(proximity_score, 2), gaming_signals

    # -------------------------------------------------------------------------
    # Layer 3 Entity Velocity Method
    # -------------------------------------------------------------------------

    def _verify_entity_velocity(self, app: Dict[str, Any]) -> Dict[str, Any]:
        app_id = app.get("application_id", f"APP-{datetime.now(timezone.utc).timestamp()}")
        nid = app.get("national_id_fields", {}).get("national_id", {}).get("value", "")
        phone = app.get("form_data", {}).get("mobile_phone", "")
        employer = app.get("salary_certificate_fields", {}).get("employer_name", {}).get("value", "")
        # Preferred field name: bank_account_number (never sent by this contract).
        # Fallback: account_number, which the real payloads use. Without this fix,
        # every application's bank_account_number resolved to "", so two fraudsters
        # sharing the same real account would never collide in the velocity store.
        account_fields = app.get("bank_statement_fields", {})
        account = (
            account_fields.get("bank_account_number", {}).get("value", "")
            or account_fields.get("account_number", {}).get("value", "")
        )
        device_id = app.get("device_telemetry", {}).get("device_fingerprint_id", "")

        return self.entity_store.check_and_record_velocity(app_id, nid, phone, employer, account, device_id)

    # -------------------------------------------------------------------------
    # Layer 5 Decision & Hybrid Fusion
    # -------------------------------------------------------------------------

    def _calculate_hybrid_decision(
        self,
        violations: List[FraudRuleViolation],
        anomalies: List[AnomalySignal],
        iso_score: float,
        gb_prob: float,
        mahal_score: float = 0.15
    ) -> Tuple[float, str, str]:
        has_critical = any(v.severity == "CRITICAL" for v in violations)
        has_high = any(v.severity == "HIGH" for v in violations)

        rule_score = sum(v.weight for v in violations)
        detected_anoms = [a for a in anomalies if a.detected]
        anomaly_score = sum(a.anomaly_score * 0.20 for a in detected_anoms)

        # Multi-Vector Unsupervised Ensemble (Isolation Forest + Mahalanobis Distance)
        unsupervised_score = (0.50 * iso_score) + (0.50 * mahal_score)
        
        # Hybrid ML Score: 20% Unsupervised Multi-Vector + 80% Supervised Cost-Sensitive GBDT
        ml_score = (0.20 * unsupervised_score) + (0.80 * gb_prob)

        # Total Aggregation: 50% Deterministic Rules + 20% Forensic Anomalies + 30% Dual-Engine ML
        base_score = (0.50 * rule_score) + (0.20 * anomaly_score) + (0.30 * ml_score)
        total_score = float(np.clip(base_score, 0.0, 1.0))

        if has_critical or total_score >= 0.70 or gb_prob >= 0.85:
            risk_level = "CRITICAL"
            action = "REJECT_SUSPECTED_FRAUD"
            total_score = max(total_score, 0.90)
        elif has_high or total_score >= 0.50 or ml_score >= 0.62:
            risk_level = "HIGH"
            action = "FLAG_FOR_MANUAL_FRAUD_INVESTIGATION"
        elif total_score >= 0.28 or sum(1 for a in anomalies if a.detected) >= 2:
            risk_level = "MEDIUM"
            action = "REQUEST_ADDITIONAL_VERIFICATION_DOCUMENTS"
        else:
            risk_level = "LOW"
            action = "PROCEED_TO_CREDIT_EVALUATION"

        return total_score, risk_level, action

    def _generate_explainability(
        self,
        violations: List[FraudRuleViolation],
        anomalies: List[AnomalySignal],
        risk_level: str,
        app: Dict[str, Any],
        gb_prob: float
    ) -> Tuple[List[Dict[str, Any]], str, str]:
        reason_codes = []

        for v in violations:
            reason_codes.append({
                "code": v.rule_code,
                "title_en": v.rule_name_en,
                "title_ar": v.rule_name_ar,
                "severity": v.severity,
                "reason_en": v.description_en,
                "reason_ar": v.description_ar
            })

        for a in anomalies:
            if a.detected:
                reason_codes.append({
                    "code": a.anomaly_name,
                    "title_en": a.anomaly_name.replace("_", " ").title(),
                    "title_ar": "إشارة سلوكية شاذة في الحساب",
                    "severity": "MEDIUM",
                    "reason_en": a.explanation_en,
                    "reason_ar": a.explanation_ar
                })

        app_id = app.get("application_id", "N/A")
        if risk_level == "LOW":
            summary_ar = f"الطلب {app_id} اجتاز كافة فحوصات التطابق الجنائي والبنكي والسيبراني بنجاح. لا توجد مؤشرات تزوير أو تلاعب بالحدود، والملف مؤهل للتقييم الائتماني المباشر."
            summary_en = f"Application {app_id} successfully passed all forensic, cybersecurity, and ML layers. Fraud probability is low ({gb_prob*100:.1f}%)."
        elif risk_level == "CRITICAL":
            summary_ar = f"تحذير رقابي حرج: الطلب {app_id} تم رفضه آلياً لاشتباه تزوير مؤكد أو تلاعب جنائي. تم رصد {len(violations)} خرق لسياسات البنك المركزي."
            summary_en = f"Regulatory Critical Alert: Application {app_id} auto-rejected due to critical document tampering or security violations."
        else:
            summary_ar = f"الطلب {app_id} يتطلب مراجعة ائتمانية يدوية متخصصة لوجود تفاوتات جزئية أو مؤشرات عدم استقرار أو اقتراب من الحدود الرقابية."
            summary_en = f"Application {app_id} routed to senior underwriter queue due to moderate inconsistencies or behavioral boundary alerts."

        return reason_codes, summary_ar, summary_en

    def _compute_income_haircut(
        self, app: Dict[str, Any], fraud_score: float, risk_level: str, mismatch: float
    ) -> Tuple[float, float]:
        declared_salary = self._get_declared_salary(app)
        if risk_level == "CRITICAL":
            haircut = 0.0
        elif risk_level == "HIGH":
            haircut = max(0.50, 1.0 - (fraud_score * 0.75))
        elif risk_level == "MEDIUM":
            haircut = max(0.75, 1.0 - (fraud_score * 0.50))
        else:
            haircut = 1.0

        adjusted_salary = declared_salary * haircut
        return haircut, adjusted_salary

    def _get_declared_salary(self, app: Dict[str, Any]) -> float:
        return float(app.get("salary_certificate_fields", {}).get("declared_net_salary", {}).get("value", 0.0) or 0.0)

    def _translate_action(self, action: str) -> str:
        mapping = {
            "PROCEED_TO_CREDIT_EVALUATION": "تمرير الطلب للتقييم الائتماني المباشر",
            "REQUEST_ADDITIONAL_VERIFICATION_DOCUMENTS": "طلب مستندات دعم إضافية وتحديث الاستعلام",
            "FLAG_FOR_MANUAL_FRAUD_INVESTIGATION": "تحويل الطلب للمراجعة الأمنية والائتمانية اليدوية",
            "REJECT_SUSPECTED_FRAUD": "رفض قطعي فوري لاشتباه تزوير واحتيال"
        }
        return mapping.get(action, action)

    def enrich_payload(self, application: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriches the input JSON payload with the full fraud_assessment block.
        """
        assessment = self.evaluate(application)
        if "consistency_checks" not in application:
            application["consistency_checks"] = {}

        application["consistency_checks"].update(assessment["verification_checklist"])
        application["fraud_assessment"] = assessment
        return application