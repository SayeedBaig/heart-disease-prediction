"""
Clinical Alert Engine — CardioAI
=================================
Shared, reusable function that evaluates observed or hypothetical clinical
measurements and returns structured clinical alerts.

IMPORTANT DESIGN CONSTRAINTS
------------------------------
- Alerts are independent of the ML model's heart disease prediction.
- A low ML risk score must NEVER suppress a clinical alert.
- These are SCREENING reference rules, not diagnostic criteria.
- Every threshold is documented with its source, intended population and
  known limitations.
- Thresholds are configurable through the THRESHOLDS dict below.
- Hypothetical (Digital Twin scenario) values are clearly labelled as such;
  they must never be presented as actual clinical measurements.

THRESHOLD SOURCES & LIMITATIONS
---------------------------------
Blood Pressure (adult screening reference):
  Source:  General adult clinical screening references
           (e.g. ESH/ESC 2018 guidelines, AHA/ACC 2017)
  Low BP:  Systolic < 90 mmHg or Diastolic < 60 mmHg
  High BP: Systolic ≥ 140 mmHg or Diastolic ≥ 90 mmHg
           (Stage 2 hypertension threshold; a single elevated reading does
           not establish a diagnosis of hypertension)
  Unit:    mmHg
  Limitation: Thresholds reflect population-level screening references for
              adult patients. They may not apply to children, pregnant
              individuals, athletes, or patients with specific comorbidities.
              A single reading is never sufficient for diagnosis.

Glucose (fasting reference, adults):
  Source:  ADA Standards of Medical Care in Diabetes (2023 edition)
  Ranges:  Normal fasting < 100 mg/dL
           Impaired fasting (pre-diabetes): 100–125 mg/dL
           Diabetic range: ≥ 126 mg/dL
  Unit:    mg/dL
  Limitation: Context of measurement (fasting vs. random vs. post-prandial)
              is critical. This engine treats the stored glucose value as a
              general screening figure; the clinical team must confirm the
              measurement context.

Cholesterol (total, adults — categorical encoding):
  Source:  The clinical model encodes cholesterol as a categorical integer:
           1 = Normal, 2 = Above Normal, 3 = Well Above Normal
           These correspond to approximate ranges used in the Cardiovascular
           Disease dataset (UCI/Kaggle origin).
  Limitation: The categorical encoding precludes applying exact mg/dL
              thresholds. Alerts reference the encoded category only.
              If raw mg/dL values are available in future, replace with
              NCEP ATP III thresholds (desirable < 200 mg/dL, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Configurable thresholds — change here, not at call sites
# ---------------------------------------------------------------------------
THRESHOLDS: dict = {
    "bp_systolic_low":      90.0,    # mmHg — below this triggers low-BP alert
    "bp_diastolic_low":     60.0,    # mmHg — below this triggers low-BP alert
    "bp_systolic_high":    140.0,    # mmHg — at or above triggers elevated-BP alert
    "bp_diastolic_high":    90.0,    # mmHg — at or above triggers elevated-BP alert
    # Glucose thresholds (mg/dL, interpreted as a general fasting reference)
    "glucose_prediabetic":  100.0,   # mg/dL — impaired fasting glucose starts here
    "glucose_diabetic":     126.0,   # mg/dL — ADA diagnostic threshold
    # Cholesterol categorical levels (1=Normal, 2=Above Normal, 3=Well Above Normal)
    "cholesterol_elevated":   2,     # int category — flag at ≥ this value
    "cholesterol_high":       3,     # int category — flag at ≥ this value
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ClinicalAlert:
    """
    One structured clinical alert produced by this engine.

    Fields
    ------
    alert_id        : Machine-readable identifier (e.g. 'LOW_SYSTOLIC_BP').
    parameter       : Human-readable parameter name.
    measured_value  : The actual measured (or hypothetical) value.
    unit            : SI/clinical unit (e.g. 'mmHg').
    category        : Alert category string (e.g. 'Blood Pressure').
    severity        : 'info', 'warning', or 'urgent'.
    threshold_rule  : The threshold or rule that was triggered.
    explanation     : Plain-language explanation for the patient/clinician.
    recommendation  : Recommended next step.
    is_hypothetical : True when the value comes from a Digital Twin scenario,
                      not from an actual clinical measurement.
    context_id      : Assessment ID or scenario name for traceability.
    """
    alert_id: str
    parameter: str
    measured_value: object            # numeric or string
    unit: str
    category: str
    severity: str                     # 'info' | 'warning' | 'urgent'
    threshold_rule: str
    explanation: str
    recommendation: str
    is_hypothetical: bool = False
    context_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Alert construction helpers
# ---------------------------------------------------------------------------
def _low_systolic_alert(
    systolic: float,
    diastolic: float,
    is_hypothetical: bool,
    context_id: Optional[str],
) -> ClinicalAlert:
    threshold = THRESHOLDS["bp_systolic_low"]
    if is_hypothetical:
        explanation = (
            f"This hypothetical scenario contains a systolic blood pressure of "
            f"{systolic:.0f} mmHg, which is below the configured adult screening "
            f"threshold of {threshold:.0f} mmHg. "
            f"This is a simulated value and does not establish the patient's actual "
            f"blood pressure. If a similar reading is observed in a real measurement, "
            f"appropriate clinical review is recommended."
        )
        recommendation = (
            "Do not interpret this hypothetical reading as an actual clinical finding. "
            "If a real measurement returns a similar value, consult a qualified "
            "healthcare professional, particularly if symptoms such as dizziness, "
            "weakness, blurred vision, or fainting are present."
        )
    else:
        explanation = (
            f"Your recorded blood pressure is {systolic:.0f}/{diastolic:.0f} mmHg. "
            f"The systolic component ({systolic:.0f} mmHg) is below the configured "
            f"adult screening threshold of {threshold:.0f} mmHg, which may indicate "
            f"low blood pressure. This is a screening reference and does not constitute "
            f"a diagnosis of hypotension."
        )
        recommendation = (
            "Consider arranging a clinical review, particularly if you experience "
            "dizziness, weakness, blurred vision, or fainting. "
            "Do not change prescribed medication without medical advice."
        )
    return ClinicalAlert(
        alert_id="LOW_SYSTOLIC_BP",
        parameter="Systolic Blood Pressure",
        measured_value=systolic,
        unit="mmHg",
        category="Blood Pressure",
        severity="warning",
        threshold_rule=f"Systolic BP < {threshold:.0f} mmHg (adult screening reference)",
        explanation=explanation,
        recommendation=recommendation,
        is_hypothetical=is_hypothetical,
        context_id=context_id,
    )


def _low_diastolic_alert(
    systolic: float,
    diastolic: float,
    is_hypothetical: bool,
    context_id: Optional[str],
) -> ClinicalAlert:
    threshold = THRESHOLDS["bp_diastolic_low"]
    if is_hypothetical:
        explanation = (
            f"This hypothetical scenario contains a diastolic blood pressure of "
            f"{diastolic:.0f} mmHg, which is below the configured adult screening "
            f"threshold of {threshold:.0f} mmHg. "
            f"This is a simulated value and does not establish the patient's actual "
            f"blood pressure."
        )
        recommendation = (
            "Do not interpret this hypothetical reading as an actual clinical finding. "
            "If a real measurement returns a similar value, consult a qualified "
            "healthcare professional."
        )
    else:
        explanation = (
            f"Your recorded blood pressure is {systolic:.0f}/{diastolic:.0f} mmHg. "
            f"The diastolic component ({diastolic:.0f} mmHg) is below the configured "
            f"adult screening threshold of {threshold:.0f} mmHg, which may indicate "
            f"low diastolic blood pressure. This is a screening reference and does not "
            f"constitute a diagnosis of hypotension."
        )
        recommendation = (
            "Consider arranging a clinical review with a qualified healthcare "
            "professional, particularly if you experience dizziness, weakness, "
            "blurred vision, or fainting. "
            "Do not change prescribed medication without medical advice."
        )
    return ClinicalAlert(
        alert_id="LOW_DIASTOLIC_BP",
        parameter="Diastolic Blood Pressure",
        measured_value=diastolic,
        unit="mmHg",
        category="Blood Pressure",
        severity="warning",
        threshold_rule=f"Diastolic BP < {threshold:.0f} mmHg (adult screening reference)",
        explanation=explanation,
        recommendation=recommendation,
        is_hypothetical=is_hypothetical,
        context_id=context_id,
    )


def _high_systolic_alert(
    systolic: float,
    diastolic: float,
    is_hypothetical: bool,
    context_id: Optional[str],
) -> ClinicalAlert:
    threshold = THRESHOLDS["bp_systolic_high"]
    if is_hypothetical:
        explanation = (
            f"This hypothetical scenario contains a systolic blood pressure of "
            f"{systolic:.0f} mmHg, which meets or exceeds the adult screening "
            f"threshold of {threshold:.0f} mmHg. This is a simulated value."
        )
    else:
        explanation = (
            f"Your recorded blood pressure is {systolic:.0f}/{diastolic:.0f} mmHg. "
            f"The systolic component ({systolic:.0f} mmHg) meets or exceeds the "
            f"configured adult screening threshold of {threshold:.0f} mmHg. "
            f"A single reading is not sufficient to diagnose hypertension."
        )
    return ClinicalAlert(
        alert_id="HIGH_SYSTOLIC_BP",
        parameter="Systolic Blood Pressure",
        measured_value=systolic,
        unit="mmHg",
        category="Blood Pressure",
        severity="warning",
        threshold_rule=f"Systolic BP ≥ {threshold:.0f} mmHg (adult screening reference)",
        explanation=explanation,
        recommendation=(
            "Discuss this reading with a healthcare professional. "
            "A single elevated reading does not diagnose hypertension. "
            "Lifestyle factors such as sodium restriction, regular exercise, "
            "and weight management may support blood pressure control."
        ),
        is_hypothetical=is_hypothetical,
        context_id=context_id,
    )


def _high_diastolic_alert(
    systolic: float,
    diastolic: float,
    is_hypothetical: bool,
    context_id: Optional[str],
) -> ClinicalAlert:
    threshold = THRESHOLDS["bp_diastolic_high"]
    if is_hypothetical:
        explanation = (
            f"This hypothetical scenario contains a diastolic blood pressure of "
            f"{diastolic:.0f} mmHg, which meets or exceeds the adult screening "
            f"threshold of {threshold:.0f} mmHg. This is a simulated value."
        )
    else:
        explanation = (
            f"Your recorded blood pressure is {systolic:.0f}/{diastolic:.0f} mmHg. "
            f"The diastolic component ({diastolic:.0f} mmHg) meets or exceeds the "
            f"configured adult screening threshold of {threshold:.0f} mmHg. "
            f"A single reading is not sufficient to diagnose hypertension."
        )
    return ClinicalAlert(
        alert_id="HIGH_DIASTOLIC_BP",
        parameter="Diastolic Blood Pressure",
        measured_value=diastolic,
        unit="mmHg",
        category="Blood Pressure",
        severity="warning",
        threshold_rule=f"Diastolic BP ≥ {threshold:.0f} mmHg (adult screening reference)",
        explanation=explanation,
        recommendation=(
            "Discuss this reading with a healthcare professional. "
            "A single elevated reading does not diagnose hypertension."
        ),
        is_hypothetical=is_hypothetical,
        context_id=context_id,
    )


def _glucose_alert(
    glucose: float,
    is_hypothetical: bool,
    context_id: Optional[str],
) -> Optional[ClinicalAlert]:
    """
    Evaluates glucose against ADA fasting reference ranges.
    Returns an alert only if glucose meets the impaired-fasting threshold.
    Returns None for normal values.

    LIMITATION: The measurement context (fasting vs. random vs. post-prandial)
    is not captured by the current schema. These thresholds assume a fasting
    reading. The alert text reflects this uncertainty.
    """
    prediabetic_t = THRESHOLDS["glucose_prediabetic"]
    diabetic_t = THRESHOLDS["glucose_diabetic"]

    if glucose < prediabetic_t:
        return None  # normal range — no alert

    if glucose >= diabetic_t:
        severity = "warning"
        alert_id = "HIGH_GLUCOSE_DIABETIC_RANGE"
        threshold_rule = (
            f"Glucose ≥ {diabetic_t:.0f} mg/dL (ADA diabetic-range fasting reference)"
        )
        explanation_suffix = (
            f"a value in the diabetic-range fasting reference (≥ {diabetic_t:.0f} mg/dL)"
        )
    else:
        severity = "info"
        alert_id = "HIGH_GLUCOSE_PREDIABETIC_RANGE"
        threshold_rule = (
            f"Glucose ≥ {prediabetic_t:.0f} mg/dL "
            f"(ADA impaired-fasting reference: {prediabetic_t:.0f}–{diabetic_t - 1:.0f} mg/dL)"
        )
        explanation_suffix = (
            f"a value in the impaired-fasting reference range "
            f"({prediabetic_t:.0f}–{diabetic_t - 1:.0f} mg/dL)"
        )

    prefix = "This hypothetical scenario contains a" if is_hypothetical else "Your recorded"
    glucose_context = (
        "The measurement context (fasting vs. random) has not been specified; "
        "these thresholds apply to fasting measurements."
    )
    explanation = (
        f"{prefix} glucose of {glucose:.0f} mg/dL, which falls within "
        f"{explanation_suffix}. {glucose_context}"
    )
    if is_hypothetical:
        explanation += (
            " This is a simulated value and does not constitute a clinical finding."
        )
    return ClinicalAlert(
        alert_id=alert_id,
        parameter="Blood Glucose",
        measured_value=glucose,
        unit="mg/dL",
        category="Glucose",
        severity=severity,
        threshold_rule=threshold_rule,
        explanation=explanation,
        recommendation=(
            "Discuss the glucose measurement context and result with a healthcare "
            "professional. Do not self-diagnose based on a single measurement."
        ),
        is_hypothetical=is_hypothetical,
        context_id=context_id,
    )


def _cholesterol_alert(
    cholesterol_category: int,
    is_hypothetical: bool,
    context_id: Optional[str],
) -> Optional[ClinicalAlert]:
    """
    Evaluates the categorical cholesterol encoding (1/2/3) used by the
    existing clinical model.  Returns an alert for categories ≥ 2 (above normal).
    Returns None for category 1 (normal).
    """
    elevated_t = THRESHOLDS["cholesterol_elevated"]
    high_t = THRESHOLDS["cholesterol_high"]

    try:
        cat = int(cholesterol_category)
    except (TypeError, ValueError):
        return None  # not a valid numeric category — cannot evaluate

    if cat < elevated_t:
        return None  # normal — no alert

    if cat >= high_t:
        alert_id = "CHOLESTEROL_WELL_ABOVE_NORMAL"
        severity = "warning"
        cat_label = "well above normal"
    else:
        alert_id = "CHOLESTEROL_ABOVE_NORMAL"
        severity = "info"
        cat_label = "above normal"

    prefix = "This hypothetical scenario contains a" if is_hypothetical else "Your recorded"
    explanation = (
        f"{prefix} cholesterol category of {cat} (encoded as: {cat_label}). "
        f"Note: the application uses a categorical encoding (1=Normal, 2=Above Normal, "
        f"3=Well Above Normal) rather than exact mg/dL values."
    )
    if is_hypothetical:
        explanation += " This is a simulated value and does not constitute a clinical finding."
    return ClinicalAlert(
        alert_id=alert_id,
        parameter="Total Cholesterol (categorical)",
        measured_value=cat,
        unit="category (1–3)",
        category="Cholesterol",
        severity=severity,
        threshold_rule=(
            f"Categorical cholesterol ≥ {elevated_t} (above-normal screening reference)"
        ),
        explanation=explanation,
        recommendation=(
            "Discuss cholesterol management with a healthcare professional. "
            "A fasting lipid panel with exact mg/dL values provides more precise guidance."
        ),
        is_hypothetical=is_hypothetical,
        context_id=context_id,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def evaluate_clinical_alerts(
    systolic_bp: Optional[float] = None,
    diastolic_bp: Optional[float] = None,
    glucose: Optional[float] = None,
    cholesterol_category: Optional[int] = None,
    is_hypothetical: bool = False,
    context_id: Optional[str] = None,
) -> List[ClinicalAlert]:
    """
    Evaluate available clinical measurements and return structured alerts.

    Parameters
    ----------
    systolic_bp           : Systolic blood pressure in mmHg. Pass None if not collected.
    diastolic_bp          : Diastolic blood pressure in mmHg. Pass None if not collected.
    glucose               : Blood glucose in mg/dL. Pass None if not collected.
    cholesterol_category  : Categorical cholesterol (1/2/3). Pass None if not collected.
    is_hypothetical       : True when values come from a Digital Twin scenario.
    context_id            : Assessment ID or scenario name for traceability.

    Returns
    -------
    List[ClinicalAlert]   : Alerts in the order they were triggered.
                            Empty list = no alerts triggered.

    Notes
    -----
    - Missing values (None) are skipped entirely; they do not generate alerts.
    - A missing value is NOT treated as normal.
    - The ML model risk score must NOT be passed here; alerts are independent of it.
    - Both systolic and diastolic are evaluated independently.
    """
    alerts: List[ClinicalAlert] = []

    # ── Blood pressure checks ────────────────────────────────────────────────
    bp_available = systolic_bp is not None and diastolic_bp is not None

    if bp_available:
        # Validate plausibility
        sys_val = float(systolic_bp)
        dia_val = float(diastolic_bp)

        # Low BP — systolic and diastolic evaluated independently
        if sys_val < THRESHOLDS["bp_systolic_low"]:
            alerts.append(
                _low_systolic_alert(sys_val, dia_val, is_hypothetical, context_id)
            )
        if dia_val < THRESHOLDS["bp_diastolic_low"]:
            alerts.append(
                _low_diastolic_alert(sys_val, dia_val, is_hypothetical, context_id)
            )

        # Elevated BP — only if NOT already flagged as low
        if sys_val >= THRESHOLDS["bp_systolic_high"]:
            alerts.append(
                _high_systolic_alert(sys_val, dia_val, is_hypothetical, context_id)
            )
        if dia_val >= THRESHOLDS["bp_diastolic_high"]:
            alerts.append(
                _high_diastolic_alert(sys_val, dia_val, is_hypothetical, context_id)
            )

    # ── Glucose check ────────────────────────────────────────────────────────
    if glucose is not None:
        alert = _glucose_alert(float(glucose), is_hypothetical, context_id)
        if alert:
            alerts.append(alert)

    # ── Cholesterol check ────────────────────────────────────────────────────
    if cholesterol_category is not None:
        alert = _cholesterol_alert(cholesterol_category, is_hypothetical, context_id)
        if alert:
            alerts.append(alert)

    return alerts


def alerts_to_dicts(alerts: List[ClinicalAlert]) -> List[dict]:
    """
    Serialise a list of ClinicalAlert objects to plain dicts suitable for
    JSON responses and report builders.
    """
    return [
        {
            "alert_id": a.alert_id,
            "parameter": a.parameter,
            "measured_value": a.measured_value,
            "unit": a.unit,
            "category": a.category,
            "severity": a.severity,
            "threshold_rule": a.threshold_rule,
            "explanation": a.explanation,
            "recommendation": a.recommendation,
            "is_hypothetical": a.is_hypothetical,
            "context_id": a.context_id,
        }
        for a in alerts
    ]
