# clinical_module/agent/clinical_agent.py
# ============================================================
# CLINICAL AGENT - Week 6
# Purpose: Wrap the trained stacking model into a callable
#          agent function for use in the system pipeline.
# ============================================================

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


MODEL_DIR = os.environ.get("CLINICAL_MODEL_DIR", os.path.dirname(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "stack_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


def _force_single_worker(estimator) -> None:
    """Disable parallel worker pools that fail under this Windows sandbox."""

    if estimator is None:
        return

    if hasattr(estimator, "n_jobs"):
        try:
            estimator.set_params(n_jobs=1)
        except Exception:
            try:
                estimator.n_jobs = 1
            except Exception:
                pass

    for child in getattr(estimator, "estimators_", []):
        _force_single_worker(child)

    for pair in getattr(estimator, "estimators", []):
        if isinstance(pair, tuple) and len(pair) == 2:
            _force_single_worker(pair[1])

    _force_single_worker(getattr(estimator, "final_estimator_", None))
    _force_single_worker(getattr(estimator, "final_estimator", None))


def _load_artifacts():
    """Load the trained model and scaler once at import time."""

    try:
        stack_model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        _force_single_worker(stack_model)
        return stack_model, scaler, None
    except FileNotFoundError as exc:
        return None, None, str(exc)
    except Exception as exc:
        return None, None, str(exc)


_stack_model, _scaler, _MODEL_LOAD_ERROR = _load_artifacts()


FEATURE_COLS = [
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "smoke",
    "alco",
    "active",
    "bmi",
    "hypertension",
    "pulse_pressure",
    "age_bmi_interaction",
    "bp_ratio",
    "bmi_age_ratio",
    "chol_gluc_product",
    "lifestyle_score",
    "cholesterol_2",
    "cholesterol_3",
    "gluc_2",
    "gluc_3",
]

LABEL_MAP = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

VALID_RANGES = {
    "age": (1, 120),
    "gender": (1, 2),
    "height": (100, 250),
    "weight": (30, 300),
    "ap_hi": (50, 300),
    "ap_lo": (30, 200),
    "cholesterol": (1, 3),
    "gluc": (1, 3),
    "smoke": (0, 1),
    "alco": (0, 1),
    "active": (0, 1),
}


def _validate_input(data: dict) -> list[str]:
    """Check required fields and numeric ranges."""

    errors: list[str] = []

    for field, (lower, upper) in VALID_RANGES.items():
        if field not in data:
            errors.append(f"Missing required field: '{field}'")
            continue

        value = data[field]
        if not isinstance(value, (int, float)):
            errors.append(f"'{field}' must be a number, got {type(value).__name__}")
            continue

        if not (lower <= value <= upper):
            errors.append(f"'{field}' = {value} is out of valid range [{lower}, {upper}]")

    if "ap_hi" in data and "ap_lo" in data:
        ap_hi = data["ap_hi"]
        ap_lo = data["ap_lo"]
        if isinstance(ap_hi, (int, float)) and isinstance(ap_lo, (int, float)):
            if ap_hi < ap_lo:
                errors.append(f"ap_hi ({ap_hi}) must be >= ap_lo ({ap_lo})")

    return errors


def validate_clinical_input(data: dict) -> list[str]:
    """Public validation helper for CLI input and other callers."""

    return _validate_input(data)


def _build_features(data: dict) -> pd.DataFrame:
    """Compute the engineered features expected by the trained pipeline."""

    age = data["age"]
    gender = data["gender"]
    height = data["height"]
    weight = data["weight"]
    ap_hi = data["ap_hi"]
    ap_lo = data["ap_lo"]
    cholesterol = data["cholesterol"]
    gluc = data["gluc"]
    smoke = data["smoke"]
    alco = data["alco"]
    active = data["active"]

    bmi = weight / ((height / 100) ** 2)

    row = {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "bmi": bmi,
        "hypertension": int(ap_hi >= 140 or ap_lo >= 90),
        "pulse_pressure": ap_hi - ap_lo,
        "age_bmi_interaction": age * bmi,
        "bp_ratio": ap_hi / (ap_lo + 1),
        "bmi_age_ratio": bmi / (age + 1),
        "chol_gluc_product": cholesterol * gluc,
        "lifestyle_score": smoke + alco - active,
        "cholesterol_2": int(cholesterol == 2),
        "cholesterol_3": int(cholesterol == 3),
        "gluc_2": int(gluc == 2),
        "gluc_3": int(gluc == 3),
    }

    return pd.DataFrame([row]).reindex(columns=FEATURE_COLS, fill_value=0)


def _build_clinical_narrative(data: dict, level: str, confidence: float | None = None) -> str:
    """
    Generate a rich, patient-specific clinical narrative from raw input values.
    Works with or without ML model output.
    """
    age = data["age"]
    height = data["height"]
    weight = data["weight"]
    ap_hi = data["ap_hi"]
    ap_lo = data["ap_lo"]
    cholesterol = data["cholesterol"]
    gluc = data["gluc"]
    smoke = data["smoke"]
    alco = data["alco"]
    active = data["active"]

    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = ap_hi - ap_lo

    # ── BMI classification ────────────────────────────────────────────────
    if bmi < 18.5:
        bmi_label = "underweight"
    elif bmi < 25.0:
        bmi_label = "normal weight"
    elif bmi < 30.0:
        bmi_label = "overweight"
    else:
        bmi_label = "obese"

    # ── Blood pressure classification ─────────────────────────────────────
    if ap_hi >= 180 or ap_lo >= 120:
        bp_label = "hypertensive crisis"
    elif ap_hi >= 140 or ap_lo >= 90:
        bp_label = "Stage 2 hypertension"
    elif ap_hi >= 130 or ap_lo >= 80:
        bp_label = "Stage 1 hypertension"
    elif ap_hi >= 120:
        bp_label = "elevated blood pressure"
    else:
        bp_label = "normal blood pressure"

    # ── Cholesterol label ─────────────────────────────────────────────────
    chol_labels = {1: "normal", 2: "above normal", 3: "well above normal"}
    chol_label = chol_labels.get(cholesterol, "unknown")

    # ── Glucose label ─────────────────────────────────────────────────────
    gluc_labels = {1: "normal", 2: "above normal", 3: "well above normal"}
    gluc_label = gluc_labels.get(gluc, "unknown")

    # ── Lifestyle summary ─────────────────────────────────────────────────
    lifestyle_parts = []
    if smoke:
        lifestyle_parts.append("active smoker")
    if alco:
        lifestyle_parts.append("alcohol consumer")
    if not active:
        lifestyle_parts.append("physically inactive")
    lifestyle_str = (
        ", ".join(lifestyle_parts) if lifestyle_parts else "non-smoker, non-drinker, physically active"
    )

    # ── Age risk note ─────────────────────────────────────────────────────
    age_note = ""
    if age >= 60:
        age_note = f"At age {age}, cardiovascular risk is significantly elevated."
    elif age >= 45:
        age_note = f"At age {age}, regular cardiac monitoring is advisable."
    else:
        age_note = f"At age {age}, baseline cardiovascular risk is lower."

    # ── Confidence string ─────────────────────────────────────────────────
    conf_str = f" (model confidence: {confidence * 100:.1f}%)" if confidence is not None else " (rule-based estimate)"

    # ── Final narrative ───────────────────────────────────────────────────
    narrative = (
        f"Risk assessment: {level} risk{conf_str}. "
        f"Patient profile — Age: {age} years, BMI: {bmi:.1f} ({bmi_label}), "
        f"Blood Pressure: {ap_hi}/{ap_lo} mmHg ({bp_label}, pulse pressure: {pulse_pressure} mmHg), "
        f"Cholesterol: {chol_label}, Glucose: {gluc_label}. "
        f"Lifestyle: {lifestyle_str}. "
        f"{age_note}"
    )

    return narrative


def _rule_based_clinical(data: dict) -> dict:
    """
    Fallback rule-based clinical risk assessment when the ML model is unavailable.
    Uses clinically validated thresholds to compute a risk level and score.
    """
    score = 0.0
    age = data["age"]
    bmi = data["weight"] / ((data["height"] / 100) ** 2)
    ap_hi = data["ap_hi"]
    ap_lo = data["ap_lo"]
    cholesterol = data["cholesterol"]
    gluc = data["gluc"]
    smoke = data["smoke"]
    alco = data["alco"]
    active = data["active"]

    # Age risk
    if age >= 60:
        score += 0.25
    elif age >= 45:
        score += 0.15
    elif age >= 35:
        score += 0.05

    # BMI risk
    if bmi >= 30:
        score += 0.15
    elif bmi >= 25:
        score += 0.08

    # Blood pressure risk
    if ap_hi >= 180 or ap_lo >= 120:
        score += 0.30
    elif ap_hi >= 140 or ap_lo >= 90:
        score += 0.20
    elif ap_hi >= 130 or ap_lo >= 80:
        score += 0.10

    # Cholesterol risk
    if cholesterol == 3:
        score += 0.15
    elif cholesterol == 2:
        score += 0.08

    # Glucose risk
    if gluc == 3:
        score += 0.12
    elif gluc == 2:
        score += 0.06

    # Lifestyle risk
    if smoke:
        score += 0.10
    if alco:
        score += 0.05
    if not active:
        score += 0.08

    # Clamp score to [0, 1]
    score = min(score, 1.0)

    if score >= 0.55:
        level = "High"
    elif score >= 0.30:
        level = "Medium"
    else:
        level = "Low"

    reason = _build_clinical_narrative(data, level, confidence=None)

    return {
        "level": level,
        "score": round(score, 6),
        "reason": reason,
    }


def clinical_agent(input_data: dict) -> dict:
    """
    Run the clinical model and return a standardized response.
    Falls back to rule-based assessment if the ML model is unavailable.
    """

    errors = _validate_input(input_data)
    if errors:
        return {
            "level": None,
            "score": 0.0,
            "reason": "Validation errors: " + " | ".join(errors),
        }

    # ── Fallback: ML model not loaded ────────────────────────────────────
    if _stack_model is None or _scaler is None:
        result = _rule_based_clinical(input_data)
        result["reason"] = (
            "[Rule-based fallback — ML model unavailable] " + result["reason"]
        )
        return result

    # ── ML model path ─────────────────────────────────────────────────────
    try:
        features = _build_features(input_data)
        scaled = _scaler.transform(features)
    except Exception as exc:
        result = _rule_based_clinical(input_data)
        result["reason"] = f"[Rule-based fallback — feature error: {exc}] " + result["reason"]
        return result

    try:
        probabilities = _stack_model.predict_proba(scaled)[0]
        predicted_label = int(np.argmax(probabilities))
        predicted_class = LABEL_MAP[predicted_label]
        confidence = float(probabilities[predicted_label])
    except Exception as exc:
        result = _rule_based_clinical(input_data)
        result["reason"] = f"[Rule-based fallback — model error: {exc}] " + result["reason"]
        return result

    level_map = {"Low Risk": "Low", "Medium Risk": "Medium", "High Risk": "High"}
    level = level_map[predicted_class]

    reason = _build_clinical_narrative(input_data, level, confidence=confidence)

    return {
        "level": level,
        "score": round(confidence, 6),
        "reason": reason,
    }


def clinical_agent_batch(patient_list: list[dict]) -> list[dict]:
    """Run clinical_agent on a list of patients."""

    results = []
    for index, patient in enumerate(patient_list):
        result = clinical_agent(patient)
        result["patient_id"] = index
        results.append(result)
    return results


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("       CLINICAL AGENT - QUICK TEST")
    print("=" * 55)

    test_patients = [
        {
            "age": 60,
            "gender": 2,
            "height": 170,
            "weight": 95,
            "ap_hi": 180,
            "ap_lo": 110,
            "cholesterol": 3,
            "gluc": 2,
            "smoke": 1,
            "alco": 1,
            "active": 0,
        },
        {
            "age": 30,
            "gender": 1,
            "height": 165,
            "weight": 60,
            "ap_hi": 115,
            "ap_lo": 75,
            "cholesterol": 1,
            "gluc": 1,
            "smoke": 0,
            "alco": 0,
            "active": 1,
        },
    ]

    for patient in test_patients:
        print(json.dumps(clinical_agent(patient), indent=2))

    print(json.dumps(clinical_agent_batch(test_patients), indent=2))
