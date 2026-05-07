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


def clinical_agent(input_data: dict) -> dict:
    """Run the clinical model and return a standardized response."""

    if _stack_model is None or _scaler is None:
        return {
            "level": None,
            "score": 0.0,
            "reason": f"Clinical model unavailable: {_MODEL_LOAD_ERROR or 'model files not loaded'}",
        }

    errors = _validate_input(input_data)
    if errors:
        return {
            "level": None,
            "score": 0.0,
            "reason": " | ".join(errors),
        }

    try:
        features = _build_features(input_data)
        scaled = _scaler.transform(features)
    except Exception as exc:
        return {
            "level": None,
            "score": 0.0,
            "reason": f"Feature engineering failed: {exc}",
        }

    try:
        probabilities = _stack_model.predict_proba(scaled)[0]
        predicted_label = int(np.argmax(probabilities))
        predicted_class = LABEL_MAP[predicted_label]
        confidence = float(probabilities[predicted_label])
    except Exception as exc:
        return {
            "level": None,
            "score": 0.0,
            "reason": f"Clinical prediction failed: {exc}",
        }

    level_map = {
        "Low Risk": "Low",
        "Medium Risk": "Medium",
        "High Risk": "High",
    }
    level = level_map[predicted_class]

    if level == "High":
        reason = "High blood pressure and cholesterol levels detected"
    elif level == "Medium":
        reason = "Moderate risk due to clinical indicators"
    else:
        reason = "No major clinical risk factors detected"

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
