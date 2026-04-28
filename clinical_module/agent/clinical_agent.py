# clinical_module/agent/clinical_agent.py
# ============================================================
# CLINICAL AGENT — Week 6
# Author  : Rishitha
# Purpose : Wraps the trained stacking model into a clean,
#           callable agent function for use in the system pipeline.
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── 1. MODEL LOADING (happens ONCE when this file is imported) ──────────────
# The .pkl files must be in the same folder as this script,
# OR you can set the MODEL_DIR environment variable to point elsewhere.

MODEL_DIR   = os.environ.get("CLINICAL_MODEL_DIR", os.path.dirname(__file__))
MODEL_PATH  = os.path.join(MODEL_DIR, "stack_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

try:
    _stack_model = joblib.load(MODEL_PATH)
    _scaler      = joblib.load(SCALER_PATH)
    print(f"[ClinicalAgent] ✅ Model loaded from: {MODEL_DIR}")
except FileNotFoundError as e:
    _stack_model = None
    _scaler      = None
    print(f"[ClinicalAgent] ⚠️  Model files not found — {e}")
    print("[ClinicalAgent]    Place stack_model.pkl and scaler.pkl next to this file.")

# ── 2. CONSTANTS ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "smoke", "alco", "active", "bmi", "hypertension",
    "pulse_pressure", "age_bmi_interaction", "bp_ratio",
    "bmi_age_ratio", "chol_gluc_product", "lifestyle_score",
    "cholesterol_2", "cholesterol_3", "gluc_2", "gluc_3"
]

LABEL_MAP = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

# Valid ranges for input validation
VALID_RANGES = {
    "age"        : (1,   120),
    "gender"     : (1,   2),
    "height"     : (100, 250),
    "weight"     : (30,  300),
    "ap_hi"      : (50,  300),
    "ap_lo"      : (30,  200),
    "cholesterol": (1,   3),
    "gluc"       : (1,   3),
    "smoke"      : (0,   1),
    "alco"       : (0,   1),
    "active"     : (0,   1),
}

# ── 3. INPUT VALIDATION ───────────────────────────────────────────────────────

def _validate_input(data: dict) -> list:
    """
    Checks all required fields are present and within valid ranges.
    Returns a list of error strings (empty list = all good).
    """
    errors = []

    required = list(VALID_RANGES.keys())
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")
            continue  # skip range check if field is absent

        val = data[field]

        # Type check — must be a number
        if not isinstance(val, (int, float)):
            errors.append(f"'{field}' must be a number, got {type(val).__name__}")
            continue

        lo, hi = VALID_RANGES[field]
        if not (lo <= val <= hi):
            errors.append(f"'{field}' = {val} is out of valid range [{lo}, {hi}]")

    # Extra: systolic must be >= diastolic
    if "ap_hi" in data and "ap_lo" in data:
        if isinstance(data["ap_hi"], (int, float)) and isinstance(data["ap_lo"], (int, float)):
            if data["ap_hi"] < data["ap_lo"]:
                errors.append(
                    f"ap_hi ({data['ap_hi']}) must be >= ap_lo ({data['ap_lo']})"
                )

    return errors

# ── 4. FEATURE ENGINEERING ────────────────────────────────────────────────────

def _build_features(data: dict) -> pd.DataFrame:
    """
    Takes raw validated input and computes all engineered features,
    exactly matching how the training pipeline was built.
    """
    age        = data["age"]
    gender     = data["gender"]
    height     = data["height"]
    weight     = data["weight"]
    ap_hi      = data["ap_hi"]
    ap_lo      = data["ap_lo"]
    cholesterol= data["cholesterol"]
    gluc       = data["gluc"]
    smoke      = data["smoke"]
    alco       = data["alco"]
    active     = data["active"]

    bmi = weight / ((height / 100) ** 2)

    row = {
        # Raw features
        "age"                : age,
        "gender"             : gender,
        "height"             : height,
        "weight"             : weight,
        "ap_hi"              : ap_hi,
        "ap_lo"              : ap_lo,
        "smoke"              : smoke,
        "alco"               : alco,
        "active"             : active,

        # Engineered features (mirrors training pipeline exactly)
        "bmi"                : bmi,
        "hypertension"       : int(ap_hi >= 140 or ap_lo >= 90),
        "pulse_pressure"     : ap_hi - ap_lo,
        "age_bmi_interaction": age * bmi,
        "bp_ratio"           : ap_hi / (ap_lo + 1),
        "bmi_age_ratio"      : bmi / (age + 1),
        "chol_gluc_product"  : cholesterol * gluc,
        "lifestyle_score"    : smoke + alco - active,

        # One-hot encoded (drop_first=True was used in training)
        "cholesterol_2"      : int(cholesterol == 2),
        "cholesterol_3"      : int(cholesterol == 3),
        "gluc_2"             : int(gluc == 2),
        "gluc_3"             : int(gluc == 3),
    }

    df = pd.DataFrame([row]).reindex(columns=FEATURE_COLS, fill_value=0)
    return df

# ── 5. THE AGENT FUNCTION ─────────────────────────────────────────────────────

def clinical_agent(input_data: dict) -> dict:
    """
    The main callable agent function.

    Parameters
    ----------
    input_data : dict
        Required keys:
            age         — int,   age in years (1–120)
            gender      — int,   1 = female, 2 = male
            height      — int,   cm (100–250)
            weight      — float, kg (30–300)
            ap_hi       — int,   systolic blood pressure mmHg (50–300)
            ap_lo       — int,   diastolic blood pressure mmHg (30–200)
            cholesterol — int,   1 = normal, 2 = above normal, 3 = well above normal
            gluc        — int,   1 = normal, 2 = above normal, 3 = well above normal
            smoke       — int,   0 or 1
            alco        — int,   0 or 1
            active      — int,   0 or 1 (physical activity)

    Returns
    -------
    dict with keys:
        success          : bool   — True if prediction succeeded
        source           : str    — always "clinical" (for fusion pipeline)
        predicted_class  : str    — "Low Risk" / "Medium Risk" / "High Risk"
        predicted_label  : int    — 0 / 1 / 2
        confidence       : float  — probability of the predicted class
        prob_low         : float  — probability of Low Risk
        prob_medium      : float  — probability of Medium Risk
        prob_high        : float  — probability of High Risk
        error            : str    — error message if success=False, else None
    """

    # ── Check model is loaded ──────────────────────────────────
    if _stack_model is None or _scaler is None:
        return {
            "success"        : False,
            "source"         : "clinical",
            "predicted_class": None,
            "predicted_label": None,
            "confidence"     : None,
            "prob_low"       : None,
            "prob_medium"    : None,
            "prob_high"      : None,
            "error"          : "Model files not loaded. Check stack_model.pkl and scaler.pkl."
        }

    # ── Input validation ──────────────────────────────────────
    errors = _validate_input(input_data)
    if errors:
        return {
            "success"        : False,
            "source"         : "clinical",
            "predicted_class": None,
            "predicted_label": None,
            "confidence"     : None,
            "prob_low"       : None,
            "prob_medium"    : None,
            "prob_high"      : None,
            "error"          : " | ".join(errors)
        }

    # ── Build features & scale ────────────────────────────────
    try:
        df     = _build_features(input_data)
        scaled = _scaler.transform(df)
    except Exception as e:
        return {
            "success"        : False,
            "source"         : "clinical",
            "predicted_class": None,
            "predicted_label": None,
            "confidence"     : None,
            "prob_low"       : None,
            "prob_medium"    : None,
            "prob_high"      : None,
            "error"          : f"Feature engineering failed: {str(e)}"
        }

    # ── Predict ───────────────────────────────────────────────
    try:
        proba       = _stack_model.predict_proba(scaled)[0]   # shape: (3,)
        pred_label  = int(np.argmax(proba))
        pred_class  = LABEL_MAP[pred_label]
        confidence  = float(proba[pred_label])
    except Exception as e:
        return {
            "success"        : False,
            "source"         : "clinical",
            "predicted_class": None,
            "predicted_label": None,
            "confidence"     : None,
            "prob_low"       : None,
            "prob_medium"    : None,
            "prob_high"      : None,
            "error"          : f"Prediction failed: {str(e)}"
        }

    # ── Return standardized output ────────────────────────────
    return {
        "success"        : True,
        "source"         : "clinical",          # fusion uses this to identify the module
        "predicted_class": pred_class,
        "predicted_label": pred_label,
        "confidence"     : round(confidence, 6),
        "prob_low"       : round(float(proba[0]), 6),
        "prob_medium"    : round(float(proba[1]), 6),
        "prob_high"      : round(float(proba[2]), 6),
        "error"          : None
    }

# ── 6. BATCH AGENT (optional — for testing multiple patients at once) ─────────

def clinical_agent_batch(patient_list: list) -> list:
    """
    Runs clinical_agent() on a list of patient dicts.
    Returns a list of result dicts (one per patient).
    """
    results = []
    for i, patient in enumerate(patient_list):
        result = clinical_agent(patient)
        result["patient_id"] = i
        results.append(result)
    return results

# ── 7. QUICK TEST (runs only when you execute this file directly) ─────────────

if __name__ == "__main__":

    print("\n" + "="*55)
    print("       CLINICAL AGENT — QUICK TEST")
    print("="*55)

    # Test 1: High-risk patient
    print("\n🔬 Test 1: High-risk patient")
    result = clinical_agent({
        "age": 60, "gender": 2, "height": 170, "weight": 95,
        "ap_hi": 180, "ap_lo": 110,
        "cholesterol": 3, "gluc": 2,
        "smoke": 1, "alco": 1, "active": 0
    })
    print(json.dumps(result, indent=2))

    # Test 2: Low-risk patient
    print("\n🔬 Test 2: Low-risk patient")
    result2 = clinical_agent({
        "age": 30, "gender": 1, "height": 165, "weight": 60,
        "ap_hi": 115, "ap_lo": 75,
        "cholesterol": 1, "gluc": 1,
        "smoke": 0, "alco": 0, "active": 1
    })
    print(json.dumps(result2, indent=2))

    # Test 3: Borderline patient
    print("\n🔬 Test 3: Borderline patient")
    result3 = clinical_agent({
        "age": 45, "gender": 2, "height": 175, "weight": 82,
        "ap_hi": 135, "ap_lo": 88,
        "cholesterol": 2, "gluc": 1,
        "smoke": 0, "alco": 1, "active": 1
    })
    print(json.dumps(result3, indent=2))

    # Test 4: Bad input (validation test)
    print("\n🔬 Test 4: Invalid input (should return error)")
    result4 = clinical_agent({
        "age": 200,   # out of range
        "gender": 1,
        # height is missing
        "weight": 60,
        "ap_hi": 80,
        "ap_lo": 120,  # ap_lo > ap_hi — invalid
        "cholesterol": 1, "gluc": 1,
        "smoke": 0, "alco": 0, "active": 1
    })
    print(json.dumps(result4, indent=2))

    # Test 5: Batch prediction
    print("\n📋 Test 5: Batch prediction")
    patients = [
        {"age": 60, "gender": 2, "height": 170, "weight": 95,
         "ap_hi": 180, "ap_lo": 110, "cholesterol": 3, "gluc": 2,
         "smoke": 1, "alco": 1, "active": 0},
        {"age": 30, "gender": 1, "height": 165, "weight": 60,
         "ap_hi": 115, "ap_lo": 75, "cholesterol": 1, "gluc": 1,
         "smoke": 0, "alco": 0, "active": 1},
    ]
    batch_results = clinical_agent_batch(patients)
    for r in batch_results:
        print(f"  Patient {r['patient_id']}: {r['predicted_class']} "
              f"(confidence: {r['confidence']:.2%})")
