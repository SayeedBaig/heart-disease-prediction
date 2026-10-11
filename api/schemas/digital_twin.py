"""
Pydantic schemas for Digital Twin scenario simulation requests and responses.

These are deliberately separate from ClinicalInput so that:
- Hypothetical / simulated inputs are never confused with real clinical measurements.
- The scenario endpoint can accept a richer set of lifestyle parameters.
- The ML model is reused through the existing TwinEngine, not the full prediction pipeline.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ScenarioParams(BaseModel):
    """
    Hypothetical parameter overrides for a single Digital Twin scenario.
    All fields are optional — only the parameters the user wants to change
    need to be supplied. Everything else falls back to the patient baseline.
    """

    age: Optional[int] = Field(None, ge=1, le=120)
    systolic_bp: Optional[float] = Field(None, ge=50, le=300)
    diastolic_bp: Optional[float] = Field(None, ge=30, le=200)
    cholesterol: Optional[float] = Field(None, ge=50, le=600)
    glucose: Optional[float] = Field(None, ge=50, le=500)
    bmi: Optional[float] = Field(None, ge=10, le=70)
    weight: Optional[float] = Field(None, ge=20, le=300)
    smoking_status: Optional[bool] = None
    alcohol_consumption: Optional[bool] = None
    exercise_level: Optional[str] = Field(None, pattern="^(Low|Moderate|High)$")
    # Fusion baseline scores forwarded from original prediction
    clinical_risk_score: Optional[float] = Field(None, ge=0, le=100)
    ecg_risk_score: Optional[float] = Field(None, ge=0, le=100)
    echo_risk_score: Optional[float] = Field(None, ge=0, le=100)
    fusion_risk_percentage: Optional[float] = Field(None, ge=0, le=100)


class ScenarioRequest(BaseModel):
    """
    Full Digital Twin simulation request body.

    The caller supplies the patient's baseline values and a set of
    scenario-specific overrides. The backend merges them and runs the
    TwinEngine on the resulting hypothetical profile without touching
    any stored patient record.
    """

    # --- patient identification (for authorization only, never overwritten) ---
    patient_id: str

    # --- patient baseline values (mandatory, sourced from the latest prediction) ---
    baseline: ScenarioParams

    # --- per-scenario override maps ---
    scenarios: dict[str, ScenarioParams] = Field(
        default_factory=dict,
        description="Mapping of scenario_name -> parameter overrides relative to baseline",
    )


class ScenarioResult(BaseModel):
    scenario_name: str
    # TwinEngine composite risk score (NOT a calibrated probability)
    twin_risk_score: float
    changed_params: dict
    unchanged_params: dict
    # Clinical alerts evaluated for this scenario's hypothetical values.
    # Each dict is a serialised ClinicalAlert from the shared alert engine.
    # These are clearly marked is_hypothetical=True and must never be
    # presented as actual clinical findings.
    clinical_alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Clinical alerts for this scenario's hypothetical parameter values.",
    )


class ScenarioResponse(BaseModel):
    patient_id: str
    baseline_twin_risk_score: float
    # Clinical alerts evaluated for the patient's OBSERVED baseline values.
    # These are marked is_hypothetical=False.
    # A scenario with lower-risk values does NOT remove a baseline alert.
    baseline_clinical_alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Clinical alerts for the patient's observed baseline measurements.",
    )
    scenarios: list[ScenarioResult]
    disclaimer: str = (
        "These figures are TwinEngine composite scores derived from a rule-based model. "
        "They are NOT calibrated clinical probabilities. "
        "Clinical alerts are evaluated independently of the risk score. "
        "Do not use them as a substitute for professional medical advice."
    )
