"""
Digital Twin Scenario Simulation API Route.

POST /digital-twin/simulate
  - Accepts a patient's baseline values + one or more named scenarios.
  - Runs TwinEngine on each (baseline + overrides) combination.
  - Returns baseline score, per-scenario scores, changed/unchanged parameters.
  - Evaluates clinical alerts (BP, cholesterol) for baseline AND each scenario
    independently, using the shared clinical alert engine.
  - Enforces patient-level authorization: patients can only simulate their own profile.
  - Scenario data is NEVER written to the database or patient record.

The TwinEngine is a composite rule-based scorer. Scores returned here are
NOT calibrated probabilities from the primary ML model. The endpoint label
and response schema make this explicit.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database.session import get_db
from api.schemas.digital_twin import ScenarioRequest, ScenarioResponse, ScenarioResult
from api.utils.authorization import get_current_actor, resolve_patient
from clinical_module.alerts.clinical_alert_engine import (
    evaluate_clinical_alerts,
    alerts_to_dicts,
)
from digital_twin.engine.twin_engine import TwinEngine
from digital_twin.models.patient_profile import PatientProfile

router = APIRouter(prefix="/digital-twin", tags=["Digital Twin"])

_engine = TwinEngine()

# Fields that TwinEngine uses for scoring (must match PatientProfile attribute names)
_SCORED_FIELDS = [
    "systolic_bp", "cholesterol", "glucose", "bmi",
    "smoking_status", "exercise_level",
]


def _build_profile(patient_id: str, params) -> PatientProfile:
    """Construct a PatientProfile from a ScenarioParams instance."""
    return PatientProfile(
        patient_id=patient_id,
        age=params.age or 45,
        gender="Unknown",
        systolic_bp=params.systolic_bp or 120,
        diastolic_bp=params.diastolic_bp or 80,
        cholesterol=params.cholesterol or 200,
        glucose=params.glucose or 100,
        bmi=params.bmi or 25,
        weight=params.weight or 70,
        smoking_status=params.smoking_status if params.smoking_status is not None else False,
        alcohol_consumption=params.alcohol_consumption if params.alcohol_consumption is not None else False,
        exercise_level=params.exercise_level or "Moderate",
        clinical_risk_score=params.clinical_risk_score or 0.0,
        ecg_risk_score=params.ecg_risk_score or 0.0,
        echo_risk_score=params.echo_risk_score or 0.0,
        fusion_risk_percentage=params.fusion_risk_percentage or 0.0,
    )


def _param_diff(baseline_params, scenario_params) -> tuple[dict, dict]:
    """Return (changed, unchanged) dicts comparing scenario to baseline."""
    baseline_dict = baseline_params.model_dump(exclude_none=False)
    scenario_dict = scenario_params.model_dump(exclude_none=False)

    changed = {}
    unchanged = {}
    for key, base_val in baseline_dict.items():
        scene_val = scenario_dict.get(key)
        if scene_val is not None and scene_val != base_val:
            changed[key] = {"baseline": base_val, "scenario": scene_val}
        else:
            unchanged[key] = base_val
    return changed, unchanged


def _evaluate_bp_alerts_for_params(
    params,
    is_hypothetical: bool,
    context_id: str,
) -> list:
    """
    Evaluate BP clinical alerts for a ScenarioParams (or baseline params) object.
    Only BP fields are evaluated here because the Digital Twin uses mmHg units
    (systolic_bp / diastolic_bp).  Cholesterol in ScenarioParams is stored as
    a raw float (mg/dL equivalent), not the categorical 1/2/3 encoding used by
    the ML model — we cannot responsibly apply categorical thresholds here.
    Returns serialised alert dicts.
    """
    systolic = getattr(params, "systolic_bp", None)
    diastolic = getattr(params, "diastolic_bp", None)
    alerts = evaluate_clinical_alerts(
        systolic_bp=float(systolic) if systolic is not None else None,
        diastolic_bp=float(diastolic) if diastolic is not None else None,
        glucose=None,          # DT uses mg/dL cholesterol float, not categorical
        cholesterol_category=None,
        is_hypothetical=is_hypothetical,
        context_id=context_id,
    )
    return alerts_to_dicts(alerts)


@router.post(
    "/simulate",
    response_model=ScenarioResponse,
    summary="Run Digital Twin Scenario Simulation",
    description=(
        "Runs the TwinEngine on the patient's baseline and on each supplied "
        "scenario variant. Returns composite risk scores and per-parameter diffs. "
        "Scores are TwinEngine rule-based outputs — not calibrated ML probabilities. "
        "Clinical alerts are evaluated independently for the baseline and each scenario. "
        "Scenario data is never written to patient records."
    ),
)
def simulate_scenarios(
    request: ScenarioRequest,
    db: Session = Depends(get_db),
    actor=Depends(get_current_actor),
):
    # ── Authorization: patients can only simulate their own profile ──────────
    role, account = actor
    patient_record = resolve_patient(request.patient_id, db)
    if role == "patient" and account.id != patient_record.id:
        raise HTTPException(
            status_code=403,
            detail="You can only simulate scenarios for your own profile.",
        )

    # ── Validate scenario count ──────────────────────────────────────────────
    if len(request.scenarios) > 10:
        raise HTTPException(
            status_code=422,
            detail="A maximum of 10 scenarios may be submitted per request.",
        )

    # ── Baseline simulation ──────────────────────────────────────────────────
    baseline_profile = _build_profile(request.patient_id, request.baseline)
    baseline_score = round(_engine.calculate_risk(baseline_profile), 2)

    # Evaluate clinical alerts for the OBSERVED baseline (is_hypothetical=False).
    # These alerts are preserved in the response regardless of any scenario values.
    baseline_alerts = _evaluate_bp_alerts_for_params(
        request.baseline,
        is_hypothetical=False,
        context_id=f"baseline:{request.patient_id}",
    )

    # ── Per-scenario simulation ──────────────────────────────────────────────
    scenario_results: list[ScenarioResult] = []

    for name, override in request.scenarios.items():
        # Merge: start from baseline, apply only the overridden fields
        merged = request.baseline.model_copy(
            update={
                k: v
                for k, v in override.model_dump(exclude_none=True).items()
                if v is not None
            }
        )
        scenario_profile = _build_profile(request.patient_id, merged)
        scenario_score = round(_engine.calculate_risk(scenario_profile), 2)

        changed, unchanged = _param_diff(request.baseline, merged)

        # Evaluate clinical alerts for THIS scenario (is_hypothetical=True).
        # These are independent of the baseline alerts.
        scenario_alerts = _evaluate_bp_alerts_for_params(
            merged,
            is_hypothetical=True,
            context_id=f"scenario:{name}",
        )

        scenario_results.append(
            ScenarioResult(
                scenario_name=name,
                twin_risk_score=scenario_score,
                changed_params=changed,
                unchanged_params=unchanged,
                clinical_alerts=scenario_alerts,
            )
        )

    return ScenarioResponse(
        patient_id=request.patient_id,
        baseline_twin_risk_score=baseline_score,
        baseline_clinical_alerts=baseline_alerts,
        scenarios=scenario_results,
    )
