"""Download/cache and smoke-test CardioAI inference assets before serving."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from clinical_module.agent.clinical_agent import clinical_agent_with_status
from ecg_module.agent.ecg_agent import ECGAgent
from echo_module.models.final_echo_model import build_echo_model
from echo_module.models.final_echo_model import RISK_MAP
from fusion_module.models.fusion_model import FusionModel
from fusion_module.rag.generator import generate_explanation
from rag.retriever.retriever import (
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_REVISION,
    RAGRetriever,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ECHO_CHECKPOINT = PROJECT_ROOT / "echo_module" / "models" / "echo_best_model.keras"

SYNTHETIC_PATIENT = {
    "age": 50,
    "gender": 2,
    "height": 170,
    "weight": 70,
    "ap_hi": 130,
    "ap_lo": 85,
    "smoke": 0,
    "alco": 0,
    "active": 1,
    "cholesterol": 2,
    "gluc": 1,
}


def main() -> None:
    clinical_result, clinical_source = clinical_agent_with_status(SYNTHETIC_PATIENT)
    if clinical_source != "ml":
        raise RuntimeError("Clinical saved model did not produce an ML prediction.")

    ecg_result = ECGAgent().predict(np.zeros((12, 1000), dtype=np.float32))
    if "Class scores:" not in ecg_result["reason"]:
        raise RuntimeError("ECG checkpoint did not produce a model prediction.")

    echo_model = build_echo_model(weights_path=str(ECHO_CHECKPOINT))
    echo_result = echo_model(
        np.zeros((1, 224, 224, 3), dtype=np.float32),
        training=False,
    ).numpy()
    if echo_result.shape != (1, 4) or not np.isfinite(echo_result).all():
        raise RuntimeError("Echo checkpoint did not produce a valid model prediction.")

    echo_class = int(np.argmax(echo_result[0]))
    echo_output = {
        "level": RISK_MAP[echo_class],
        "score": float(np.max(echo_result[0])),
        "reason": "Synthetic echo checkpoint inference",
    }
    fusion_output = FusionModel().predict(echo_output, ecg_result, clinical_result)
    explanation = generate_explanation(
        fusion_output,
        echo_output,
        ecg_result,
        clinical_result,
    )
    if fusion_output["final_level"] not in {"Low", "Medium", "High"}:
        raise RuntimeError("Fusion model did not produce a valid risk level.")
    if not explanation.get("explanation"):
        raise RuntimeError("Fusion explanation pipeline returned no summary.")

    from digital_twin.models.patient_profile import PatientProfile
    from digital_twin.simulation.simulator import TwinSimulator

    twin_patient = PatientProfile(
        patient_id="synthetic-deployment-smoke",
        age=50,
        gender="Unknown",
        systolic_bp=130,
        diastolic_bp=85,
        cholesterol=200,
        glucose=100,
        bmi=24.2,
        weight=70,
        smoking_status=False,
        exercise_level="Moderate",
        fusion_risk_percentage=float(fusion_output["risk_percentage"]),
    )
    twin_scenarios = TwinSimulator().run_all(twin_patient)
    if len(twin_scenarios) != 8 or any("risk_after" not in item for item in twin_scenarios):
        raise RuntimeError("Digital Twin scenarios did not all produce risk outputs.")

    from api.services.digital_twin_service import DigitalTwinService

    twin_service_result = DigitalTwinService().simulate_future(
        {**SYNTHETIC_PATIENT, "patient_id": "synthetic-deployment-smoke"},
        {"fusion": fusion_output},
    )
    if twin_service_result.get("status") != "success" or len(
        twin_service_result.get("simulations", [])
    ) != 8:
        raise RuntimeError("Prediction service Digital Twin integration failed.")

    retriever = RAGRetriever()
    retrieval = retriever.retrieve_for_prediction("High", "MI", 35.0)
    if not retrieval["chunks"]:
        raise RuntimeError("RAG assets loaded but retrieval returned no chunks.")

    print(
        json.dumps(
            {
                "clinical": "ml_pass",
                "ecg": "pass",
                "echo": "pass",
                "fusion": "pass",
                "digital_twin_scenarios": len(twin_scenarios),
                "digital_twin_service": "pass",
                "rag": "pass",
                "embedding_model": EMBEDDING_MODEL,
                "embedding_model_revision": EMBEDDING_MODEL_REVISION,
                "rag_index_vectors": int(retriever.index.ntotal),
                "rag_chunks_returned": len(retrieval["chunks"]),
                "synthetic_clinical_level": clinical_result["level"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
