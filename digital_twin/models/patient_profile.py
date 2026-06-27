from dataclasses import dataclass


@dataclass
class PatientProfile:

    patient_id: str

    age: int
    gender: str

    systolic_bp: float
    diastolic_bp: float

    cholesterol: float
    glucose: float

    bmi: float
    weight: float

    smoking_status: bool = False

    alcohol_consumption: bool = False

    exercise_level: str = "Moderate"

    clinical_risk_score: float = 0.0

    ecg_risk_score: float = 0.0

    echo_risk_score: float = 0.0

    fusion_risk_percentage: float = 0.0