from copy import deepcopy

from digital_twin.models.patient_profile import PatientProfile


class TwinEngine:

    def calculate_risk(self, patient: PatientProfile):

        risk = getattr(
            patient,
            "fusion_risk_percentage",
            50
        )

        systolic_bp = (
            getattr(patient, "systolic_bp", None)
            or 120
        )

        cholesterol = (
            getattr(patient, "cholesterol", None)
            or 200
        )

        glucose = (
            getattr(patient, "glucose", None)
            or 100
        )

        bmi = (
            getattr(patient, "bmi", None)
            or 25
        )

        smoking_status = (
            getattr(patient, "smoking_status", None)
        )

        if smoking_status is None:
            smoking_status = False

        exercise_level = (
            getattr(patient, "exercise_level", None)
        )

        if exercise_level is None:
            exercise_level = "Moderate"

        if systolic_bp < 140:
            risk -= 4

        if cholesterol < 240:
            risk -= 4

        if glucose < 140:
            risk -= 3

        if bmi < 30:
            risk -= 3

        if not smoking_status:
            risk -= 3

        if exercise_level == "High":
            risk -= 5

        risk = max(0, risk)

        return round(risk, 2)

    def calculate_confidence(
        self,
        patient: PatientProfile
    ):

        fields = [

            getattr(
                patient,
                "systolic_bp",
                None
            ),

            getattr(
                patient,
                "cholesterol",
                None
            ),

            getattr(
                patient,
                "glucose",
                None
            ),

            getattr(
                patient,
                "bmi",
                None
            ),

            getattr(
                patient,
                "exercise_level",
                None
            )

        ]

        available = sum(

            value is not None

            for value in fields

        )

        total = len(fields)

        return round(

            available / total,

            2

        )

    def simulate(
        self,
        patient: PatientProfile
    ):

        risk = self.calculate_risk(
            patient
        )

        confidence = self.calculate_confidence(
            patient
        )

        return {

            "risk": risk,

            "confidence": confidence

        }

    # -----------------------------
    # Phase 2 Features
    # -----------------------------

    def simulate_custom(
        self,
        patient: PatientProfile,
        changes: dict
    ):
        """
        Apply doctor-defined changes to a copy of the patient
        and calculate the projected future risk.
        """

        updated = deepcopy(patient)

        for field, value in changes.items():

            if hasattr(updated, field):

                setattr(updated, field, value)

        return {

            "updated_patient": updated,

            "risk": self.calculate_risk(updated),

            "confidence": self.calculate_confidence(updated)

        }

    def compare_patients(
        self,
        original: PatientProfile,
        modified: PatientProfile
    ):
        """
        Compare two patient states.
        """

        current_risk = self.calculate_risk(original)

        future_risk = self.calculate_risk(modified)

        reduction = max(

            0,

            current_risk - future_risk

        )

        improvement_percentage = 0

        if current_risk > 0:

            improvement_percentage = (

                reduction / current_risk

            ) * 100

        return {

            "current_risk": round(

                current_risk,

                2

            ),

            "future_risk": round(

                future_risk,

                2

            ),

            "risk_reduction": round(

                reduction,

                2

            ),

            "improvement_percentage": round(

                improvement_percentage,

                2

            )

        }

    def generate_report(
        self,
        patient: PatientProfile
    ):
        """
        Returns a complete Digital Twin report.
        """

        return {

            "patient_id": patient.patient_id,

            "risk": self.calculate_risk(patient),

            "confidence": self.calculate_confidence(patient),

            "blood_pressure": f"{patient.systolic_bp}/{patient.diastolic_bp}",

            "cholesterol": patient.cholesterol,

            "glucose": patient.glucose,

            "bmi": patient.bmi,

            "weight": patient.weight,

            "exercise_level": patient.exercise_level,

            "smoking": patient.smoking_status

        }