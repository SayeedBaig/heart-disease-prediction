from copy import deepcopy


def simulate(patient):

    updated = deepcopy(patient)

    updated.systolic_bp = max(
        0,
        patient.systolic_bp - 15
    )

    updated.diastolic_bp = max(
        0,
        patient.diastolic_bp - 10
    )

    updated.cholesterol = max(
        0,
        patient.cholesterol - 30
    )

    return updated