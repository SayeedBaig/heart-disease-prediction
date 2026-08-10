from copy import deepcopy

from digital_twin.scenarios.medication import simulate as medication

from digital_twin.scenarios.exercise import simulate as exercise

from digital_twin.scenarios.diabetes import simulate as diabetes

from digital_twin.scenarios.smoking_scenario import simulate as smoking


def simulate(patient):

    updated = deepcopy(patient)

    updated = medication(updated)

    updated = exercise(updated)

    updated = diabetes(updated)

    updated = smoking(updated)

    updated.alcohol_consumption = False

    return updated