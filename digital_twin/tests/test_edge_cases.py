from digital_twin.engine.twin_engine import TwinEngine

from digital_twin.evaluation.edge_cases import patients


def test_risk():

    engine = TwinEngine()

    for patient in patients:

        risk = engine.calculate_risk(patient)

        assert risk >= 0


def test_confidence():

    engine = TwinEngine()

    for patient in patients:

        confidence = engine.calculate_confidence(patient)

        assert 0 <= confidence <= 1