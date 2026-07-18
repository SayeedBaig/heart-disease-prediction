from copy import deepcopy

from digital_twin.engine.twin_engine import TwinEngine


class ScenarioManager:
    """
    Stores and compares multiple doctor-created scenarios.
    """

    def __init__(self):
        self.engine = TwinEngine()
        self.scenarios = {}

    def save_scenario(self, name, patient):
        """
        Save a patient state as a named scenario.
        """
        self.scenarios[name] = deepcopy(patient)

    def delete_scenario(self, name):
        """
        Remove a saved scenario.
        """
        if name in self.scenarios:
            del self.scenarios[name]

    def list_scenarios(self):
        """
        Return all scenario names.
        """
        return list(self.scenarios.keys())

    def simulate_all(self):
        """
        Calculate the projected risk for every saved scenario.
        """
        results = []

        for name, patient in self.scenarios.items():
            risk = self.engine.calculate_risk(patient)

            results.append({
                "scenario": name,
                "risk": risk,
                "confidence": self.engine.calculate_confidence(patient)
            })

        return results