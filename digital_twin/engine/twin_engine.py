from digital_twin.models.twin_model import PatientTwin

from digital_twin.simulation.simulator import simulate


class TwinEngine:


    def run(self, data):

        twin = PatientTwin(data)

        simulations = simulate(twin)

        return {

            "baseline_risk":

                round(twin.risk,3),

            "simulations":

                simulations

        }