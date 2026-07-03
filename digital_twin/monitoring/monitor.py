from digital_twin.engine.twin_engine import TwinEngine

engine = TwinEngine()


def monitor_patient(visits):

    risks = []

    for patient in visits:

        risks.append(

            engine.calculate_risk(

                patient

            )

        )

    trend = (

        "Improving"

        if risks[-1]

        < risks[0]

        else "Stable"

    )

    return {

        "history":

            risks,

        "latest":

            risks[-1],

        "trend":

            trend

    }