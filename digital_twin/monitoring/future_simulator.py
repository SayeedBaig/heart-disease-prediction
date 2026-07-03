from digital_twin.engine.twin_engine import TwinEngine

engine = TwinEngine()


def simulate_future(patient):

    current = engine.calculate_risk(

        patient

    )

    future = max(

        current - 12,

        0

    )

    return {

        "current":

            current,

        "future":

            future,

        "improvement":

            round(

                current - future,

                2

            )

    }