def compare(current_risk, simulations):

    best = min(

        simulations,

        key=lambda x: x["risk_after"]

    )

    improvement = (

        current_risk

        -

        best["risk_after"]

    )

    return {

        "current_risk":

            current_risk,

        "best_scenario":

            best["scenario"],

        "projected_risk":

            best["risk_after"],

        "risk_reduction":

            round(

                improvement,

                2

            )

    }