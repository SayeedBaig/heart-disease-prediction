def generate_summary(simulations):

    best = min(

        simulations,

        key=lambda x: x["risk_after"]

    )

    summary = {

        "best_intervention":

            best["scenario"],

        "projected_risk":

            best["risk_after"],

        "all_simulations":

            simulations

    }

    return summary