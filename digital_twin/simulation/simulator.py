import copy

from scenarios.bp_scenario import lower_bp
from scenarios.cholesterol_scenario import improve_cholesterol
from scenarios.weight_scenario import reduce_weight
from scenarios.smoking_scenario import smoking_cessation


def simulate(patient):

    baseline = patient.risk

    results = []

    scenario_list = [

        ("Lower BP", lower_bp),

        ("Improve Cholesterol",
         improve_cholesterol),

        ("Weight Reduction",
         reduce_weight),

        ("Smoking Cessation",
         smoking_cessation)

    ]


    for name, func in scenario_list:

        p = copy.deepcopy(patient)

        p = func(p)

        results.append(

            {

                "scenario": name,

                "risk":

                    round(p.risk,3),

                "change":

                    round(

                    (

                    (baseline-p.risk)

                    / baseline

                    )*100

                    ,1)

            }

        )

    return results