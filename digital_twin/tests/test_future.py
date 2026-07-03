from digital_twin.monitoring.future_simulator import simulate_future

from digital_twin.evaluation.edge_cases import patients


result = simulate_future(

    patients[0]

)

print(result)