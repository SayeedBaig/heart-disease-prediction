from digital_twin.monitoring.monitor import monitor_patient

from digital_twin.evaluation.edge_cases import patients


result = monitor_patient(

    patients

)

print(result)