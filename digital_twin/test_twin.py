from digital_twin.engine.twin_engine import TwinEngine


patient = {

    "age":55,

    "bp":150,

    "cholesterol":245,

    "bmi":31,

    "smoking":1,

    "risk_score":0.81

}


engine = TwinEngine()

result = engine.run(patient)

print(result)