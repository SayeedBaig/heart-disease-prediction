from echo_module.agent.echo_agent import echo_agent
from ecg_module.agent.ecg_agent import predict_ecg_signal as ecg_agent
from clinical_module.agent.clinical_agent import clinical_agent
from fusion_module.models.fusion_model import FusionModel
from fusion_module.rag.rag_pipeline import run_rag


class SystemPipeline:

    def __init__(self):
        self.fusion = FusionModel()

    def run(self, echo_input, ecg_input, clinical_input):

        # 🔹 Step 1: Agents
        echo_output = echo_agent(echo_input)
        ecg_output = ecg_agent(ecg_input)
        clinical_output = clinical_agent(clinical_input)

        print("\n--- Agent Outputs ---")
        print("Echo     :", echo_output)
        print("ECG      :", ecg_output)
        print("Clinical :", clinical_output)

        # 🔹 Step 2: Fusion
        fusion_output = self.fusion.predict(
            echo_output,
            ecg_output,
            clinical_output
        )

        rag_output = run_rag({
        "echo": echo_output,
        "ecg": ecg_output,
        "clinical": clinical_output,
        "fusion": fusion_output
    })

        return {
            "echo": echo_output,
            "ecg": ecg_output,
            "clinical": clinical_output,
            "fusion": fusion_output,
            "rag": rag_output
        }


# 🔻 MAIN (User Input comes HERE)
if __name__ == "__main__":

    pipeline = SystemPipeline()

    print("\n===== HEART DISEASE SYSTEM  =====\n")

    # 🔹 Clinical Input
    age = int(input("Age: "))
    gender = int(input("Gender (1=Female, 2=Male): "))
    height = float(input("Height: "))
    weight = float(input("Weight: "))
    ap_hi = int(input("Systolic BP: "))
    ap_lo = int(input("Diastolic BP: "))
    cholesterol = int(input("Cholesterol (1/2/3): "))
    gluc = int(input("Glucose (1/2/3): "))
    smoke = int(input("Smoke (0/1): "))
    alco = int(input("Alcohol (0/1): "))
    active = int(input("Active (0/1): "))

    clinical_input = {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active
    }

    # Dummy inputs (for now)
    # echo_input = "data/echo_samples/test1.avi"
    echo_input = input("\nEnter Echo video path: ")
    ecg_input = [[0.1] * 1000 for _ in range(12)]

    result = pipeline.run(echo_input, ecg_input, clinical_input)

    print("\n===== AGENT OUTPUTS =====")

print("\n[Echo Agent]")
print(f"Level : {result['echo'].get('level')}")
print(f"Score : {result['echo'].get('score')}")

print("\n[ECG Agent]")
print(f"Level : {result['ecg'].get('level')}")
print(f"Score : {result['ecg'].get('score')}")
print("NOTE: This is Using simulated ECG signal (real input can be added later)")

print("\n[Clinical Agent]")
print(f"Level : {result['clinical'].get('level')}")
print(f"Confidence : {result['clinical'].get('score')}")

print("\n===== FINAL RESULT =====")

print(f"Risk Level      : {result['fusion']['final_level']}")
print(f"Risk Percentage : {result['fusion']['risk_percentage']} %")

print("\n===== EXPLANATION =====")

print("Summary:")
print(result["rag"]["explanation"])

print("\nDetails:")
for item in result["rag"]["details"]:
    print("-", item)



    