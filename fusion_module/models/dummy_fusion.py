from fusion_module.utils.shared_memory import SharedMemory


class DummyFusion:
    def predict(self, echo_result, ecg_result, clinical_result):
        total = echo_result + ecg_result + clinical_result

        if total == 0:
            return "Low"
        elif total == 1:
            return "Medium"
        else:
            return "High"


if __name__ == "__main__":
    # Step 1: Create shared memory
    memory = SharedMemory()

    # Step 2: Simulate module outputs
    memory.store("echo", {"result": 1})
    memory.store("ecg", {"result": 0})
    memory.store("clinical", {"result": 1})

    # Step 3: Retrieve data
    echo_result = memory.get("echo")["result"]
    ecg_result = memory.get("ecg")["result"]
    clinical_result = memory.get("clinical")["result"]

    # Step 4: Apply fusion
    fusion = DummyFusion()
    final_result = fusion.predict(echo_result, ecg_result, clinical_result)

    print("Final Risk Level:", final_result)