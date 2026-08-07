"""
Suggested Questions
Author: Akash
Module 1 (landing page) / Module 7 (suggested questions)

Purpose:
    Static list of common questions shown on the public landing page
    so users have a starting point instead of a blank chat box.
"""

PUBLIC_SUGGESTED_QUESTIONS = [
    "What causes chest pain?",
    "Can diabetes increase heart disease risk?",
    "What is ECG?",
    "Should I visit a cardiologist?",
    "How does high BP affect the heart?",
    "How often should I exercise?",
    "Can stress increase heart disease?",
]

DOCTOR_SUGGESTED_QUESTIONS = [
    "Why is this prediction High Risk?",
    "Explain the ECG findings for this patient.",
    "What diagnostic tests are recommended next?",
    "What are the current clinical guidelines for this condition?",
    "What treatment options should be considered?",
    "Is there recent research relevant to this case?",
]


def get_doctor_suggested_questions() -> dict:
    """
    Returns the suggested clinical questions for the doctor dashboard.
    """
    return {
        "questions": DOCTOR_SUGGESTED_QUESTIONS
    }

def get_public_suggested_questions() -> dict:
    """
    Returns the suggested questions for the public landing page.
    """
    return {
        "questions": PUBLIC_SUGGESTED_QUESTIONS
    }


if __name__ == "__main__":
    print("=== Suggested Questions Test ===\n")

    print("Public questions:")
    result = get_public_suggested_questions()
    for i, q in enumerate(result["questions"], 1):
        print(f"{i}. {q}")

    print("\nDoctor questions:")
    result = get_doctor_suggested_questions()
    for i, q in enumerate(result["questions"], 1):
        print(f"{i}. {q}")