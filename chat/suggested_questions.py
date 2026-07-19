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


def get_public_suggested_questions() -> dict:
    """
    Returns the suggested questions for the public landing page.
    """
    return {
        "questions": PUBLIC_SUGGESTED_QUESTIONS
    }


if __name__ == "__main__":
    print("=== Suggested Questions Test ===\n")
    result = get_public_suggested_questions()
    for i, q in enumerate(result["questions"], 1):
        print(f"{i}. {q}")