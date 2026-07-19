"""
Safety Guardrails — Public AI Health Assistant
Author: Akash
Module 1 — Public AI Health Assistant

Purpose:
    The public chatbot must NEVER diagnose, predict, or assess an
    individual's personal risk/condition. It only educates.
    This module detects diagnostic-style questions and blocks them
    before they ever reach the generator.
"""

import re

# Phrases that indicate the user wants a personal diagnosis/prediction
# rather than general education.
BLOCKED_PATTERNS = [
    r"\bdo i have\b",
    r"\bam i at risk\b",
    r"\bam i having\b",
    r"\bwhat('s| is) my risk\b",
    r"\bdiagnos(e|is|ing) me\b",
    r"\bpredict my\b",
    r"\bcan you tell if i\b",
    r"\bis my heart\b.*\b(okay|ok|fine|healthy)\b",
    r"\bshould i be worried about my\b",
    r"\bcheck my (heart|risk|condition)\b",
]

DIAGNOSTIC_REDIRECT_MESSAGE = (
    "I can't assess your personal health or provide a diagnosis — "
    "I'm only able to share general medical education. "
    "For anything about your own symptoms or risk, please consult "
    "a doctor or cardiologist. They can properly evaluate your situation."
)


def is_diagnostic_request(user_query: str) -> bool:
    """
    Returns True if the query is asking for a personal diagnosis
    or prediction rather than general information.
    """
    query_lower = user_query.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, query_lower):
            return True
    return False


def check_query_safety(user_query: str) -> dict:
    """
    Main safety check called before any retrieval/generation.

    Returns:
        dict with 'safe' (bool) and 'redirect_message' (str or None)
    """
    if is_diagnostic_request(user_query):
        return {
            "safe": False,
            "redirect_message": DIAGNOSTIC_REDIRECT_MESSAGE
        }

    return {
        "safe": True,
        "redirect_message": None
    }


if __name__ == "__main__":
    print("=== Safety Guardrail Test ===\n")

    test_cases = [
        ("What causes chest pain?", True),
        ("Do I have heart disease?", False),
        ("Am I at risk of a heart attack?", False),
        ("How does high BP affect the heart?", True),
        ("Can you tell if I have a heart problem?", False),
        ("Should I visit a cardiologist?", True),
        ("Check my heart condition", False),
    ]

    for query, expected_safe in test_cases:
        result = check_query_safety(query)
        status = "PASS" if result["safe"] == expected_safe else "FAIL"
        print(f"[{status}] '{query}' -> safe={result['safe']}")
        if not result["safe"]:
            print(f"       Redirect: {result['redirect_message']}")