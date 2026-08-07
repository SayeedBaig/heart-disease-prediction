"""
History Context Builder
Author: Akash
Module 8 — RAG Integration

Purpose:
    Formats a patient's prediction history (from Sayeed's database)
    into readable context for the generator, so doctors can ask
    comparison questions like "Why did risk increase since last visit?"

    Input format expected from Sayeed's history API:
    [
        {"date": "2026-05-01", "risk_level": "Medium", "ecg_class": "Normal", "ef_value": 55.0},
        {"date": "2026-07-01", "risk_level": "High", "ecg_class": "MI", "ef_value": 35.0}
    ]
    (most recent visit last)
"""


def format_history_for_prompt(history: list) -> str:
    """
    Converts a list of past prediction records into a readable
    text block for the LLM prompt.
    """
    if not history or len(history) == 0:
        return ""

    lines = ["PATIENT VISIT HISTORY (oldest to most recent):"]
    for i, visit in enumerate(history, 1):
        lines.append(
            f"Visit {i} ({visit.get('date', 'unknown date')}): "
            f"Risk={visit.get('risk_level', 'N/A')}, "
            f"ECG={visit.get('ecg_class', 'N/A')}, "
            f"EF={visit.get('ef_value', 'N/A')}"
        )

    return "\n".join(lines) + "\n"


def get_latest_vs_previous(history: list) -> dict:
    """
    Returns the two most recent visits for direct comparison,
    or None if there's less than 2 visits.
    """
    if not history or len(history) < 2:
        return None

    return {
        "previous": history[-2],
        "current": history[-1]
    }


if __name__ == "__main__":
    print("=== History Context Builder Test ===\n")

    sample_history = [
        {"date": "2026-05-01", "risk_level": "Medium", "ecg_class": "Normal", "ef_value": 55.0},
        {"date": "2026-07-01", "risk_level": "High", "ecg_class": "MI", "ef_value": 35.0}
    ]

    formatted = format_history_for_prompt(sample_history)
    print("Formatted history:")
    print(formatted)

    comparison = get_latest_vs_previous(sample_history)
    print(f"Comparison data: {comparison}")