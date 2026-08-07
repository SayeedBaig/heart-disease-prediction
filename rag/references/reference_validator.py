"""
Medical Reference Validator
Author: Akash
Final Phase — Advanced RAG & Medical Intelligence

Purpose:
    Validates and formats medical references from retrieved chunks
    for display in both doctor and patient reports.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))


# ── Known Guideline Registry ──────────────────────────────────────────────────

GUIDELINE_REGISTRY = {
    "ACC_AHA_2019_Primary_Prevention.pdf": {
        "full_name": "ACC/AHA 2019 Guideline on Primary Prevention of CVD",
        "publisher": "American College of Cardiology / American Heart Association",
        "year": 2019,
        "url": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000000678"
    },
    "ESC_2021_CVD_Prevention.pdf": {
        "full_name": "ESC 2021 Guidelines on Cardiovascular Disease Prevention",
        "publisher": "European Society of Cardiology",
        "year": 2021,
        "url": "https://www.escardio.org/Guidelines"
    },
    "AHA_2021_Heart_Stroke_Statistics.pdf": {
        "full_name": "AHA 2021 Heart Disease and Stroke Statistics",
        "publisher": "American Heart Association",
        "year": 2021,
        "url": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000000950"
    },
    "WHO_2020_HEARTS_Package.pdf": {
        "full_name": "WHO HEARTS Technical Package",
        "publisher": "World Health Organization",
        "year": 2020,
        "url": "https://www.who.int/publications/i/item/who-cvd-19-1"
    },
    "AHA_2019_Echo_Appropriate_Use.pdf": {
        "full_name": "AHA 2019 Appropriate Use Criteria for Echocardiography",
        "publisher": "American Heart Association",
        "year": 2019,
        "url": "https://www.jacc.org/doi/10.1016/j.jacc.2011.01.002"
    },
    "PTB_XL_2020_ECG_Documentation.pdf": {
        "full_name": "PTB-XL: A Large Publicly Available ECG Dataset",
        "publisher": "PhysioNet",
        "year": 2020,
        "url": "https://physionet.org/content/ptb-xl/1.0.3/"
    }
}


# ── Reference Formatter ───────────────────────────────────────────────────────

def format_reference(chunk: dict) -> dict:
    """
    Format a retrieved chunk into a proper medical reference.

    Args:
        chunk: Retrieved chunk dict with source, page, category, confidence

    Returns:
        Formatted reference dict
    """
    source = chunk.get("source", "Unknown")
    registry_entry = GUIDELINE_REGISTRY.get(source, {})

    return {
        "source_file": source,
        "full_name": registry_entry.get("full_name", source),
        "publisher": registry_entry.get("publisher", "Unknown"),
        "year": registry_entry.get("year", "Unknown"),
        "page": chunk.get("page", "N/A"),
        "category": chunk.get("category", "general"),
        "confidence": chunk.get("confidence", 0),
        "url": registry_entry.get("url", ""),
        "excerpt": chunk.get("text", "")[:150] + "..."
    }


def validate_references(chunks: list) -> dict:
    """
    Validate and format all retrieved chunks as proper references.

    Args:
        chunks: List of retrieved chunks from RAGRetriever

    Returns:
        dict with formatted references and validation summary
    """
    formatted = []
    known_sources = 0
    unknown_sources = 0

    for chunk in chunks:
        ref = format_reference(chunk)
        formatted.append(ref)

        if chunk.get("source") in GUIDELINE_REGISTRY:
            known_sources += 1
        else:
            unknown_sources += 1

    # Sort by confidence descending
    formatted.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "references": formatted,
        "total_references": len(formatted),
        "known_guidelines": known_sources,
        "unknown_sources": unknown_sources,
        "validation_status": "VERIFIED" if unknown_sources == 0 else "PARTIAL"
    }


def format_for_report(chunks: list, report_type: str = "doctor") -> list:
    """
    Format references specifically for doctor or patient report.

    Args:
        chunks: Retrieved chunks
        report_type: "doctor" or "patient"

    Returns:
        List of formatted reference strings
    """
    validation = validate_references(chunks)
    refs = validation["references"]

    if report_type == "doctor":
        return [
            f"{r['full_name']} ({r['publisher']}, {r['year']}) — Page {r['page']}"
            for r in refs
        ]
    else:
        return [
            f"{r['full_name']} ({r['year']})"
            for r in refs
        ]


if __name__ == "__main__":
    print("=== Reference Validator Test ===\n")

    sample_chunks = [
        {
            "text": "Patients with high cardiovascular risk should receive statins.",
            "source": "ACC_AHA_2019_Primary_Prevention.pdf",
            "category": "risk_factors",
            "page": 14,
            "confidence": 0.85
        },
        {
            "text": "Lifestyle modification is recommended for all risk levels.",
            "source": "WHO_2020_HEARTS_Package.pdf",
            "category": "lifestyle_recommendations",
            "page": 22,
            "confidence": 0.72
        },
        {
            "text": "EF below 40% indicates heart failure with reduced ejection fraction.",
            "source": "AHA_2019_Echo_Appropriate_Use.pdf",
            "category": "echo_findings",
            "page": 8,
            "confidence": 0.68
        }
    ]

    result = validate_references(sample_chunks)

    print(f"Total references: {result['total_references']}")
    print(f"Known guidelines: {result['known_guidelines']}")
    print(f"Validation status: {result['validation_status']}\n")

    print("Doctor report references:")
    for ref in format_for_report(sample_chunks, "doctor"):
        print(f"  - {ref}")

    print("\nPatient report references:")
    for ref in format_for_report(sample_chunks, "patient"):
        print(f"  - {ref}")