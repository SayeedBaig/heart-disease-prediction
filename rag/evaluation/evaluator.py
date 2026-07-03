"""
RAG Evaluator
Author: Akash
Week: 5 — RAG Improvement & Knowledge Validation

Purpose:
    Validates retrieval quality by testing the RAG pipeline
    against known clinical scenarios and scoring the results.
"""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from rag.retriever.retriever import RAGRetriever, build_query


# ── Test Cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "name": "High risk + MI + Low EF",
        "input": {"risk_level": "High", "ecg_class": "MI", "ef_value": 35.0},
        "expected_keywords": ["myocardial", "infarction", "ejection", "risk"],
        "expected_categories": ["risk_factors", "echo_findings"]
    },
    {
        "name": "Low risk + NORM",
        "input": {"risk_level": "Low", "ecg_class": "NORM", "ef_value": 62.0},
        "expected_keywords": ["prevention", "lifestyle", "cardiovascular"],
        "expected_categories": ["risk_factors", "lifestyle_recommendations"]
    },
    {
        "name": "Medium risk + STTC",
        "input": {"risk_level": "Medium", "ecg_class": "STTC", "ef_value": 50.0},
        "expected_keywords": ["risk", "management", "treatment"],
        "expected_categories": ["risk_factors"]
    },
    {
        "name": "High risk + HYP",
        "input": {"risk_level": "High", "ecg_class": "HYP", "ef_value": 45.0},
        "expected_keywords": ["hypertension", "hypertrophy", "risk"],
        "expected_categories": ["risk_factors"]
    },
    {
        "name": "Medium risk + CD",
        "input": {"risk_level": "Medium", "ecg_class": "CD", "ef_value": None},
        "expected_keywords": ["conduction", "management", "cardiovascular"],
        "expected_categories": ["risk_factors"]
    }
]


# ── Scoring Functions ─────────────────────────────────────────────────────────

def score_keyword_coverage(chunks: list, keywords: list) -> float:
    """
    Check what percentage of expected keywords appear in retrieved chunks.
    """
    combined_text = " ".join([c["text"].lower() for c in chunks])
    found = sum(1 for kw in keywords if kw.lower() in combined_text)
    return round(found / len(keywords), 3) if keywords else 0.0


def score_category_coverage(chunks: list, expected_categories: list) -> float:
    """
    Check what percentage of expected categories are represented.
    """
    retrieved_categories = set(c["category"] for c in chunks)
    expected_set = set(expected_categories)
    overlap = retrieved_categories & expected_set
    return round(len(overlap) / len(expected_set), 3) if expected_set else 0.0


def score_confidence(chunks: list) -> float:
    """
    Calculate average confidence score of retrieved chunks.
    """
    if not chunks:
        return 0.0
    avg = sum(c["confidence"] for c in chunks) / len(chunks)
    return round(avg, 3)


def check_duplicates(chunks: list) -> int:
    """
    Count how many duplicate sources appear in retrieved chunks.
    """
    sources = [c["source"] for c in chunks]
    return len(sources) - len(set(sources))


# ── Evaluator ─────────────────────────────────────────────────────────────────

def evaluate_retrieval(retriever: RAGRetriever) -> dict:
    """
    Run all test cases and score retrieval quality.

    Returns:
        dict with per-test results and overall summary
    """
    print("=== RAG Evaluation ===\n")

    results = []
    total_keyword = 0
    total_category = 0
    total_confidence = 0
    total_duplicates = 0

    for test in TEST_CASES:
        print(f"Testing: {test['name']}")

        inp = test["input"]
        retrieval = retriever.retrieve_for_prediction(
            risk_level=inp["risk_level"],
            ecg_class=inp.get("ecg_class"),
            ef_value=inp.get("ef_value")
        )
        chunks = retrieval["chunks"]

        keyword_score = score_keyword_coverage(chunks, test["expected_keywords"])
        category_score = score_category_coverage(chunks, test["expected_categories"])
        confidence_score = score_confidence(chunks)
        duplicates = check_duplicates(chunks)

        result = {
            "test": test["name"],
            "chunks_retrieved": len(chunks),
            "keyword_coverage": keyword_score,
            "category_coverage": category_score,
            "avg_confidence": confidence_score,
            "duplicate_sources": duplicates,
            "sources": [f"{c['source']} p.{c['page']}" for c in chunks]
        }

        results.append(result)

        total_keyword += keyword_score
        total_category += category_score
        total_confidence += confidence_score
        total_duplicates += duplicates

        print(f"  Keyword coverage:  {keyword_score:.1%}")
        print(f"  Category coverage: {category_score:.1%}")
        print(f"  Avg confidence:    {confidence_score:.3f}")
        print(f"  Duplicate sources: {duplicates}")
        print()

    n = len(TEST_CASES)
    summary = {
        "total_tests": n,
        "avg_keyword_coverage": round(total_keyword / n, 3),
        "avg_category_coverage": round(total_category / n, 3),
        "avg_confidence": round(total_confidence / n, 3),
        "total_duplicates_found": total_duplicates,
        "overall_quality": "GOOD" if (total_keyword / n) >= 0.7 else "NEEDS IMPROVEMENT"
    }

    print("=== Summary ===")
    print(f"Avg keyword coverage:  {summary['avg_keyword_coverage']:.1%}")
    print(f"Avg category coverage: {summary['avg_category_coverage']:.1%}")
    print(f"Avg confidence:        {summary['avg_confidence']:.3f}")
    print(f"Total duplicates:      {summary['total_duplicates_found']}")
    print(f"Overall quality:       {summary['overall_quality']}")

    return {"results": results, "summary": summary}


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    retriever = RAGRetriever()
    report = evaluate_retrieval(retriever)

    # Save evaluation report
    output_path = "rag/evaluation/evaluation_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation report saved to: {output_path}")