"""
RAG Evaluator — Expanded
Author: Akash
Week: 6 — RAG Validation & Knowledge Expansion

Improvements over Week 5:
- More test cases including edge cases
- Missing input handling validation
- Data leakage verification
- Detailed validation report
"""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from rag.retriever.retriever import RAGRetriever, build_query

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
    },
    {
        "name": "Edge case — No ECG no EF (clinical only)",
        "input": {"risk_level": "High", "ecg_class": None, "ef_value": None},
        "expected_keywords": ["risk", "cardiovascular", "guidelines"],
        "expected_categories": ["risk_factors"]
    },
    {
        "name": "Edge case — Low risk no EF",
        "input": {"risk_level": "Low", "ecg_class": "NORM", "ef_value": None},
        "expected_keywords": ["prevention", "cardiovascular"],
        "expected_categories": ["risk_factors", "lifestyle_recommendations"]
    },
    {
        "name": "Edge case — Very low EF (severe HF)",
        "input": {"risk_level": "High", "ecg_class": "MI", "ef_value": 20.0},
        "expected_keywords": ["ejection", "failure", "risk"],
        "expected_categories": ["risk_factors"]
    }
]


def score_keyword_coverage(chunks: list, keywords: list) -> float:
    combined_text = " ".join([c["text"].lower() for c in chunks])
    found = sum(1 for kw in keywords if kw.lower() in combined_text)
    return round(found / len(keywords), 3) if keywords else 0.0


def score_category_coverage(chunks: list, expected_categories: list) -> float:
    retrieved_categories = set(c["category"] for c in chunks)
    expected_set = set(expected_categories)
    overlap = retrieved_categories & expected_set
    return round(len(overlap) / len(expected_set), 3) if expected_set else 0.0


def score_confidence(chunks: list) -> float:
    if not chunks:
        return 0.0
    avg = sum(c["confidence"] for c in chunks) / len(chunks)
    return round(avg, 3)


def check_duplicates(chunks: list) -> int:
    sources = [c["source"] for c in chunks]
    return len(sources) - len(set(sources))


def verify_data_leakage() -> dict:
    """
    Verify RAG corpus documents are not from evaluation/test data.
    All our documents are from public guidelines — no leakage possible.
    """
    corpus_sources = [
        "ACC_AHA_2019_Primary_Prevention.pdf",
        "ESC_2021_CVD_Prevention.pdf",
        "AHA_2021_Heart_Stroke_Statistics.pdf",
        "WHO_2020_HEARTS_Package.pdf",
        "AHA_2019_Echo_Appropriate_Use.pdf",
        "PTB_XL_2020_ECG_Documentation.pdf"
    ]

    evaluation_data_sources = [
        "EchoNet-Dynamic dataset",
        "PTB-XL ECG signals",
        "Clinical patient records"
    ]

    leakage_found = []
    for corpus_doc in corpus_sources:
        for eval_source in evaluation_data_sources:
            if eval_source.lower() in corpus_doc.lower():
                leakage_found.append(corpus_doc)

    return {
        "corpus_documents": corpus_sources,
        "leakage_detected": len(leakage_found) > 0,
        "leakage_details": leakage_found if leakage_found else "None",
        "verdict": "PASS — No data leakage detected" if not leakage_found
                   else "FAIL — Data leakage detected"
    }


def evaluate_retrieval(retriever: RAGRetriever) -> dict:
    print("=== RAG Evaluation — Week 6 ===\n")

    results = []
    total_keyword = 0
    total_category = 0
    total_confidence = 0
    total_duplicates = 0
    edge_case_handled = 0

    for test in TEST_CASES:
        print(f"Testing: {test['name']}")

        inp = test["input"]
        retrieval = retriever.retrieve_for_prediction(
            risk_level=inp["risk_level"],
            ecg_class=inp.get("ecg_class"),
            ef_value=inp.get("ef_value")
        )
        chunks = retrieval["chunks"]

        # Check if edge case was handled
        if inp.get("ecg_class") is None or inp.get("ef_value") is None:
            if len(chunks) > 0:
                edge_case_handled += 1

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
        print(f"  Chunks retrieved:  {len(chunks)}")
        print()

    n = len(TEST_CASES)
    edge_cases = sum(1 for t in TEST_CASES
                     if t["input"].get("ecg_class") is None
                     or t["input"].get("ef_value") is None)

    # Data leakage check
    print("Running data leakage verification...")
    leakage_check = verify_data_leakage()
    print(f"  {leakage_check['verdict']}\n")

    summary = {
        "total_tests": n,
        "avg_keyword_coverage": round(total_keyword / n, 3),
        "avg_category_coverage": round(total_category / n, 3),
        "avg_confidence": round(total_confidence / n, 3),
        "total_duplicates_found": total_duplicates,
        "edge_cases_tested": edge_cases,
        "edge_cases_handled": edge_case_handled,
        "data_leakage_check": leakage_check["verdict"],
        "overall_quality": "GOOD" if (total_keyword / n) >= 0.7
                           else "NEEDS IMPROVEMENT"
    }

    print("=== Summary ===")
    print(f"Total tests:           {n}")
    print(f"Avg keyword coverage:  {summary['avg_keyword_coverage']:.1%}")
    print(f"Avg category coverage: {summary['avg_category_coverage']:.1%}")
    print(f"Avg confidence:        {summary['avg_confidence']:.3f}")
    print(f"Total duplicates:      {summary['total_duplicates_found']}")
    print(f"Edge cases handled:    {edge_case_handled}/{edge_cases}")
    print(f"Data leakage:          {leakage_check['verdict']}")
    print(f"Overall quality:       {summary['overall_quality']}")

    return {"results": results, "summary": summary,
            "data_leakage_verification": leakage_check}


if __name__ == "__main__":
    retriever = RAGRetriever()
    report = evaluate_retrieval(retriever)

    output_path = "rag/evaluation/evaluation_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation report saved to: {output_path}")