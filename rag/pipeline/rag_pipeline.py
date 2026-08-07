"""
RAG Pipeline — Final
Author: Akash
Final Phase — Advanced RAG & Medical Intelligence

Purpose:
    Complete RAG pipeline with dual explanations,
    reference validation and medical disclaimer.
    Single entry point for Sayeed's RAGService.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from rag.retriever.retriever import RAGRetriever
from rag.generator.generator import generate_explanation
from rag.explanation.explainer import generate_dual_explanation
from rag.references.reference_validator import validate_references, format_for_report


class RAGPipeline:
    """
    Final RAG pipeline.
    Sayeed calls run() and gets back the complete RAG output.
    """

    def __init__(self):
        print("Initializing RAG Pipeline...")
        self.retriever = RAGRetriever()
        print("RAG Pipeline ready.\n")

    def run(self, prediction: dict) -> dict:
        """
        Full pipeline: retrieve → validate refs → generate explanations.

        Args:
            prediction: dict with risk_level, risk_percentage,
                        ecg_class, ef_value

        Returns:
            Complete RAG output dict
        """
        try:
            risk_level = prediction.get("risk_level", "Medium")
            ecg_class = prediction.get("ecg_class", None)
            ef_value = prediction.get("ef_value", None)

            # Step 1 — Retrieve
            print("Step 1: Retrieving chunks...")
            retrieval = self.retriever.retrieve_for_prediction(
                risk_level=risk_level,
                ecg_class=ecg_class,
                ef_value=ef_value
            )
            chunks = retrieval["chunks"]
            query = retrieval["query"]

            if not chunks:
                return self._empty_response(query)

            # Step 2 — Validate references
            print("Step 2: Validating references...")
            ref_validation = validate_references(chunks)
            doctor_refs = format_for_report(chunks, "doctor")
            patient_refs = format_for_report(chunks, "patient")

            # Step 3 — Generate dual explanation
            print("Step 3: Generating explanations...")
            dual = generate_dual_explanation(prediction, chunks)

            return {
                "status": "success",
                "query": query,
                "chunks": chunks,
                "doctor_explanation": dual["doctor_explanation"],
                "patient_explanation": dual["patient_explanation"],
                "confidence_analysis": dual["confidence_analysis"],
                "disclaimer": dual["disclaimer"],
                "references": {
                    "validation": ref_validation,
                    "doctor_format": doctor_refs,
                    "patient_format": patient_refs
                }
            }

        except Exception as e:
            print(f"RAG Pipeline error: {e}")
            return self._error_response(str(e))

    def _empty_response(self, query: str) -> dict:
        return {
            "status": "no_results",
            "query": query,
            "chunks": [],
            "doctor_explanation": {},
            "patient_explanation": {},
            "confidence_analysis": {},
            "disclaimer": (
                "No relevant medical guidelines found. "
                "Please consult a healthcare professional."
            ),
            "references": {"validation": {}, "doctor_format": [],
                           "patient_format": []}
        }

    def _error_response(self, error: str) -> dict:
        return {
            "status": "error",
            "error": error,
            "chunks": [],
            "doctor_explanation": {},
            "patient_explanation": {},
            "confidence_analysis": {},
            "disclaimer": (
                "An error occurred generating the medical explanation. "
                "Please consult a healthcare professional."
            ),
            "references": {}
        }


if __name__ == "__main__":
    import json
    print("=== Final RAG Pipeline Test ===\n")

    pipeline = RAGPipeline()

    test_prediction = {
        "risk_level": "High",
        "risk_percentage": 78,
        "ecg_class": "MI",
        "ef_value": 35.0
    }

    print(f"Input: {test_prediction}\n")
    result = pipeline.run(test_prediction)

    print(f"\nStatus: {result['status']}")
    print(f"Chunks retrieved: {len(result['chunks'])}")
    print(f"References validated: "
          f"{result['references']['validation'].get('total_references', 0)}")
    print(f"Validation status: "
          f"{result['references']['validation'].get('validation_status', 'N/A')}")
    print(f"\nDoctor explanation keys: "
          f"{list(result['doctor_explanation'].keys())}")
    print(f"Patient explanation keys: "
          f"{list(result['patient_explanation'].keys())}")
    print(f"\nConfidence analysis: {result['confidence_analysis']}")
    print(f"\nDisclaimer: {result['disclaimer'][:100]}...")