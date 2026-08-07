"""
RAG Document Cleaner
Author: Akash Hiremath 
Week: 2 — RAG Knowledge Preparation

Purpose:
    Cleans and preprocesses medical PDF documents from the corpus
    before they are chunked and embedded into ChromaDB in Week 3.
"""

import os
import re


# ── Configuration ─────────────────────────────────────────────────────────────

CORPUS_DIR = "rag/corpus"
OUTPUT_DIR = "rag/documents"

CATEGORIES = [
    "risk_factors",
    "lifestyle_recommendations",
    "echo_findings",
    "ecg_findings",
    "symptoms",
]


# ── Text Cleaning Functions ────────────────────────────────────────────────────

def remove_extra_whitespace(text: str) -> str:
    """Remove extra spaces, tabs and blank lines."""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def remove_page_headers_footers(text: str) -> str:
    """Remove common PDF header/footer patterns like page numbers."""
    text = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\s*/\s*\d+\b', '', text)
    return text


def remove_special_characters(text: str) -> str:
    """Remove non-printable and non-ASCII characters."""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\/\%\+\=]', '', text)
    return text


def normalize_text(text: str) -> str:
    """Run all cleaning steps in sequence."""
    text = remove_page_headers_footers(text)
    text = remove_special_characters(text)
    text = remove_extra_whitespace(text)
    return text


# ── Document Processing ────────────────────────────────────────────────────────

def get_all_pdfs() -> list:
    """
    Scan corpus directory and return list of all PDF file paths
    organized by category.
    """
    pdf_files = []
    for category in CATEGORIES:
        category_path = os.path.join(CORPUS_DIR, category)
        if not os.path.exists(category_path):
            continue
        for filename in os.listdir(category_path):
            if filename.endswith(".pdf"):
                pdf_files.append({
                    "filename": filename,
                    "category": category,
                    "path": os.path.join(category_path, filename)
                })
    return pdf_files


def clean_document(text: str, filename: str) -> dict:
    """
    Clean a single document's text and return structured output.

    Args:
        text: Raw extracted text from PDF
        filename: Original PDF filename

    Returns:
        dict with cleaned text and metadata
    """
    cleaned_text = normalize_text(text)

    return {
        "filename": filename,
        "original_length": len(text),
        "cleaned_length": len(cleaned_text),
        "cleaned_text": cleaned_text,
        "status": "cleaned"
    }


def save_cleaned_document(cleaned_doc: dict, category: str) -> str:
    """
    Save cleaned text to rag/documents/<category>/ as a .txt file.

    Args:
        cleaned_doc: Output from clean_document()
        category: Document category folder name

    Returns:
        Output file path
    """
    output_category_dir = os.path.join(OUTPUT_DIR, category)
    os.makedirs(output_category_dir, exist_ok=True)

    output_filename = cleaned_doc["filename"].replace(".pdf", "_cleaned.txt")
    output_path = os.path.join(output_category_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_doc["cleaned_text"])

    return output_path


def process_all_documents() -> list:
    """
    Main pipeline: scan corpus → clean → save to documents/.
    Run this function to prepare all documents for Week 3 embedding.

    Returns:
        List of processing results for all documents
    """
    pdf_files = get_all_pdfs()

    if not pdf_files:
        print("No PDF files found in corpus. Add PDFs to rag/corpus/ first.")
        return []

    results = []
    for doc in pdf_files:
        print(f"Processing: {doc['filename']} ({doc['category']})")

        # NOTE: PDF text extraction will be added in Week 3 using PyMuPDF.
        # For now, we validate the file exists and is readable.
        if not os.path.exists(doc["path"]):
            print(f"  WARNING: File not found — {doc['path']}")
            results.append({
                "filename": doc["filename"],
                "status": "not_found"
            })
            continue

        file_size = os.path.getsize(doc["path"])
        print(f"  OK: {file_size / 1024:.1f} KB — ready for extraction")
        results.append({
            "filename": doc["filename"],
            "category": doc["category"],
            "file_size_kb": round(file_size / 1024, 1),
            "status": "ready"
        })

    print(f"\nSummary: {len(results)} documents found and validated.")
    return results


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== RAG Document Cleaner ===\n")
    results = process_all_documents()

    print("\nDocument Status:")
    for r in results:
        print(f"  [{r['status'].upper()}] {r['filename']}")