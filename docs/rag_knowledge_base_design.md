# RAG Knowledge Base Design Document
**Author:** Akash Hiremath (1RF23CS013)
**Week:** 0 — Project Foundation

---

## 1. Purpose
This document defines the structure, sources, and categorization strategy
for the RAG knowledge base used in CardioAI's recommendation pipeline.
The RAG module retrieves relevant clinical guideline text and passes it
to the Claude API to generate evidence-cited, patient-specific recommendations.

---

## 2. How RAG Fits Into CardioAI
/predict request
↓
Fusion module → risk%, ECG class, EF value
↓
RAG Retriever (rag/retriever.py)

→ build_query() → query string

→ retrieve() → top-4 chunks from ChromaDB

↓

RAG Generator (rag/generator.py)

→ Claude API call with chunks + fusion output

→ {summary, details, recommendations[]} with citations

↓

/predict response includes rag_explanation field

---

## 3. Documents Collected

| File | Category | Why Chosen |
|---|---|---|
| ACC_AHA_2019_Primary_Prevention.pdf | risk_factors | Gold standard US prevention guidelines |
| ESC_2021_CVD_Prevention.pdf | risk_factors | Latest European prevention guidelines |
| AHA_2021_Heart_Stroke_Statistics.pdf | risk_factors | Current CVD statistics and risk data |
| WHO_2020_HEARTS_Package.pdf | lifestyle_recommendations | WHO lifestyle intervention framework |
| AHA_2019_Echo_Appropriate_Use.pdf | echo_findings | EF interpretation and echo use criteria |
| PTB_XL_2020_ECG_Documentation.pdf | ecg_findings | PTB-XL ECG class definitions (NORM/MI/STTC/CD/HYP) |

---

## 4. Folder Structure
rag/corpus/

├── risk_factors/             # Hypertension, cholesterol, diabetes guidelines

├── lifestyle_recommendations/# Diet, exercise, smoking cessation

├── echo_findings/            # EF ranges, wall motion, cardiac structure

├── ecg_findings/             # ECG class interpretation

└── symptoms/                 # (reserved for future documents)

---

## 5. Chunking Strategy
- **Splitter:** RecursiveCharacterTextSplitter
- **Chunk size:** 512 tokens
- **Overlap:** 50 tokens
- **Reason:** 512 tokens preserves paragraph-level clinical context.
  50-token overlap prevents cutting sentences at boundaries.

---

## 6. Embedding Model
- **Model:** sentence-transformers/all-MiniLM-L6-v2
- **Why:** Lightweight (80MB), runs on CPU, strong semantic similarity
  for medical text, no GPU required during retrieval.
- **Vector store:** ChromaDB (stored at db/chroma/, gitignored)
- **Collection name:** cardiology_guidelines

---

## 7. Query Builder Design
build_query() maps fusion output to search queries:

| Fusion Output | Query Generated |
|---|---|
| High risk + MI class | "myocardial infarction high cardiovascular risk treatment guidelines" |
| Medium risk + STTC | "ST segment T wave changes cardiovascular risk management" |
| Low risk + NORM | "primary prevention cardiovascular disease lifestyle recommendations" |
| Any + EF < 40% | "heart failure reduced ejection fraction management guidelines" |
| High risk + HYP | "cardiac hypertrophy hypertension treatment recommendations" |

---

## 8. Generator Output Format
Claude API must return exactly this JSON (no markdown, no preamble):

{
  "summary": "One sentence clinical summary.",
  "details": "2-3 sentence elaboration with guideline references.",
  "recommendations": [
    {
      "text": "Recommendation text.",
      "source": "ACC/AHA 2019",
      "page": 14,
      "category": "risk_factors"
    }
  ]
}

Minimum 3, maximum 5 recommendations per prediction.

---

## 9. Week 0 Checklist
- [x] Corpus folder structure created
- [x] 6 PDFs downloaded and categorized
- [x] Chunking strategy defined
- [x] Embedding model selected
- [x] Query builder logic designed
- [x] Generator output format specified