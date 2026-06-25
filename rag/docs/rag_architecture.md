# RAG Architecture Document
**Author:** Akash Hiremath 
**Week:** 1 — RAG Foundation

---

## 1. Overview
The RAG (Retrieval-Augmented Generation) module retrieves relevant
medical guideline chunks and passes them to the Claude API to generate
evidence-cited, patient-specific recommendations after every prediction.

---

## 2. System Flow
Patient Data (clinical + ecg + echo)

↓

/predict endpoint (Sayeed)

↓

Fusion Module → risk%, ECG class, EF value

↓

RAG Retriever — build_query(risk%, ECG class, EF)

↓

ChromaDB Vector Store — top-4 relevant chunks

↓

RAG Generator — Claude API call

↓

{summary, recommendations[]} with citations

↓

Added to /predict response as rag_explanation

---

## 3. Folder Structure
rag/

├── corpus/

│   ├── risk_factors/            # ACC/AHA, ESC, AHA Statistics PDFs

│   ├── lifestyle_recommendations/ # WHO HEARTS Package

│   ├── echo_findings/           # AHA Echo guidelines

│   ├── ecg_findings/            # PTB-XL documentation

│   └── symptoms/                # reserved

├── documents/                   # organized document store

├── metadata/                    # JSON metadata for each PDF

├── docs/                        # architecture and design documents

├── ingest.py                    # (Week 2) chunk + embed + write to ChromaDB

├── retriever.py                 # (Week 2) query builder + top-k retrieval

└── generator.py                 # (Week 2) Claude API call with context

---

## 4. Document Corpus

| File | Category | Key Topics |
|---|---|---|
| ACC_AHA_2019_Primary_Prevention.pdf | risk_factors | hypertension, cholesterol, diabetes |
| ESC_2021_CVD_Prevention.pdf | risk_factors | cardiovascular risk, prevention |
| AHA_2021_Heart_Stroke_Statistics.pdf | risk_factors | CVD statistics, risk data |
| WHO_2020_HEARTS_Package.pdf | lifestyle_recommendations | diet, exercise, smoking |
| AHA_2019_Echo_Appropriate_Use.pdf | echo_findings | EF ranges, cardiac structure |
| PTB_XL_2020_ECG_Documentation.pdf | ecg_findings | NORM, MI, STTC, CD, HYP |

---

## 5. Chunking Strategy

| Parameter | Value | Reason |
|---|---|---|
| Splitter | RecursiveCharacterTextSplitter | Respects sentence boundaries |
| Chunk size | 512 tokens | Preserves paragraph-level context |
| Overlap | 50 tokens | Prevents cutting mid-sentence |

Each chunk carries metadata:
```json
{
  "source": "ACC_AHA_2019_Primary_Prevention.pdf",
  "year": 2019,
  "category": "risk_factors",
  "page": 12
}
```

---

## 6. Embedding Model

- **Model:** sentence-transformers/all-MiniLM-L6-v2
- **Why:** Lightweight, runs on CPU, strong semantic similarity for medical text
- **Vector Store:** ChromaDB
- **Collection:** cardiology_guidelines
- **Location:** db/chroma/ (gitignored, rebuilt via rag/ingest.py)

---

## 7. Retrieval Strategy

build_query() maps fusion output to a search query:

| Fusion Output | Query |
|---|---|
| High risk + MI | "myocardial infarction high risk treatment guidelines" |
| Medium risk + STTC | "ST segment T wave changes risk management" |
| Low risk + NORM | "primary prevention cardiovascular lifestyle recommendations" |
| Any + EF < 40% | "heart failure reduced ejection fraction guidelines" |
| High risk + HYP | "cardiac hypertrophy hypertension treatment" |

retrieve() returns top-4 chunks from ChromaDB for the query.

---

## 8. Generator Output Format

Claude API returns strictly this JSON:

```json
{
  "summary": "One sentence clinical summary.",
  "details": "2-3 sentence elaboration with guideline references.",
  "recommendations": [
    {
      "text": "Specific recommendation.",
      "source": "ACC/AHA 2019",
      "page": 14,
      "category": "risk_factors"
    }
  ]
}
```

Minimum 3, maximum 5 recommendations per prediction.

---

## 9. Week 1 Checklist

- [x] Document corpus organized (6 PDFs in corpus/)
- [x] Metadata JSON created for all 6 documents
- [x] Folder structure created (documents/, metadata/, docs/)
- [x] Retrieval strategy defined
- [x] Architecture document written
- [ ] ingest.py — Week 2
- [ ] retriever.py — Week 2
- [ ] generator.py — Week 2