"""
RAG Generator
Author: Akash
Week: 4 — RAG Implementation (Core)

Purpose:
    Takes retrieved medical chunks and prediction data,
    builds a structured prompt, and generates a medical
    explanation using the Groq API (llama-3.1-70b).
"""

import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1000


# ── Prompt Builder ────────────────────────────────────────────────────────────

def build_prompt(prediction: dict, chunks: list) -> str:
    risk_level = prediction.get("risk_level", "Unknown")
    risk_pct = prediction.get("risk_percentage", 0)
    ecg_class = prediction.get("ecg_class", "NORM")
    ef_value = prediction.get("ef_value", None)

    context_text = ""
    for i, chunk in enumerate(chunks[:4]):
        context_text += (
            f"\n[Reference {i+1}] Source: {chunk['source']} "
            f"(Page {chunk['page']})\n{chunk['text'][:300]}\n"
        )

    ef_text = f"{ef_value}%" if ef_value is not None else "Not provided"

    prompt = f"""You are a clinical decision support AI. Based on the patient's
prediction results and the retrieved medical guidelines below, generate a
structured medical explanation.

PATIENT PREDICTION RESULTS:
- Risk Level: {risk_level}
- Risk Percentage: {risk_pct}%
- ECG Classification: {ecg_class}
- Ejection Fraction: {ef_text}

RETRIEVED MEDICAL GUIDELINES:
{context_text}

Generate a response in this EXACT JSON format (no markdown, no preamble):
{{
  "summary": "One sentence clinical summary of the patient's condition.",
  "details": "2-3 sentences elaborating on the findings with reference to guidelines.",
  "recommendations": [
    {{
      "text": "Specific actionable recommendation.",
      "source": "Guideline source name",
      "page": 1,
      "category": "lifestyle/medication/monitoring"
    }}
  ],
  "lifestyle_suggestions": [
    "Specific lifestyle suggestion 1",
    "Specific lifestyle suggestion 2",
    "Specific lifestyle suggestion 3"
  ]
}}

Provide exactly 3-5 recommendations and exactly 3 lifestyle suggestions.
Base everything strictly on the retrieved guidelines provided above."""

    return prompt


# ── API Caller ────────────────────────────────────────────────────────────────

def call_groq_api(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")

    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": "You are a clinical decision support AI. "
                           "Always respond with valid JSON only."
            },
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# ── Response Parser ───────────────────────────────────────────────────────────

def parse_response(raw_text: str) -> dict:
    clean = re.sub(r"```json|```", "", raw_text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "summary": "Medical explanation generated from guidelines.",
            "details": raw_text[:300],
            "recommendations": [],
            "lifestyle_suggestions": []
        }


# ── Main Generator ────────────────────────────────────────────────────────────

def generate_explanation(prediction: dict, chunks: list) -> dict:
    print("Generating medical explanation...")
    prompt = build_prompt(prediction, chunks)
    raw_response = call_groq_api(prompt)
    result = parse_response(raw_response)
    print("Explanation generated successfully.")
    return result


# ── Entry Point (Test) ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== RAG Generator Test ===\n")

    sample_prediction = {
        "risk_level": "High",
        "risk_percentage": 78,
        "ecg_class": "MI",
        "ef_value": 35.0
    }

    sample_chunks = [
        {
            "text": "Patients with high cardiovascular risk and reduced EF should "
                    "receive evidence-based therapy including ACE inhibitors, "
                    "beta-blockers, and statins per ACC/AHA guidelines.",
            "source": "ACC_AHA_2019_Primary_Prevention.pdf",
            "category": "risk_factors",
            "page": 14
        },
        {
            "text": "For patients with myocardial infarction, early intervention "
                    "and lifestyle modification including smoking cessation, "
                    "dietary changes, and regular exercise is recommended.",
            "source": "ESC_2021_CVD_Prevention.pdf",
            "category": "risk_factors",
            "page": 42
        }
    ]

    result = generate_explanation(sample_prediction, sample_chunks)
    print("\nGenerated Explanation:")
    print(json.dumps(result, indent=2))