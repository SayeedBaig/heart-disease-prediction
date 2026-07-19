"""
Doctor Chat Generator
Author: Akash
Module 2 — Doctor AI Assistant

Purpose:
    Generates evidence-based, clinically-toned answers for doctors.
    Can optionally incorporate a patient's prediction context
    (risk level, ECG class, EF value) when the doctor is asking
    about a specific case.
"""

import os
import json
import re
from groq import Groq
from dotenv import load_dotenv
from chat.history_context_builder import format_history_for_prompt

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 700


def build_doctor_prompt(user_query: str, chunks: list, prediction_context: dict = None,
                         history: list = None) -> str:
    context_text = ""
    for i, chunk in enumerate(chunks[:5]):
        context_text += (
            f"\n[Reference {i+1}] Source: {chunk['source']} "
            f"(Page {chunk['page']})\n{chunk['text'][:350]}\n"
        )

    case_text = ""
    if prediction_context:
        case_text = f"""
PATIENT PREDICTION CONTEXT:
Risk Level: {prediction_context.get('risk_level', 'N/A')}
ECG Classification: {prediction_context.get('ecg_class', 'N/A')}
Ejection Fraction: {prediction_context.get('ef_value', 'N/A')}
"""

    history_text = ""
    if history:
        history_text = "\n" + format_history_for_prompt(history)

    prompt = f"""You are a clinical decision-support assistant for doctors on a
heart health platform. Respond with evidence-based, professional medical
language — the audience is a licensed physician, not a patient.

DOCTOR'S QUESTION:
{user_query}
{case_text}{history_text}
RETRIEVED CLINICAL GUIDELINES:
{context_text}

Generate a response in this EXACT JSON format (no markdown, no preamble):
{{
  "answer": "Clinical, evidence-based answer in 3-6 sentences, using proper medical terminology. If visit history is provided, explicitly compare visits and explain the change.",
  "clinical_guidelines_cited": ["Guideline source 1", "Guideline source 2"],
  "recommended_next_steps": ["step 1", "step 2"]
}}

Base your answer strictly on the retrieved guidelines and prediction/history
context provided. Do not invent clinical facts or statistics."""

    return prompt


def call_groq_api(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")

    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "You are a clinical decision-support AI for "
                           "physicians. Always respond with valid JSON only."
            },
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


def parse_response(raw_text: str) -> dict:
    clean = re.sub(r"```json|```", "", raw_text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "answer": raw_text[:400],
            "clinical_guidelines_cited": [],
            "recommended_next_steps": []
        }


def generate_doctor_answer(user_query: str, chunks: list, prediction_context: dict = None,
                            history: list = None) -> dict:
    print("Generating doctor chatbot answer...")
    prompt = build_doctor_prompt(user_query, chunks, prediction_context, history)
    raw_response = call_groq_api(prompt)
    result = parse_response(raw_response)
    print("Answer generated successfully.")
    return result


if __name__ == "__main__":
    print("=== Doctor Chat Generator Test ===\n")

    sample_query = "Why is this prediction High Risk?"
    sample_prediction_context = {
        "risk_level": "High",
        "ecg_class": "MI",
        "ef_value": 35.0
    }
    sample_chunks = [
        {
            "text": "Left ventricular ejection fraction below 40% indicates "
                    "heart failure with reduced ejection fraction (HFrEF), "
                    "associated with significantly elevated cardiovascular risk.",
            "source": "ESC_2021_CVD_Prevention.pdf",
            "category": "risk_factors",
            "page": 66
        }
    ]

    result = generate_doctor_answer(sample_query, sample_chunks, sample_prediction_context)
    print("\nGenerated Answer:")
    print(json.dumps(result, indent=2))