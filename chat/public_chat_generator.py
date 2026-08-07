"""
Public Chat Generator
Author: Akash
Module 1 — Public AI Health Assistant

Purpose:
    Generates conversational, educational answers to free-text
    health questions using retrieved medical chunks. Reuses the
    same Groq client setup as rag/generator/generator.py but with
    a conversational prompt instead of a prediction-based one.
"""

import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 600


def build_public_prompt(user_query: str, chunks: list) -> str:
    context_text = ""
    for i, chunk in enumerate(chunks[:4]):
        context_text += (
            f"\n[Reference {i+1}] Source: {chunk['source']} "
            f"(Page {chunk['page']})\n{chunk['text'][:300]}\n"
        )

    prompt = f"""You are a public health education assistant for a heart health
platform. You are NOT a doctor and must NEVER diagnose, predict disease,
or assess anyone's personal risk. You only educate, guide, and recommend
consulting a doctor when appropriate.

USER QUESTION:
{user_query}

RETRIEVED MEDICAL GUIDELINES:
{context_text}

Generate a response in this EXACT JSON format (no markdown, no preamble):
{{
  "answer": "Clear, simple 2-4 sentence educational answer based strictly on the guidelines above.",
  "recommend_doctor": true or false,
  "sources": ["Guideline source name 1", "Guideline source name 2"]
}}

Set "recommend_doctor" to true if the question relates to symptoms,
personal risk, or anything requiring professional evaluation.
Keep language simple — this is for the general public, not medical staff.
Base your answer strictly on the retrieved guidelines. Do not invent facts."""

    return prompt


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
                "content": "You are a public health education AI. "
                           "Always respond with valid JSON only. "
                           "Never diagnose or predict disease."
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
            "answer": raw_text[:300],
            "recommend_doctor": True,
            "sources": []
        }


def generate_public_answer(user_query: str, chunks: list) -> dict:
    print("Generating public chatbot answer...")
    prompt = build_public_prompt(user_query, chunks)
    raw_response = call_groq_api(prompt)
    result = parse_response(raw_response)
    print("Answer generated successfully.")
    return result


if __name__ == "__main__":
    print("=== Public Chat Generator Test ===\n")

    sample_query = "What causes chest pain?"
    sample_chunks = [
        {
            "text": "Chest pain can result from cardiac causes such as angina "
                    "or myocardial ischemia, as well as non-cardiac causes "
                    "including musculoskeletal strain or acid reflux.",
            "source": "AHA_2021_Heart_Stroke_Statistics.pdf",
            "category": "symptoms",
            "page": 378
        }
    ]

    result = generate_public_answer(sample_query, sample_chunks)
    print("\nGenerated Answer:")
    print(json.dumps(result, indent=2))