"""
Patient Chat Generator
Author: Akash
Module 3 — Patient AI Assistant

Purpose:
    Generates simplified, jargon-free answers for patients.
    Unlike Doctor mode, avoids medical terminology entirely.
    Can incorporate food/lifestyle recommendations when relevant.
"""

import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 500


def build_patient_prompt(user_query: str, chunks: list, food_context: dict = None) -> str:
    context_text = ""
    for i, chunk in enumerate(chunks[:4]):
        context_text += (
            f"\n[Reference {i+1}] {chunk['text'][:300]}\n"
        )

    food_text = ""
    if food_context:
        food_text = f"""
FOOD/LIFESTYLE GUIDANCE FOR THIS PATIENT:
Recommended foods: {', '.join(food_context.get('recommended_foods', []))}
Avoid: {', '.join(food_context.get('avoid_foods', []))}
Exercise: {food_context.get('exercise', '')}
"""

    prompt = f"""You are a friendly health assistant talking directly to a
patient. You are NOT a doctor and must NEVER diagnose or predict disease.

RULES:
- Use simple, everyday words. NO medical jargon (no terms like "myocardial",
  "ejection fraction", "HFrEF" — explain things the way you would to a
  friend with no medical background).
- Keep it warm, clear, and reassuring but honest.
- Recommend seeing a doctor for anything about their personal symptoms or risk.

PATIENT'S QUESTION:
{user_query}
{food_text}
MEDICAL GUIDELINES (for grounding your answer, don't quote directly):
{context_text}

Generate a response in this EXACT JSON format (no markdown, no preamble):
{{
  "answer": "Simple, warm 2-4 sentence answer in plain language.",
  "recommend_doctor": true or false
}}"""

    return prompt


def call_groq_api(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")

    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0.4,
        messages=[
            {
                "role": "system",
                "content": "You are a friendly patient-facing health "
                           "assistant. Always respond with valid JSON only. "
                           "Never use medical jargon. Never diagnose."
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
            "recommend_doctor": True
        }


def generate_patient_answer(user_query: str, chunks: list, food_context: dict = None) -> dict:
    print("Generating patient chatbot answer...")
    prompt = build_patient_prompt(user_query, chunks, food_context)
    raw_response = call_groq_api(prompt)
    result = parse_response(raw_response)
    print("Answer generated successfully.")
    return result


if __name__ == "__main__":
    print("=== Patient Chat Generator Test ===\n")

    sample_query = "Can I eat rice?"
    sample_chunks = [
        {
            "text": "A balanced diet including whole grains, in moderate "
                    "portions, supports cardiovascular health when part "
                    "of an overall healthy eating pattern.",
            "source": "ESC_2021_CVD_Prevention.pdf",
            "category": "lifestyle",
            "page": 82
        }
    ]
    sample_food_context = {
        "recommended_foods": ["Leafy greens", "Oats", "Berries"],
        "avoid_foods": ["Fried foods", "Sugary drinks"],
        "exercise": "Short daily walks"
    }

    result = generate_patient_answer(sample_query, sample_chunks, sample_food_context)
    print("\nGenerated Answer:")
    print(json.dumps(result, indent=2))