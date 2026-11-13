"""Answer generation: feeds retrieved chunks + the system prompt to Gemini.

Plain-text output for now — the structured {answer, sources, confidence}
JSON schema is a separate step (parsing the model's citations/confidence
back out of prose reliably is exactly why that schema exists).
"""

from google import genai
from google.genai import types

from .prompts import SYSTEM_PROMPT, build_context_block

GENERATION_MODEL = "gemini-3.1-flash-lite"
TEMPERATURE = 0.2


def generate_answer(client: genai.Client, question: str, hits: list[dict]) -> str:
    context_block = build_context_block(hits)
    user_content = f"Question: {question}\n\nExcerpts:\n{context_block}"

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=TEMPERATURE,
        ),
    )
    return response.text
