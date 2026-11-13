"""Implementations behind the declarations in tool_schemas.py.

These are plain functions — nothing here knows about Gemini's function-
calling protocol. The agent loop (next step) is the only thing responsible
for translating a model-issued FunctionCall into a call to one of these,
and translating the return value back into a FunctionResponse.

search_book wraps retriever.retrieve() directly rather than duplicating its
embedding/query logic — bookagent is a second entry point into the same
book index bookask.py builds, not a separate retrieval stack.
"""

from google import genai
from google.genai import types

from .retriever import retrieve

DEFINE_TERM_MODEL = "gemini-3.1-flash-lite"
DEFINE_TERM_TEMPERATURE = 0.0

DEFINE_TERM_SYSTEM_PROMPT = """\
You are a dictionary. Given a word or short term, respond with a single \
concise, plain-language definition — one to two sentences, no examples, no \
usage notes, no extra commentary. Give the term's ordinary, general-purpose \
meaning. You have no knowledge of any specific book and should not try to \
guess one; if the word is ambiguous, give its most common meaning.
"""


def search_book(gemini_client: genai.Client, collection, query: str, k: int = 4) -> dict:
    """Runs bookask's retrieval for `query` and returns the chunks found.

    Shape is deliberately close to retriever.retrieve()'s own hit dicts —
    chunk_id, text, distance — because this whole dict is what gets
    serialized back to the model as the tool's result. The model needs the
    full chunk text (not a preview) to ground its answer and later write an
    accurate `sources[].excerpt` in the final JSON; trimming it here would
    just make the model guess at content it can't see.
    """
    hits = retrieve(gemini_client, collection, query, k=k)

    if not hits:
        return {"query": query, "results": [], "note": "No relevant passages found in the book for this query."}

    return {
        "query": query,
        "results": [
            {
                "chunk_id": hit["chunk_id"],
                "text": hit["text"],
                "distance": hit["distance"],
            }
            for hit in hits
        ],
    }


def define_term(gemini_client: genai.Client, word: str) -> dict:
    """Plain dictionary-style definition, deliberately with no book context.

    Temperature 0.0 (vs. 0.2 for the book's answer-generation step) because
    a definition has essentially one correct answer — there's no synthesis
    or reasoning-style variety worth preserving here, so determinism is
    strictly better: same word in, same definition out, every time.
    """
    response = gemini_client.models.generate_content(
        model=DEFINE_TERM_MODEL,
        contents=f"Define: {word}",
        config=types.GenerateContentConfig(
            system_instruction=DEFINE_TERM_SYSTEM_PROMPT,
            temperature=DEFINE_TERM_TEMPERATURE,
        ),
    )
    return {"word": word, "definition": response.text.strip()}
