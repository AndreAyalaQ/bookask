"""Function-calling declarations for the agent's tools.

These are descriptions *about* the tools, not the tools themselves — Gemini
never sees search_book()'s or define_term()'s code. It only sees the name,
description, and parameter schema declared here, and decides from that text
alone whether and how to call each tool. tools.py (next step) holds the
actual implementations these declarations point to.
"""

from google.genai import types

SEARCH_BOOK_DECLARATION = types.FunctionDeclaration(
    name="search_book",
    description=(
        "Search the indexed book for passages relevant to a specific query. "
        "Use this for any question about the book's plot, characters, "
        "themes, setting, or specific content — anything the answer to "
        "which depends on what the book actually says. If a question has "
        "multiple distinct parts (e.g. comparing two points in the story, "
        "or asking about two different characters), call this tool once "
        "per part with a focused query for that part, rather than one "
        "broad query covering everything."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(
                type=types.Type.STRING,
                description=(
                    "A focused, natural-language description of what to "
                    "find in the book. Be specific about the aspect and, if "
                    "relevant, the part of the story (e.g. 'the narrator's "
                    "emotional state near the beginning of the story') "
                    "rather than a bare keyword (e.g. 'feelings')."
                ),
            ),
        },
        required=["query"],
    ),
)

DEFINE_TERM_DECLARATION = types.FunctionDeclaration(
    name="define_term",
    description=(
        "Look up a plain, general-knowledge definition of a single word or "
        "term. Use this only when the question needs the ordinary meaning "
        "of a word and that meaning does NOT depend on this book's content "
        "— for example, an unfamiliar vocabulary word used in the question "
        "itself. Do not use this for questions about how the book uses a "
        "word, what a character means by something, or anything that "
        "requires the book's text to answer — use search_book for those."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "word": types.Schema(
                type=types.Type.STRING,
                description="The single word or short term to define.",
            ),
        },
        required=["word"],
    ),
)

AGENT_TOOLS = types.Tool(
    function_declarations=[SEARCH_BOOK_DECLARATION, DEFINE_TERM_DECLARATION]
)
