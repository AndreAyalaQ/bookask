"""The core agent loop.

One call to Gemini rarely settles a question that needs tool use — the
model has to see a tool's *results* before it can decide whether it has
enough to answer, or needs to look up something else. So this isn't one
request/response; it's a conversation the code drives on the model's
behalf: ask, let the model request tools, run them, hand the results back
as the next turn, and repeat until the model responds with a plain-text
answer instead of another tool request.
"""

import logging

from google import genai
from google.genai import types

from .tool_schemas import AGENT_TOOLS
from .tools import define_term, search_book

logger = logging.getLogger(__name__)

AGENT_MODEL = "gemini-3.1-flash-lite"
AGENT_TEMPERATURE = 0.2
MAX_ITERATIONS = 5

AGENT_SYSTEM_PROMPT = """\
You are a research agent that answers questions about a specific book \
using two tools:

- search_book(query): retrieves excerpts from the book relevant to a \
query. Use this for anything about the book's plot, characters, themes, \
or content.
- define_term(word): looks up a plain, general-knowledge definition of a \
word. Use this only when a question needs the ordinary meaning of a \
word, not the book's content.

You have no built-in knowledge of this book's content — only what \
search_book returns to you. Never answer a book-content question from \
memory or assumption; call search_book first.

If a question has multiple parts (for example, comparing two points in \
the story, or covering two different characters), call search_book \
separately for each part with a focused query, rather than one broad \
query for the whole question.

Once you have enough information to answer, stop calling tools and \
respond with your answer in plain text. If, after searching, the book \
does not contain enough information to answer, say so plainly rather \
than guessing.
"""


def build_tool_dispatch(gemini_client: genai.Client, collection) -> dict:
    """Binds the runtime context (the live Gemini client and Chroma
    collection) to each tool, so the loop can call dispatch[name](args)
    using only what the model itself supplies. The model only ever
    produces the args its schema declares (e.g. `query`) — it has no way
    to hand back a client or collection object, so something outside the
    model has to supply that half of each call. This is that binding."""
    return {
        "search_book": lambda args: search_book(gemini_client, collection, **args),
        "define_term": lambda args: define_term(gemini_client, **args),
    }


def run_agent_loop(gemini_client: genai.Client, collection, question: str) -> dict:
    """Runs the ask -> tool calls -> feed results back -> repeat cycle.

    Returns {"final_text", "tool_call_log", "iterations_used",
    "hit_iteration_cap"}. tool_call_log is a flat list of
    {"tool", "input", "result"} dicts in call order — a structured record
    of what the agent actually did, kept separate from the raw Gemini
    `contents` history because the next step (structured JSON output)
    needs exactly this, not the wire-format conversation.
    """
    dispatch = build_tool_dispatch(gemini_client, collection)
    tool_call_log: list[dict] = []

    # `contents` is the full conversation so far, in Gemini's own format.
    # It starts with just the user's question and grows by two entries
    # per loop pass: the model's turn (which may request tools) and our
    # reply turn (the tool results). This *entire* growing list is resent
    # on every call — Gemini's API is stateless per-request, so "remembering"
    # earlier tool results is really just re-sending them each time.
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]

    config = types.GenerateContentConfig(
        system_instruction=AGENT_SYSTEM_PROMPT,
        temperature=AGENT_TEMPERATURE,
        tools=[AGENT_TOOLS],
    )

    for iteration in range(1, MAX_ITERATIONS + 1):
        logger.debug("Agent loop iteration %d/%d", iteration, MAX_ITERATIONS)
        response = gemini_client.models.generate_content(
            model=AGENT_MODEL, contents=contents, config=config,
        )

        model_content = response.candidates[0].content
        contents.append(model_content)

        function_calls = [part.function_call for part in model_content.parts if part.function_call]

        if not function_calls:
            # No tool request in this turn means the model considers itself
            # done — this is the loop's only "success" exit.
            return {
                "final_text": response.text,
                "tool_call_log": tool_call_log,
                "iterations_used": iteration,
                "hit_iteration_cap": False,
            }

        # A single turn can request more than one tool call (e.g. two
        # search_book calls for a two-part question, issued together
        # rather than across separate iterations). Whether the model
        # batches them like this or spreads them across iterations is its
        # own choice made per-question — the loop handles both the same
        # way, by executing whatever it's given and replying with all the
        # results in one turn before looping again.
        response_parts = []
        for call in function_calls:
            args = dict(call.args or {})
            logger.debug("Tool call: %s(%r)", call.name, args)

            tool_fn = dispatch.get(call.name)
            if tool_fn is None:
                result = {"error": f"Unknown tool '{call.name}'"}
            else:
                try:
                    result = tool_fn(args)
                except Exception as exc:
                    # Fed back to the model as the tool's result, not
                    # raised — a transient failure (rate limit, bad arg)
                    # becomes something the model can see and react to
                    # (retry differently, or say it couldn't look
                    # something up) instead of crashing the whole CLI run
                    # over one bad tool call.
                    logger.warning("Tool %s failed: %s", call.name, exc)
                    result = {"error": str(exc)}

            tool_call_log.append({"tool": call.name, "input": args, "result": result})
            response_parts.append(types.Part.from_function_response(name=call.name, response=result))

        contents.append(types.Content(role="user", parts=response_parts))

    # Iteration cap hit while the model was still requesting tools. Rather
    # than returning nothing (discarding every tool result gathered so
    # far) or raising, force one last call with tools removed from the
    # config — the model physically cannot request another tool call, so
    # it has to answer in text using whatever is already in `contents`.
    logger.warning("Agent loop hit the %d-iteration cap; forcing a final answer.", MAX_ITERATIONS)
    forced_config = types.GenerateContentConfig(
        system_instruction=(
            AGENT_SYSTEM_PROMPT
            + "\n\nYou have used all available tool calls. Answer now using "
            "only what you have already gathered above — you cannot call "
            "any more tools. If that isn't enough to fully answer, say what "
            "you were able to find and what's missing."
        ),
        temperature=AGENT_TEMPERATURE,
        # tools intentionally omitted here
    )
    forced_response = gemini_client.models.generate_content(
        model=AGENT_MODEL, contents=contents, config=forced_config,
    )
    return {
        "final_text": forced_response.text,
        "tool_call_log": tool_call_log,
        "iterations_used": MAX_ITERATIONS,
        "hit_iteration_cap": True,
    }
