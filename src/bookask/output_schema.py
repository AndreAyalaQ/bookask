"""Turns the agent loop's raw output — a free-text answer plus a trace of
tool calls — into the structured JSON contract this project promises:
{answer, tool_calls, sources, confidence}.

Why a separate call instead of having the loop emit this shape on its own
final turn? Because at the moment the loop decides it's done, that call's
config still has `tools=[AGENT_TOOLS]` attached — asking one call to both
pick from a tool menu *and* commit to a rigid response schema is asking
it to satisfy two different structural protocols at once. Splitting
"gather grounded information" (the loop) from "formalize the conclusion
into a checkable artifact" (this module) keeps each call doing one job,
and lets this one run with tools off and a schema on.
"""

from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel

SYNTHESIS_MODEL = "gemini-3.1-flash-lite"
SYNTHESIS_TEMPERATURE = 0.0


class ToolCallRecord(BaseModel):
    tool: str
    input: str
    summary: str


class SourceRecord(BaseModel):
    chunk_id: str
    excerpt: str
    relevance_note: str


class AgentAnswer(BaseModel):
    answer: str
    tool_calls: list[ToolCallRecord]
    sources: list[SourceRecord]
    confidence: Literal["high", "medium", "low"]


SYNTHESIS_SYSTEM_PROMPT = """\
You are formatting the result of a research agent's work into a strict \
JSON structure. You are given the original question, the agent's own \
draft answer, and a log of every tool call the agent made along with \
each tool's raw results.

Rules:
1. `answer` should restate the agent's conclusion clearly and concisely \
— you may tighten the wording, but do not introduce claims that are not \
supported by the tool results provided below.
2. `tool_calls` must have exactly one entry per call in the log, in the \
same order. `input` should be a short string capturing what was passed \
to the tool (the search query, or the term looked up). `summary` is a \
one-sentence description of what was searched or looked up and, \
briefly, what it returned.
3. `sources` must only include excerpts from search_book results — never \
from define_term. Only use chunk_id values that literally appear in the \
search_book results below; never invent one. Each `excerpt` must be a \
short quote or close paraphrase actually drawn from that chunk's text. \
Each `relevance_note` must explain specifically why that chunk supports \
the answer, not just that it mentions the topic. If a retrieved chunk \
was not actually used in the answer, leave it out of `sources`.
4. `confidence` reflects how directly and completely the sources support \
the answer — not how fluent the answer sounds. If the tool calls did not \
fully cover the question (for example, the agent ran out of attempts), \
reflect that with a lower confidence rather than a confident-sounding \
answer.
"""


def _format_tool_call_log(tool_call_log: list[dict]) -> str:
    """Renders the loop's tool_call_log into plain text for the prompt.

    This is the only place the model ever sees a real chunk_id — the
    loop's `final_text` is prose with no ids in it at all (see the test
    runs from the previous step). So this function is what makes accurate
    citation possible at all: without it, rule 3 above would be asking
    the model to cite ids it was never shown.
    """
    if not tool_call_log:
        return "(no tool calls were made)"

    lines = []
    for i, call in enumerate(tool_call_log, 1):
        lines.append(f"Call {i}: {call['tool']}({call['input']})")
        result = call["result"]
        if "error" in result:
            lines.append(f"  -> error: {result['error']}")
        elif call["tool"] == "search_book":
            if not result.get("results"):
                lines.append("  -> no relevant passages found")
            for r in result["results"]:
                lines.append(f"  -> [chunk_id: {r['chunk_id']}] {r['text']}")
        elif call["tool"] == "define_term":
            lines.append(f"  -> definition: {result.get('definition')}")
    return "\n".join(lines)


def synthesize_answer(
    gemini_client: genai.Client,
    question: str,
    draft_answer: str,
    tool_call_log: list[dict],
    hit_iteration_cap: bool,
) -> AgentAnswer:
    prompt = (
        f"Question: {question}\n\n"
        f"Agent's draft answer:\n{draft_answer}\n\n"
        f"Tool call log:\n{_format_tool_call_log(tool_call_log)}\n"
    )
    if hit_iteration_cap:
        prompt += (
            "\nNote: the agent hit its safety limit on tool calls before "
            "confirming it was done. Treat the draft answer as potentially "
            "incomplete when setting confidence.\n"
        )

    response = gemini_client.models.generate_content(
        model=SYNTHESIS_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYNTHESIS_SYSTEM_PROMPT,
            temperature=SYNTHESIS_TEMPERATURE,
            response_mime_type="application/json",
            response_schema=AgentAnswer,
        ),
    )
    return response.parsed
