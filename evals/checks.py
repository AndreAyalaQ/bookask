"""Deterministic checks run against one agent response for one EvalCase.

Each check takes the case's expectations plus the agent's actual output
(the tool_call_log and the synthesized AgentAnswer) and returns a
CheckResult. No LLM judge, no API calls here — everything is checkable
in plain Python, which is exactly what makes these checks repeatable and
free to run on every change.
"""

import re
from dataclasses import dataclass

from bookask.output_schema import AgentAnswer

from .cases import EvalCase

WORD_RE = re.compile(r"[a-z']+")


@dataclass
class CheckResult:
    passed: bool
    detail: str


def _words(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower()))


def check_tool_usage(case: EvalCase, tools_used: list[str]) -> CheckResult:
    """Did the agent call only tools it was allowed to, and avoid any it
    was explicitly forbidden from calling?

    allowed_tools=frozenset() means "should answer without any tool call"
    (the off-topic control question) — any tool call at all is a fail.
    """
    used = set(tools_used)
    forbidden_hit = used & case.forbidden_tools
    if forbidden_hit:
        return CheckResult(False, f"called forbidden tool(s): {sorted(forbidden_hit)}")

    if not case.allowed_tools:
        if used:
            return CheckResult(False, f"expected no tool calls, but called: {sorted(used)}")
        return CheckResult(True, "no tool calls, as expected")

    disallowed = used - case.allowed_tools
    if disallowed:
        return CheckResult(False, f"called tool(s) outside allowed set: {sorted(disallowed)}")

    return CheckResult(True, f"tools used ({sorted(used)}) are all allowed")


def check_citation_grounding(tool_call_log: list[dict], answer: AgentAnswer) -> CheckResult:
    """Every cited chunk_id must come from an actual search_book result, and
    the excerpt text attributed to it must substantially overlap with that
    chunk's real text — catching both a hallucinated chunk_id and a quote
    that doesn't actually appear in the source it's attributed to.
    """
    chunk_text_by_id: dict[str, str] = {}
    for call in tool_call_log:
        if call["tool"] != "search_book":
            continue
        for hit in call["result"].get("results", []):
            chunk_text_by_id[hit["chunk_id"]] = hit["text"]

    if not answer.sources:
        return CheckResult(True, "no sources cited, nothing to ground")

    problems = []
    for source in answer.sources:
        chunk_text = chunk_text_by_id.get(source.chunk_id)
        if chunk_text is None:
            problems.append(f"chunk_id {source.chunk_id!r} was never returned by search_book")
            continue

        excerpt_words = _words(source.excerpt)
        if not excerpt_words:
            problems.append(f"chunk_id {source.chunk_id!r} has an empty excerpt")
            continue

        overlap = excerpt_words & _words(chunk_text)
        coverage = len(overlap) / len(excerpt_words)
        if coverage < 0.6:
            problems.append(
                f"chunk_id {source.chunk_id!r} excerpt only {coverage:.0%} "
                "word-overlaps with the actual chunk text"
            )

    if problems:
        return CheckResult(False, "; ".join(problems))
    return CheckResult(True, f"all {len(answer.sources)} source(s) grounded in retrieved chunks")


def check_confidence_consistency(case: EvalCase, answer: AgentAnswer) -> CheckResult:
    """A 'high' confidence answer that cites zero sources is a contradiction
    when the book itself was the expected source of truth. Only applies to
    cases where search_book is in play — a vocabulary-only answer (sources
    always empty by the synthesis prompt's own rule) can legitimately be
    high confidence.
    """
    if "search_book" not in case.allowed_tools:
        return CheckResult(True, "case doesn't rely on search_book; confidence/sources check skipped")

    if answer.confidence == "high" and not answer.sources:
        return CheckResult(False, "confidence is 'high' but no sources were cited")
    return CheckResult(True, f"confidence ({answer.confidence}) consistent with {len(answer.sources)} source(s)")
