"""Eval harness for bookask's two agent implementations.

Runs the fixed question set in cases.py against the raw agent
(agent_loop.py) and/or the LangGraph agent (bookagent_langchain_langgraph.py),
and scores each response with the deterministic checks in checks.py:
tool-use correctness, citation grounding, and confidence/sources
consistency. No LLM judge — these are all checks a program can verify
without any subjective call, so the result is repeatable and free to run
on every change.

Requires the book to already be indexed (see README: `bookask --index`).

Usage:
    python -m evals.run_evals --book data/yellow_wallpaper.txt
    python -m evals.run_evals --book data/yellow_wallpaper.txt --impl raw
    python -m evals.run_evals --book data/yellow_wallpaper.txt --report evals/report.json
"""

import json
import logging
import time

import click

from bookask.agent_loop import run_agent_loop
from bookask.bookagent_langchain_langgraph import run_agent_graph
from bookask.bookagent_langchain_langgraph import synthesize_answer as synthesize_answer_langgraph
from bookask.chroma_store import get_client, get_or_create_collection
from bookask.config import load_settings, setup_logging
from bookask.gemini_client import get_gemini_client
from bookask.output_schema import AgentAnswer, synthesize_answer as synthesize_answer_raw

from .cases import CASES, EvalCase
from .checks import check_citation_grounding, check_confidence_consistency, check_tool_usage

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_DIR = ".bookask_chroma"
DEFAULT_COLLECTION = "book"

IMPLS = ("raw", "langgraph")


def _run_raw(gemini_client, collection, question: str) -> tuple[list[dict], AgentAnswer, dict]:
    loop_result = run_agent_loop(gemini_client, collection, question)
    answer = synthesize_answer_raw(
        gemini_client,
        question,
        loop_result["final_text"],
        loop_result["tool_call_log"],
        loop_result["hit_iteration_cap"],
    )
    return loop_result["tool_call_log"], answer, loop_result


def _run_langgraph(gemini_client, api_key, collection, question: str) -> tuple[list[dict], AgentAnswer, dict]:
    loop_result = run_agent_graph(gemini_client, api_key, collection, question)
    answer = synthesize_answer_langgraph(
        api_key,
        question,
        loop_result["final_text"],
        loop_result["tool_call_log"],
        loop_result["hit_iteration_cap"],
    )
    return loop_result["tool_call_log"], answer, loop_result


def run_case(impl: str, case: EvalCase, gemini_client, api_key: str, collection) -> dict:
    started = time.monotonic()
    try:
        if impl == "raw":
            tool_call_log, answer, loop_result = _run_raw(gemini_client, collection, case.question)
        else:
            tool_call_log, answer, loop_result = _run_langgraph(gemini_client, api_key, collection, case.question)
    except Exception as exc:
        logger.exception("Case %s (%s) raised during the agent run", case.id, impl)
        return {
            "impl": impl,
            "case_id": case.id,
            "question": case.question,
            "error": str(exc),
            "passed": False,
        }
    latency_s = time.monotonic() - started

    tools_used = [call["tool"] for call in tool_call_log]
    checks = {
        "tool_usage": check_tool_usage(case, tools_used),
        "citation_grounding": check_citation_grounding(tool_call_log, answer),
        "confidence_consistency": check_confidence_consistency(case, answer),
    }
    passed = all(result.passed for result in checks.values())

    return {
        "impl": impl,
        "case_id": case.id,
        "question": case.question,
        "latency_s": round(latency_s, 2),
        "tools_used": tools_used,
        "iterations_used": loop_result["iterations_used"],
        "hit_iteration_cap": loop_result["hit_iteration_cap"],
        "confidence": answer.confidence,
        "num_sources": len(answer.sources),
        "checks": {name: {"pass": r.passed, "detail": r.detail} for name, r in checks.items()},
        "passed": passed,
        "note": case.note,
    }


def print_report(results: list[dict]) -> None:
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        click.echo(f"[{status}] {result['impl']:9s} {result['case_id']}")
        if "error" in result:
            click.echo(f"         error: {result['error']}")
            continue
        click.echo(
            f"         tools={result['tools_used']} confidence={result['confidence']} "
            f"sources={result['num_sources']} latency={result['latency_s']}s"
        )
        for name, check in result["checks"].items():
            if not check["pass"]:
                click.echo(f"         FAILED check '{name}': {check['detail']}")
        if result["note"]:
            click.echo(f"         note: {result['note']}")
        click.echo()

    click.echo("=" * 70)
    for impl in IMPLS:
        impl_results = [r for r in results if r["impl"] == impl]
        if not impl_results:
            continue
        passed = sum(1 for r in impl_results if r["passed"])
        click.echo(f"{impl}: {passed}/{len(impl_results)} cases passed")


@click.command()
@click.option("--book", "book_path", required=True, type=click.Path(), help="Path to the book's .txt file (must already be indexed).")
@click.option("--chroma-dir", default=DEFAULT_CHROMA_DIR, show_default=True, help="Where Chroma persists its data.")
@click.option("--collection", "collection_name", default=DEFAULT_COLLECTION, show_default=True, help="Chroma collection name.")
@click.option("--impl", type=click.Choice(["raw", "langgraph", "both"]), default="both", show_default=True)
@click.option("--report", "report_path", type=click.Path(), default=None, help="Write the full JSON report to this path.")
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(book_path, chroma_dir, collection_name, impl, report_path, verbose):
    setup_logging(verbose)

    try:
        settings = load_settings(book_path, chroma_dir, collection_name)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    try:
        client = get_client(settings.chroma_persist_dir)
        collection = get_or_create_collection(client, settings.collection_name)
    except Exception as exc:
        raise click.ClickException(f"Could not connect to Chroma: {exc}")

    if collection.count() == 0:
        raise click.ClickException(
            f"Collection '{settings.collection_name}' is empty. Run "
            f"`python -m bookask.bookask --book {book_path} --index` first."
        )

    gemini_client = get_gemini_client(settings.gemini_api_key)
    impls = IMPLS if impl == "both" else (impl,)

    results = []
    for case in CASES:
        for impl_name in impls:
            click.echo(f"Running [{impl_name}] {case.id}...")
            results.append(run_case(impl_name, case, gemini_client, settings.gemini_api_key, collection))

    click.echo()
    print_report(results)

    if report_path:
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nFull report written to {report_path}")

    if any(not r["passed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
