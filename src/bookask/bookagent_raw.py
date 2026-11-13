"""CLI for the raw agent: agent_loop.py (hand-written while-loop over the
google-genai SDK) + output_schema.py (structured JSON synthesis).

Mirrors bookagent_langchain_langgraph.py's CLI shape so the two
implementations can be run and compared the same way:

    python -m bookask.bookagent_raw --book <path> --ask "<question>"
    python -m bookask.bookagent_langchain_langgraph --book <path> --ask "<question>"
"""

import json
import logging

import click

from .agent_loop import run_agent_loop
from .chroma_store import get_client, get_or_create_collection
from .config import load_settings, setup_logging
from .gemini_client import get_gemini_client
from .output_schema import synthesize_answer

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_DIR = ".bookask_chroma"
DEFAULT_COLLECTION = "book"


@click.command()
@click.option("--book", "book_path", required=True, type=click.Path(), help="Path to the book's .txt file.")
@click.option("--ask", "question", required=True, help="Ask a question about the indexed book.")
@click.option("--chroma-dir", default=DEFAULT_CHROMA_DIR, show_default=True, help="Where Chroma persists its data.")
@click.option("--collection", "collection_name", default=DEFAULT_COLLECTION, show_default=True, help="Chroma collection name.")
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(book_path, question, chroma_dir, collection_name, verbose):
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

    gemini_client = get_gemini_client(settings.gemini_api_key)

    click.echo(f"Question: {question}\n")

    loop_result = run_agent_loop(gemini_client, collection, question)
    click.echo(
        f"[loop] iterations_used={loop_result['iterations_used']} "
        f"hit_iteration_cap={loop_result['hit_iteration_cap']} "
        f"tool_calls={len(loop_result['tool_call_log'])}"
    )
    for call in loop_result["tool_call_log"]:
        click.echo(f"  - {call['tool']}({call['input']})")
    click.echo()

    answer = synthesize_answer(
        gemini_client,
        question,
        loop_result["final_text"],
        loop_result["tool_call_log"],
        loop_result["hit_iteration_cap"],
    )
    click.echo("=== Structured answer ===")
    click.echo(json.dumps(answer.model_dump(), indent=2))


if __name__ == "__main__":
    main()
