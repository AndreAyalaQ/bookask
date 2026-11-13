"""CLI entry point for the plain (non-agentic) RAG pipeline.

Usage:
    python -m bookask.bookask --book <path> --index
    python -m bookask.bookask --book <path> --ask "<question>"

--index loads, chunks, embeds, and stores the book in Chroma. --ask
retrieves relevant chunks and generates an answer directly, with no tool
use or agent loop — for the tool-using agents, see bookagent_raw.py and
bookagent_langchain_langgraph.py instead.
"""

import logging
import os

import click

from .book_loader import load_clean_book
from .chroma_store import get_client, get_or_create_collection
from .chunker import chunk_text
from .config import load_settings, setup_logging
from .embedder import index_chunks
from .gemini_client import get_gemini_client
from .generator import generate_answer
from .retriever import retrieve

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_DIR = ".bookask_chroma"
DEFAULT_COLLECTION = "book"


@click.command()
@click.option("--book", "book_path", required=True, type=click.Path(), help="Path to the book's .txt file.")
@click.option("--index", "do_index", is_flag=True, help="Load, chunk, and (re)build the index.")
@click.option("--ask", "question", default=None, help="Ask a question about the indexed book.")
@click.option("--chroma-dir", default=DEFAULT_CHROMA_DIR, show_default=True, help="Where Chroma persists its data.")
@click.option("--collection", "collection_name", default=DEFAULT_COLLECTION, show_default=True, help="Chroma collection name.")
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(book_path, do_index, question, chroma_dir, collection_name, verbose):
    setup_logging(verbose)

    if do_index and question:
        raise click.UsageError("Pass either --index or --ask, not both.")
    if not do_index and not question:
        raise click.UsageError("Pass either --index or --ask \"<question>\".")

    try:
        settings = load_settings(book_path, chroma_dir, collection_name)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    try:
        client = get_client(settings.chroma_persist_dir)
        collection = get_or_create_collection(client, settings.collection_name)
    except Exception as exc:
        raise click.ClickException(f"Could not connect to Chroma: {exc}")

    if do_index:
        run_index(settings, collection)
    else:
        run_ask(settings, collection, question)


def run_index(settings, collection) -> None:
    text = load_clean_book(settings.book_path)
    click.echo(f"Loaded book: {len(text):,} characters after stripping boilerplate")

    chunks = chunk_text(text)
    click.echo(f"Split into {len(chunks)} chunks")
    if chunks:
        click.echo(f"  first chunk preview: {chunks[0].text[:80]!r}")
        click.echo(f"  last chunk preview:  {chunks[-1].text[:80]!r}")

    book_name = os.path.splitext(os.path.basename(settings.book_path))[0]
    gemini_client = get_gemini_client(settings.gemini_api_key)

    click.echo()
    click.echo(f"Embedding {len(chunks)} chunks and storing them in '{collection.name}'...")
    try:
        index_chunks(gemini_client, collection, book_name, chunks)
    except Exception as exc:
        raise click.ClickException(f"Embedding/storage failed: {exc}")

    click.echo(f"Done. Chroma collection '{collection.name}' now has {collection.count()} chunks stored.")


def run_ask(settings, collection, question: str) -> None:
    gemini_client = get_gemini_client(settings.gemini_api_key)

    click.echo(f"Question: {question}")
    try:
        hits = retrieve(gemini_client, collection, question)
    except Exception as exc:
        raise click.ClickException(f"Retrieval failed: {exc}")

    if not hits:
        click.echo("No relevant chunks found.")
        return

    click.echo(f"Retrieved {len(hits)} relevant chunk(s):")
    for hit in hits:
        meta = hit["metadata"]
        click.echo(f"  [chunk {meta['chunk_index']}] distance={hit['distance']:.4f}")
        click.echo(f"      {hit['text'][:100]!r}")

    click.echo()
    try:
        answer = generate_answer(gemini_client, question, hits)
    except Exception as exc:
        raise click.ClickException(f"Generation failed: {exc}")

    click.echo("=== Answer ===")
    click.echo(answer)


if __name__ == "__main__":
    main()
