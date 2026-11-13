"""Embeds text chunks with Gemini and stores them in a Chroma collection.

chunk_id scheme: "<book_name>::<zero-padded chunk index>" — deterministic,
so re-running --index on the same book overwrites the same ids (via
upsert) instead of accumulating duplicates.

Embedding uses task_type="RETRIEVAL_DOCUMENT" here because these are the
documents being indexed, not a search query — Gemini's embedding model
optimizes the vector differently depending on which side of a retrieval
pair the text plays. The query side (task_type="RETRIEVAL_QUERY") belongs
in the retrieval step, not here.
"""

import logging

from google import genai
from google.genai import types

from .chunker import TextChunk

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBED_BATCH_SIZE = 32


def embed_texts(client: genai.Client, texts: list[str], task_type: str) -> list[list[float]]:
    """Embeds a single batch of `texts` (must already be <= EMBED_BATCH_SIZE)."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [embedding.values for embedding in response.embeddings]


def chunk_id_for(book_name: str, chunk: TextChunk) -> str:
    return f"{book_name}::{chunk.index:05d}"


def index_chunks(client: genai.Client, collection, book_name: str, chunks: list[TextChunk]) -> None:
    """Embeds and upserts one batch at a time, so a failure partway through
    (e.g. hitting a rate limit) still leaves already-embedded batches stored
    instead of discarding all progress made so far."""
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start:start + EMBED_BATCH_SIZE]

        ids = [chunk_id_for(book_name, chunk) for chunk in batch]
        texts = [chunk.text for chunk in batch]
        metadatas = [
            {
                "book": book_name,
                "chunk_index": chunk.index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }
            for chunk in batch
        ]

        vectors = embed_texts(client, texts, task_type="RETRIEVAL_DOCUMENT")
        collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
        logger.debug("Embedded and stored chunks %d-%d", start, start + len(batch) - 1)
