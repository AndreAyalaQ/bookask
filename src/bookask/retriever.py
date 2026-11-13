"""Retrieval: embeds a question and pulls the most relevant chunks from Chroma.

Distance threshold picked empirically for this collection's default Chroma
space ("l2"): three real test queries against "The Yellow Wallpaper" put
on-topic results at distance ~0.43-0.63 and an off-topic control question
("What is the capital of France?") at ~0.96-0.97. 0.8 sits comfortably
between the two. If you change the embedding model, the distance space
(e.g. cosine instead of l2), or the book, re-run a few real queries and
eyeball the numbers again before trusting this value.
"""

import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

QUERY_MODEL = "models/gemini-embedding-001"
DEFAULT_K = 4
DISTANCE_THRESHOLD = 0.8


def embed_query(client: genai.Client, question: str) -> list[float]:
    response = client.models.embed_content(
        model=QUERY_MODEL,
        contents=[question],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return response.embeddings[0].values


def retrieve(client: genai.Client, collection, question: str, k: int = DEFAULT_K) -> list[dict]:
    """Returns up to k chunks judged relevant enough to answer `question`.

    Each result is {"chunk_id": ..., "text": ..., "metadata": ...,
    "distance": ...}, closest first. Matches beyond DISTANCE_THRESHOLD are
    dropped, so this can return an empty list when nothing in the
    collection is actually relevant.
    """
    query_vector = embed_query(client, question)
    result = collection.query(query_embeddings=[query_vector], n_results=k)

    hits = []
    for chunk_id, doc, metadata, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        if distance > DISTANCE_THRESHOLD:
            logger.debug("Dropping chunk (distance=%.4f > threshold): %r", distance, doc[:60])
            continue
        hits.append({"chunk_id": chunk_id, "text": doc, "metadata": metadata, "distance": distance})

    return hits
