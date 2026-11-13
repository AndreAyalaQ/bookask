"""Text chunking via a standard recursive splitter (token-based, with overlap).

This wraps langchain-text-splitters' RecursiveCharacterTextSplitter, which
splits on paragraph/sentence/word boundaries where possible before falling
back to hard splits, and sizes chunks by actual token count (tiktoken)
rather than raw character count.
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE_TOKENS = 500
DEFAULT_CHUNK_OVERLAP_TOKENS = 75  # ~15% of 500


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[TextChunk]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    pieces = splitter.split_text(text)

    chunks = []
    search_from = 0
    for i, piece in enumerate(pieces):
        start_char = text.index(piece, search_from)
        end_char = start_char + len(piece)
        chunks.append(TextChunk(index=i, text=piece, start_char=start_char, end_char=end_char))
        # Overlap means the next piece can start before this one ends, so we
        # only advance the search cursor partway in to keep index() correct
        # and cheap without re-scanning from zero each time.
        search_from = start_char + 1

    return chunks
