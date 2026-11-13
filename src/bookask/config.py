"""Config loading and plumbing: .env, CLI-derived paths, defaults in one place."""

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    book_path: str
    gemini_api_key: str
    chroma_persist_dir: str
    collection_name: str


def load_settings(book_path: str, chroma_dir: str, collection_name: str) -> Settings:
    """Loads .env and assembles a Settings object. Raises ValueError if required
    config is missing — callers should turn that into a CLI-friendly error."""
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in your .env file.")

    book_path = os.path.abspath(book_path)
    if not os.path.isfile(book_path):
        raise ValueError(f"Book file does not exist: {book_path}")

    return Settings(
        book_path=book_path,
        gemini_api_key=api_key,
        chroma_persist_dir=os.path.abspath(chroma_dir),
        collection_name=collection_name,
    )


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
