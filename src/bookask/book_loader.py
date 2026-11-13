"""Load a Project Gutenberg .txt file and strip its license header/footer."""

import logging
import re

logger = logging.getLogger(__name__)

# Project Gutenberg has used a few boilerplate formats over the years, but
# modern releases all wrap the actual book text between lines like:
#   *** START OF THE PROJECT GUTENBERG EBOOK <title> ***
#   *** END OF THE PROJECT GUTENBERG EBOOK <title> ***
# ("THE" or "THIS", case varies) — match loosely and take the text between.
START_MARKER_RE = re.compile(
    r"^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*\*\*\*\s*$",
    re.IGNORECASE,
)
END_MARKER_RE = re.compile(
    r"^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*\*\*\*\s*$",
    re.IGNORECASE,
)


def load_book_text(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        return f.read()


def strip_gutenberg_boilerplate(raw_text: str) -> str:
    """Returns the text between the START/END markers. If either marker isn't
    found, logs a warning and returns the text unchanged — better to index a
    few extra boilerplate lines than to silently drop the whole book."""
    lines = raw_text.splitlines()

    start_index = None
    end_index = None
    for i, line in enumerate(lines):
        if start_index is None and START_MARKER_RE.match(line.strip()):
            start_index = i + 1
        elif start_index is not None and END_MARKER_RE.match(line.strip()):
            end_index = i
            break

    if start_index is None or end_index is None:
        logger.warning(
            "Could not find Gutenberg START/END markers — indexing the file as-is."
        )
        return raw_text

    return "\n".join(lines[start_index:end_index]).strip()


def load_clean_book(path: str) -> str:
    return strip_gutenberg_boilerplate(load_book_text(path))
