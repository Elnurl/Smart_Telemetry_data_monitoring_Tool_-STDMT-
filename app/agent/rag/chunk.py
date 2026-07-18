"""Text chunking for knowledge documents."""

from __future__ import annotations

import re
from typing import Iterator


def chunk_text(
    text: str,
    *,
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[str]:
    """Split text into overlapping character chunks on paragraph/sentence boundaries."""
    cleaned = (text or "").replace("\r\n", "\n").strip()
    if not cleaned:
        return []
    # Normalize whitespace a bit but keep newlines for structure
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    n = len(cleaned)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Prefer break at paragraph, then sentence, then space
            window = cleaned[start:end]
            br = max(window.rfind("\n\n"), window.rfind(". "), window.rfind(" "))
            if br > chunk_size // 3:
                end = start + br + (1 if window[br] == "." else 0)
                if cleaned[start:end].endswith("."):
                    end = min(end + 1, n)
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(0, end - overlap)
        if start >= end:
            start = end
    return chunks


def iter_chunks(text: str, **kwargs) -> Iterator[str]:
    yield from chunk_text(text, **kwargs)
