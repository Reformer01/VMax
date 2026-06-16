from __future__ import annotations

import re

from speechsum.config import settings
from speechsum.exceptions import ChunkingError

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def chunk_text(
    text: str,
    max_chunk_size: int | None = None,
    min_chunk_size: int | None = None,
) -> list[str]:
    if not text or not text.strip():
        raise ChunkingError("Cannot chunk empty text")

    max_size = max_chunk_size or settings.summarization_chunk_size
    min_size = min_chunk_size or (max_size // 2)

    sentences = SENTENCE_BOUNDARY.split(text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text.strip()]

    if len(text) <= max_size:
        return [text.strip()]

    chunks: list[str] = []
    current_chunk: list[str] = []

    for sentence in sentences:
        candidate = " ".join(current_chunk + [sentence])
        if len(candidate) <= max_size:
            current_chunk.append(sentence)
        else:
            if current_chunk:
                chunk_text_joined = " ".join(current_chunk)
                if len(chunk_text_joined) >= min_size:
                    chunks.append(chunk_text_joined)

            if len(sentence) > max_size:
                sub_chunks = _split_long_sentence(sentence, max_size, min_size)
                chunks.extend(sub_chunks)
                current_chunk = []
            else:
                current_chunk = [sentence]

    if current_chunk:
        remaining = " ".join(current_chunk)
        if chunks and len(remaining) < min_size:
            chunks[-1] = chunks[-1] + " " + remaining
        else:
            chunks.append(remaining)

    return chunks


def _split_long_sentence(
    sentence: str, max_size: int, min_size: int
) -> list[str]:
    words = sentence.split()
    chunks: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) <= max_size:
            current.append(word)
        else:
            if current:
                chunks.append(" ".join(current))
            current = [word]

    if current:
        remaining = " ".join(current)
        if chunks and len(remaining) < min_size:
            chunks[-1] = chunks[-1] + " " + remaining
        else:
            chunks.append(remaining)

    return chunks


def estimate_tokens(text: str) -> int:
    return len(text.split())
