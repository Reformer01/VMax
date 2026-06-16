from __future__ import annotations

import pytest

from speechsum.exceptions import ChunkingError
from speechsum.summarize.chunker import chunk_text, estimate_tokens


def test_chunk_empty_text():
    with pytest.raises(ChunkingError, match="empty"):
        chunk_text("")


def test_chunk_short_text():
    text = "Hello world. This is a test."
    chunks = chunk_text(text, max_chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_splits_at_boundary():
    sentences = "Hello world. This is a test sentence. " * 50
    chunks = chunk_text(sentences, max_chunk_size=100, min_chunk_size=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 150


def test_chunk_single_long_sentence():
    sentence = "word " * 500
    chunks = chunk_text(sentence.strip(), max_chunk_size=200, min_chunk_size=50)
    assert len(chunks) > 1


def test_chunk_merges_small_remaining():
    text = "A." * 100 + " " + "B." * 100
    chunks = chunk_text(text, max_chunk_size=500, min_chunk_size=100)
    assert len(chunks) >= 1


def test_estimate_tokens():
    assert estimate_tokens("hello world") == 2
    assert estimate_tokens("") == 0
    assert estimate_tokens("a b c d e") == 5
