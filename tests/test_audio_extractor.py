from __future__ import annotations

from pathlib import Path

import pytest

from speechsum.audio.extractor import extract_audio
from speechsum.exceptions import AudioExtractionError


def test_extract_nonexistent_video():
    with pytest.raises(AudioExtractionError, match="not found"):
        extract_audio("/nonexistent/file.mp4")


def test_extract_unsupported_format():
    path = Path(__file__)
    with pytest.raises(AudioExtractionError, match="Unsupported video format"):
        extract_audio(path)
