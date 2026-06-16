from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from speechsum.exceptions import AudioLoadError


@patch("speechsum.pipeline.Summarizer")
@patch("speechsum.pipeline.Transcriber")
def test_pipeline_file_not_found(mock_transcriber_cls, _mock_summarizer_cls):
    mock_transcriber_cls.return_value = MagicMock()
    from speechsum.pipeline import Pipeline

    pipeline = Pipeline()
    with pytest.raises(AudioLoadError, match="not found"):
        pipeline.run("/nonexistent/file.mp3")


@patch("speechsum.pipeline.Summarizer")
@patch("speechsum.pipeline.Transcriber")
def test_pipeline_unsupported_format(mock_transcriber_cls, _mock_summarizer_cls):
    mock_transcriber_cls.return_value = MagicMock()
    from speechsum.pipeline import Pipeline

    pipeline = Pipeline()
    with pytest.raises(AudioLoadError, match="Unsupported file format"):
        pipeline.run(__file__)


@patch("speechsum.pipeline.Transcriber")
@patch("speechsum.pipeline.Summarizer")
@patch("speechsum.pipeline.load_audio")
def test_pipeline_run(mock_load_audio, mock_summarizer_cls, mock_transcriber_cls):
    mock_load_audio.return_value = np.zeros(16000, dtype=np.float32)

    mock_transcriber = MagicMock()
    mock_transcriber.transcribe_with_timestamps.return_value = (
        "hello world",
        [{"word": "hello", "start": 0.0, "end": 0.5, "conf": 1.0}],
    )
    mock_transcriber_cls.return_value = mock_transcriber

    mock_summarizer = MagicMock()
    mock_summarizer.summarize.return_value = "summary text"
    mock_summarizer_cls.return_value = mock_summarizer

    from speechsum.pipeline import Pipeline

    pipeline = Pipeline()
    result = pipeline.run(some_wav := str(Path.cwd() / "test.wav"))

    assert result.transcription == "Hello World!"
    assert result.summary == "summary text"
    assert result.source == some_wav
    assert len(result.words) == 1


@patch("speechsum.pipeline.Transcriber")
@patch("speechsum.pipeline.Summarizer")
@patch("speechsum.pipeline.record_audio")
def test_pipeline_run_from_recording(mock_record, mock_summarizer_cls, mock_transcriber_cls):
    mock_record.return_value = np.zeros(16000, dtype=np.float32)

    mock_transcriber = MagicMock()
    mock_transcriber.transcribe_with_timestamps.return_value = (
        "recorded speech",
        [],
    )
    mock_transcriber_cls.return_value = mock_transcriber

    mock_summarizer = MagicMock()
    mock_summarizer.summarize.return_value = "recorded summary"
    mock_summarizer_cls.return_value = mock_summarizer

    from speechsum.pipeline import Pipeline

    pipeline = Pipeline()
    result = pipeline.run_from_recording(duration=5)

    assert result.transcription == "Recorded speech."
    assert result.summary == "recorded summary"
    assert result.source == "microphone"
