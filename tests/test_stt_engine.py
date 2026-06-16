from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from speechsum.exceptions import ModelNotFoundError, TranscriptionError


def test_transcriber_model_not_found():
    with patch("speechsum.stt.model_manager.ensure_model") as mock_ensure:
        mock_ensure.side_effect = ModelNotFoundError("not found")
        with pytest.raises(ModelNotFoundError):
            from speechsum.stt.engine import Transcriber
            Transcriber()


def test_transcriber_model_path_invalid():
    from speechsum.stt.engine import Transcriber
    with pytest.raises(ModelNotFoundError, match="Failed to load Vosk model"):
        Transcriber(model_path="/nonexistent")


@patch("speechsum.stt.engine.vosk")
def test_transcribe_success(mock_vosk):
    import json

    mock_model = MagicMock()
    mock_vosk.Model.return_value = mock_model

    mock_recognizer = MagicMock()
    mock_vosk.KaldiRecognizer.return_value = mock_recognizer
    mock_recognizer.FinalResult.return_value = json.dumps({
        "text": "hello world this is a test"
    })

    from speechsum.stt.engine import Transcriber

    transcriber = Transcriber(model_path=str(Path.cwd()))
    audio = np.zeros(16000, dtype=np.float32)
    result = transcriber.transcribe(audio)

    assert result == "hello world this is a test"
    mock_vosk.KaldiRecognizer.assert_called_once()
    mock_recognizer.AcceptWaveform.assert_called_once()


@patch("speechsum.stt.engine.vosk")
def test_transcribe_empty_result(mock_vosk):
    import json

    mock_model = MagicMock()
    mock_vosk.Model.return_value = mock_model

    mock_recognizer = MagicMock()
    mock_vosk.KaldiRecognizer.return_value = mock_recognizer
    mock_recognizer.FinalResult.return_value = json.dumps({"text": ""})

    from speechsum.stt.engine import Transcriber

    transcriber = Transcriber(model_path=str(Path.cwd()))
    audio = np.zeros(16000, dtype=np.float32)
    with pytest.raises(TranscriptionError, match="empty result"):
        transcriber.transcribe(audio)


@patch("speechsum.stt.engine.vosk")
def test_transcribe_with_timestamps(mock_vosk):
    import json

    mock_model = MagicMock()
    mock_vosk.Model.return_value = mock_model

    mock_recognizer = MagicMock()
    mock_vosk.KaldiRecognizer.return_value = mock_recognizer
    mock_recognizer.FinalResult.return_value = json.dumps({
        "text": "hello world",
        "result": [
            {"word": "hello", "start": 0.0, "end": 0.5, "conf": 1.0},
            {"word": "world", "start": 0.6, "end": 1.0, "conf": 0.95},
        ],
    })

    from speechsum.stt.engine import Transcriber

    transcriber = Transcriber(model_path=str(Path.cwd()))
    audio = np.zeros(16000, dtype=np.float32)
    text, words = transcriber.transcribe_with_timestamps(audio)

    assert text == "hello world"
    assert len(words) == 2
    assert words[0]["word"] == "hello"
    assert words[0]["start"] == 0.0
