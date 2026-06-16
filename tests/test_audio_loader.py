from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from speechsum.audio.loader import load_audio, load_audio_info
from speechsum.audio.utils import validate_audio_path
from speechsum.exceptions import AudioLoadError


def test_validate_audio_path_nonexistent():
    with pytest.raises(AudioLoadError, match="not found"):
        validate_audio_path(Path("/nonexistent/file.wav"), {".wav"})


def test_validate_audio_path_wrong_extension():
    path = Path(__file__)
    with pytest.raises(AudioLoadError, match="Unsupported format"):
        validate_audio_path(path, {".wav"})


def test_load_nonexistent_file():
    with pytest.raises(AudioLoadError, match="not found"):
        load_audio("/nonexistent/file.wav")


def test_load_unsupported_format():
    path = Path(__file__)
    with pytest.raises(AudioLoadError, match="Unsupported format"):
        load_audio(path)


def test_load_audio_info_nonexistent():
    with pytest.raises(AudioLoadError, match="not found"):
        load_audio_info("/nonexistent/file.wav")


def test_generated_wav_file(tmp_path):
    import wave

    filepath = tmp_path / "test.wav"
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)
    frequency = 440.0

    samples = (np.sin(2 * np.pi * frequency * np.arange(num_samples) / sample_rate) * 0.5).astype(np.int16)

    with wave.open(str(filepath), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())

    loaded = load_audio(str(filepath))
    assert isinstance(loaded, np.ndarray)
    assert loaded.dtype == np.float32
    assert len(loaded) > 0
    assert np.max(np.abs(loaded)) <= 1.0

    info = load_audio_info(str(filepath))
    assert info["channels"] == 1
    assert abs(info["duration_seconds"] - 1.0) < 0.1
    assert info["frame_rate"] == sample_rate
