from __future__ import annotations

from pathlib import Path

import numpy as np
from pydub import AudioSegment

from speechsum.audio.utils import validate_audio_path
from speechsum.config import settings
from speechsum.exceptions import AudioLoadError

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff"}


def load_audio(path: str | Path, sample_rate: int | None = None) -> np.ndarray:
    path = Path(path)
    validate_audio_path(path, AUDIO_EXTENSIONS)

    sr = sample_rate or settings.sample_rate

    try:
        audio = AudioSegment.from_file(path)
        audio = audio.set_frame_rate(sr).set_channels(1)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / (np.iinfo(np.int16).max if audio.sample_width == 2 else 1.0)
        return samples
    except Exception as e:
        raise AudioLoadError(f"Failed to load audio from {path}: {e}") from e


def load_audio_info(path: str | Path) -> dict:
    path = Path(path)
    validate_audio_path(path, AUDIO_EXTENSIONS)

    try:
        audio = AudioSegment.from_file(path)
        return {
            "path": str(path),
            "duration_seconds": len(audio) / 1000.0,
            "channels": audio.channels,
            "sample_width": audio.sample_width,
            "frame_rate": audio.frame_rate,
        }
    except Exception as e:
        raise AudioLoadError(f"Failed to read audio info from {path}: {e}") from e
