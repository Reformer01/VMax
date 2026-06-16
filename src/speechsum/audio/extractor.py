from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from pydub import AudioSegment

from speechsum.config import settings
from speechsum.exceptions import AudioExtractionError

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def extract_audio(
    video_path: str | Path,
    sample_rate: int | None = None,
    keep_temp: bool = False,
) -> tuple[np.ndarray, Path | None]:
    path = Path(video_path)
    if not path.exists():
        raise AudioExtractionError(f"Video file not found: {path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise AudioExtractionError(
            f"Unsupported video format: {path.suffix}. "
            f"Supported: {', '.join(sorted(VIDEO_EXTENSIONS))}"
        )

    sr = sample_rate or settings.sample_rate
    temp_file: Path | None = None

    try:
        import imageio_ffmpeg as ffmpeg

        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise AudioExtractionError(
            "ffmpeg is required for video processing. Install with: pip install imageio-ffmpeg"
        ) from e

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            temp_file = Path(tf.name)

        import subprocess

        cmd = [
            ffmpeg_exe,
            "-i", str(path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-ac", "1",
            "-y",
            str(temp_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise AudioExtractionError(
                f"ffmpeg failed: {result.stderr.strip()}"
            )

        audio = AudioSegment.from_file(temp_file)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / np.iinfo(np.int16).max

        if keep_temp:
            return samples, temp_file

        temp_file.unlink(missing_ok=True)
        return samples, None

    except AudioExtractionError:
        raise
    except subprocess.TimeoutExpired:
        raise AudioExtractionError("ffmpeg extraction timed out (300s limit)") from None
    except Exception as e:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)
        raise AudioExtractionError(f"Failed to extract audio from {path}: {e}") from e
