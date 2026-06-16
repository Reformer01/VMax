from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

import numpy as np

# Suppress pydub's misleading "Couldn't find ffmpeg" warning before importing it
import warnings
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")

from pydub import AudioSegment  # noqa: E402

from speechsum.audio.utils import validate_audio_path  # noqa: E402
from speechsum.config import settings  # noqa: E402
from speechsum.exceptions import AudioLoadError  # noqa: E402

# Point pydub to the ffmpeg bundled with imageio-ffmpeg
import importlib  # noqa: E402
_ffmpeg_spec = importlib.util.find_spec("imageio_ffmpeg")
if _ffmpeg_spec is not None:
    import imageio_ffmpeg  # noqa: E402
    _FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    AudioSegment.converter = _FFMPEG_PATH
else:
    _FFMPEG_PATH = None

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff"}


def _ffmpeg_to_wav(input_path: Path, sample_rate: int) -> Path:
    """Convert any audio file to WAV using ffmpeg (avoids needing ffprobe)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_path = Path(tmp.name)

    cmd = [
        _FFMPEG_PATH,
        "-y",
        "-i", str(input_path),
        "-ar", str(sample_rate),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        raise AudioLoadError(
            f"ffmpeg conversion failed: {result.stderr.decode(errors='ignore').strip()}"
        )
    return output_path


def load_audio(path: str | Path, sample_rate: int | None = None) -> np.ndarray:
    path = Path(path)
    validate_audio_path(path, AUDIO_EXTENSIONS)

    sr = sample_rate or settings.sample_rate

    # For non-WAV files, convert to WAV first using ffmpeg (avoids ffprobe)
    wav_path = path
    cleanup = False
    if path.suffix.lower() != ".wav":
        if _FFMPEG_PATH is None:
            raise AudioLoadError("ffmpeg not available for format conversion")
        wav_path = _ffmpeg_to_wav(path, sr)
        cleanup = True

    try:
        audio = AudioSegment.from_file(wav_path)
        if audio.frame_rate != sr or audio.channels != 1:
            audio = audio.set_frame_rate(sr).set_channels(1)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / (np.iinfo(np.int16).max if audio.sample_width == 2 else 1.0)
        return samples
    except Exception as e:
        raise AudioLoadError(f"Failed to load audio from {path}: {e}") from e
    finally:
        if cleanup and wav_path.exists():
            wav_path.unlink(missing_ok=True)


def load_audio_info(path: str | Path) -> dict:
    path = Path(path)
    validate_audio_path(path, AUDIO_EXTENSIONS)

    # For info, also convert to WAV first to avoid ffprobe
    if path.suffix.lower() != ".wav":
        if _FFMPEG_PATH is None:
            raise AudioLoadError("ffmpeg not available for format conversion")
        wav_path = _ffmpeg_to_wav(path, settings.sample_rate)
        cleanup = True
    else:
        wav_path = path
        cleanup = False

    try:
        audio = AudioSegment.from_file(wav_path)
        return {
            "path": str(path),
            "duration_seconds": len(audio) / 1000.0,
            "channels": audio.channels,
            "sample_width": audio.sample_width,
            "frame_rate": audio.frame_rate,
        }
    except Exception as e:
        raise AudioLoadError(f"Failed to read audio info from {path}: {e}") from e
    finally:
        if cleanup and wav_path.exists():
            wav_path.unlink(missing_ok=True)
