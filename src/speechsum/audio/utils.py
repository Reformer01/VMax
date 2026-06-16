from __future__ import annotations

from pathlib import Path

import numpy as np

from speechsum.exceptions import AudioLoadError


def validate_audio_path(path: Path, supported_extensions: set[str]) -> None:
    if not path.exists():
        raise AudioLoadError(f"File not found: {path}")
    if not path.is_file():
        raise AudioLoadError(f"Path is not a file: {path}")
    if path.suffix.lower() not in supported_extensions:
        raise AudioLoadError(
            f"Unsupported format: {path.suffix}. "
            f"Supported: {', '.join(sorted(supported_extensions))}"
        )


def normalize_audio(samples: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = samples / max_val
    return samples
