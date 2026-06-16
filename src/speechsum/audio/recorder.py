from __future__ import annotations

import numpy as np
import sounddevice as sd

from speechsum.config import settings
from speechsum.exceptions import RecordingError


def list_devices() -> list[dict]:
    try:
        devices = sd.query_devices()
        return [
            {
                "index": i,
                "name": d["name"],
                "channels": d["max_input_channels"],
                "sample_rate": d["default_samplerate"],
            }
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]
    except Exception as e:
        raise RecordingError(f"Failed to query audio devices: {e}") from e


def record_audio(
    duration: int | None = None,
    sample_rate: int | None = None,
    device: int | None = None,
) -> np.ndarray:
    sr = sample_rate or settings.sample_rate
    dur = duration or settings.recording_timeout
    dev = device or settings.recording_device

    try:
        if dev is not None:
            sd.check_input_settings(device=dev, samplerate=sr)
    except Exception as e:
        raise RecordingError(
            f"Device {dev} does not support {sr}Hz input: {e}"
        ) from e

    try:
        recording = sd.rec(
            int(dur * sr),
            samplerate=sr,
            channels=1,
            dtype=np.float32,
            device=dev,
        )
        sd.wait()
    except Exception as e:
        raise RecordingError(f"Recording failed: {e}") from e

    if recording is None or len(recording) == 0:
        raise RecordingError("Recording produced no audio data")

    return recording.flatten()
