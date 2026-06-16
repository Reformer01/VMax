from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import vosk

from speechsum.config import settings
from speechsum.exceptions import ModelNotFoundError, TranscriptionError
from speechsum.stt.model_manager import ensure_model


class Transcriber:
    def __init__(self, model_path: str | Path | None = None) -> None:
        self.sample_rate = settings.sample_rate
        resolved_path = Path(model_path) if model_path else ensure_model()
        self.model = self._load_model(resolved_path)

    def _load_model(self, model_path: Path) -> vosk.Model:
        if not model_path.exists():
            raise ModelNotFoundError(f"Vosk model not found at: {model_path}")
        try:
            return vosk.Model(str(model_path))
        except Exception as e:
            raise ModelNotFoundError(
                f"Failed to load Vosk model from {model_path}: {e}"
            ) from e

    def transcribe(self, audio: np.ndarray) -> str:
        recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
        recognizer.SetWords(True)

        try:
            audio_bytes = (audio * np.iinfo(np.int16).max).astype(np.int16).tobytes()
            recognizer.AcceptWaveform(audio_bytes)
            result = json.loads(recognizer.FinalResult())
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e

        text = result.get("text", "").strip()
        if not text:
            raise TranscriptionError("Transcription produced empty result")
        return text

    def transcribe_with_timestamps(
        self, audio: np.ndarray
    ) -> tuple[str, list[dict]]:
        recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
        recognizer.SetWords(True)

        try:
            audio_bytes = (audio * np.iinfo(np.int16).max).astype(np.int16).tobytes()
            recognizer.AcceptWaveform(audio_bytes)
            result = json.loads(recognizer.FinalResult())
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e

        text = result.get("text", "").strip()
        words = result.get("result", [])

        if not text:
            raise TranscriptionError("Transcription produced empty result")
        return text, words
