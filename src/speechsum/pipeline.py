from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog

from speechsum.audio.extractor import VIDEO_EXTENSIONS, extract_audio
from speechsum.audio.loader import AUDIO_EXTENSIONS, load_audio
from speechsum.audio.recorder import record_audio
from speechsum.config import settings
from speechsum.stt.engine import Transcriber
from speechsum.summarize.engine import Summarizer

logger = structlog.get_logger()

SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


class PipelineResult:
    def __init__(
        self,
        *,
        source: str,
        duration_seconds: float,
        transcription: str,
        words: list[dict] | None = None,
        summary: str | None = None,
        model_info: dict | None = None,
    ) -> None:
        self.source = source
        self.duration_seconds = duration_seconds
        self.transcription = transcription
        self.words = words or []
        self.summary = summary
        self.model_info = model_info or {}

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "duration_seconds": self.duration_seconds,
            "transcription": self.transcription,
            "words": self.words,
            "summary": self.summary,
            "model_info": self.model_info,
        }


class Pipeline:
    def __init__(
        self,
        stt_model_path: str | Path | None = None,
        summary_model: str | None = None,
    ) -> None:
        self.transcriber = Transcriber(model_path=stt_model_path)
        self.summarizer = Summarizer(model_name=summary_model) if summary_model else Summarizer()
        logger.info("pipeline_initialized")

    def run(
        self,
        source: str | Path,
        summarize: bool = True,
    ) -> PipelineResult:
        path = Path(source)
        samples, duration = self._load_audio(path)

        logger.info("transcribing", source=str(path))
        text, words = self.transcriber.transcribe_with_timestamps(samples)
        logger.info("transcription_complete", length=len(text), words=len(words))

        summary: str | None = None
        if summarize:
            logger.info("summarizing", text_length=len(text))
            summary = self.summarizer.summarize(text)
            logger.info("summary_complete", length=len(summary))

        return PipelineResult(
            source=str(path),
            duration_seconds=duration,
            transcription=text,
            words=words,
            summary=summary,
            model_info={
                "stt": settings.vosk_model_name,
                "summarization": settings.summarization_model,
                "sample_rate": settings.sample_rate,
            },
        )

    def run_from_recording(
        self,
        duration: int | None = None,
        summarize: bool = True,
    ) -> PipelineResult:
        logger.info("recording", duration=duration or settings.recording_timeout)
        samples = record_audio(duration=duration)
        duration_sec = len(samples) / settings.sample_rate
        logger.info("recording_complete", duration=duration_sec, samples=len(samples))

        logger.info("transcribing_recording")
        text, words = self.transcriber.transcribe_with_timestamps(samples)
        logger.info("transcription_complete", length=len(text))

        summary: str | None = None
        if summarize:
            logger.info("summarizing_recording")
            summary = self.summarizer.summarize(text)
            logger.info("summary_complete", length=len(summary))

        return PipelineResult(
            source="microphone",
            duration_seconds=duration_sec,
            transcription=text,
            words=words,
            summary=summary,
            model_info={
                "stt": settings.vosk_model_name,
                "summarization": settings.summarization_model,
                "sample_rate": settings.sample_rate,
            },
        )

    def _load_audio(self, path: Path) -> tuple[np.ndarray, float]:
        if path.suffix.lower() in AUDIO_EXTENSIONS:
            samples = load_audio(path)
            duration = len(samples) / settings.sample_rate
            return samples, duration
        elif path.suffix.lower() in VIDEO_EXTENSIONS:
            samples, _ = extract_audio(path)
            duration = len(samples) / settings.sample_rate
            return samples, duration
        else:
            from speechsum.exceptions import AudioLoadError
            raise AudioLoadError(
                f"Unsupported file format: {path.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
