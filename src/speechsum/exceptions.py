class SpeechSumError(Exception):
    """Base exception for all speechsum errors."""

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        self.details = details or {}
        super().__init__(message)


class AudioError(SpeechSumError):
    """Base for audio-related errors."""


class AudioLoadError(AudioError):
    """Raised when an audio file cannot be loaded."""


class AudioExtractionError(AudioError):
    """Raised when audio extraction from video fails."""


class RecordingError(AudioError):
    """Raised when microphone recording fails."""


class STTError(SpeechSumError):
    """Base for speech-to-text errors."""


class ModelNotFoundError(STTError):
    """Raised when a required model is not found."""


class TranscriptionError(STTError):
    """Raised when transcription fails."""


class SummaryError(SpeechSumError):
    """Base for summarization errors."""


class ModelLoadError(SummaryError):
    """Raised when a summarization model cannot be loaded."""


class ChunkingError(SummaryError):
    """Raised when text chunking fails."""


class ConfigError(SpeechSumError):
    """Raised when configuration is invalid."""
