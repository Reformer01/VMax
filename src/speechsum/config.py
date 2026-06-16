from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SPEECHSUM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Audio
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    channels: int = Field(default=1, ge=1, le=2)
    recording_timeout: int = Field(default=60, ge=1, le=3600)
    recording_device: int | None = Field(default=None)

    # Vosk STT
    vosk_model_path: str | None = Field(default=None)
    vosk_model_url: str = Field(
        default="https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    )
    vosk_model_name: str = Field(default="vosk-model-small-en-us-0.15")

    # Summarization
    summarization_model: str = Field(default="facebook/bart-large-cnn")
    summarization_device: Literal["auto", "cpu", "cuda"] = Field(default="auto")
    summarization_max_length: int = Field(default=150, ge=50, le=500)
    summarization_min_length: int = Field(default=40, ge=10, le=200)
    summarization_chunk_size: int = Field(default=1024, ge=256, le=2048)

    # Output
    default_output_format: Literal["console", "json", "markdown"] = Field(
        default="console"
    )

    # Paths
    model_dir: Path = Field(default=DEFAULT_MODEL_DIR)
    temp_dir: Path = Field(default=DEFAULT_MODEL_DIR.parent / "tmp")

    @model_validator(mode="after")
    def resolve_model_path(self) -> Settings:
        if self.vosk_model_path is None:
            candidate = self.model_dir / self.vosk_model_name
            if candidate.exists():
                self.vosk_model_path = str(candidate)
        return self

    def model_dirs(self) -> list[Path]:
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)
        return list(self.model_dir.iterdir()) if self.model_dir.is_dir() else []


settings = Settings()
