from __future__ import annotations

from typing import Any

import structlog
import torch
from transformers import pipeline

from speechsum.config import settings
from speechsum.exceptions import ModelLoadError, SummaryError
from speechsum.summarize.chunker import chunk_text

logger = structlog.get_logger()


class Summarizer:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        max_length: int | None = None,
        min_length: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.summarization_model
        self.device = self._resolve_device(device)
        self.max_length = max_length or settings.summarization_max_length
        self.min_length = min_length or settings.summarization_min_length
        self._pipe: Any = None

    def _resolve_device(self, device: str | None) -> str:
        if device and device != "auto":
            return device
        if torch.cuda.is_available():
            logger.info("gpu_available", device="cuda")
            return "cuda"
        logger.info("gpu_unavailable", device="cpu")
        return "cpu"

    def _get_pipeline(self) -> Any:
        if self._pipe is not None:
            return self._pipe

        try:
            logger.info(
                "loading_summarization_model",
                model=self.model_name,
                device=self.device,
            )
            self._pipe = pipeline(
                "summarization",
                model=self.model_name,
                device=self.device,
                max_length=self.max_length,
                min_length=self.min_length,
                truncation=True,
            )
            return self._pipe
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load summarization model '{self.model_name}': {e}"
            ) from e

    def summarize(self, text: str) -> str:
        if not text or not text.strip():
            raise SummaryError("Cannot summarize empty text")

        pipe = self._get_pipeline()
        chunks = chunk_text(text)

        if len(chunks) == 1:
            return self._summarize_single(pipe, chunks[0])

        logger.info("summarizing_chunks", count=len(chunks))
        summaries: list[str] = []
        for i, chunk in enumerate(chunks):
            logger.debug("summarizing_chunk", index=i, size=len(chunk))
            summary = self._summarize_single(pipe, chunk)
            summaries.append(summary)

        if len(summaries) > 1:
            combined = " ".join(summaries)
            if len(combined) > settings.summarization_chunk_size:
                logger.info("summarizing_combined_summary")
                return self._summarize_single(pipe, combined)

        return " ".join(summaries)

    def _summarize_single(self, pipe: Any, text: str) -> str:
        try:
            result = pipe(text, max_length=self.max_length, min_length=self.min_length)
            return result[0]["summary_text"].strip()
        except Exception as e:
            raise SummaryError(f"Summarization failed: {e}") from e
