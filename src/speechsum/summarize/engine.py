from __future__ import annotations

from typing import Any

import structlog
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
        self._model: Any = None
        self._tokenizer: Any = None

    def _resolve_device(self, device: str | None) -> str:
        if device and device != "auto":
            return device
        if torch.cuda.is_available():
            logger.info("gpu_available", device="cuda")
            return "cuda"
        logger.info("gpu_unavailable", device="cpu")
        return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            logger.info(
                "loading_summarization_model",
                model=self.model_name,
                device=self.device,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load summarization model '{self.model_name}': {e}"
            ) from e

    def summarize(self, text: str) -> str:
        if not text or not text.strip():
            raise SummaryError("Cannot summarize empty text")

        self._load()
        chunks = chunk_text(text)

        if len(chunks) == 1:
            return self._summarize_single(chunks[0])

        logger.info("summarizing_chunks", count=len(chunks))
        summaries: list[str] = []
        for i, chunk in enumerate(chunks):
            logger.debug("summarizing_chunk", index=i, size=len(chunk))
            summary = self._summarize_single(chunk)
            summaries.append(summary)

        if len(summaries) > 1:
            combined = " ".join(summaries)
            if len(combined) > settings.summarization_chunk_size:
                logger.info("summarizing_combined_summary")
                return self._summarize_single(combined)

        return " ".join(summaries)

    def _summarize_single(self, text: str) -> str:
        try:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=settings.summarization_chunk_size,
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    inputs["input_ids"],
                    max_length=self.max_length,
                    min_length=self.min_length,
                    num_beams=4,
                    early_stopping=True,
                )

            result = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result.strip()
        except Exception as e:
            raise SummaryError(f"Summarization failed: {e}") from e
