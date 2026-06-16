from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from speechsum.exceptions import ModelLoadError, SummaryError


@patch("speechsum.summarize.engine.AutoTokenizer")
@patch("speechsum.summarize.engine.AutoModelForSeq2SeqLM")
@patch("speechsum.summarize.engine.torch")
def test_summarize_success(mock_torch, mock_auto_model, mock_auto_tokenizer):
    mock_torch.cuda.is_available.return_value = False
    mock_torch.no_grad.return_value.__enter__ = MagicMock()
    mock_torch.no_grad.return_value.__exit__ = MagicMock()

    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "summary text"
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_model.generate.return_value = [MagicMock()]
    mock_auto_model.from_pretrained.return_value = mock_model

    from speechsum.summarize.engine import Summarizer

    summarizer = Summarizer(model_name="fake-model")
    result = summarizer.summarize("This is a long text that should be summarized.")

    assert result == "summary text"
    mock_model.generate.assert_called_once()


def test_summarize_empty_text():
    from speechsum.summarize.engine import Summarizer

    summarizer = Summarizer(model_name="fake-model")
    with pytest.raises(SummaryError, match="empty"):
        summarizer.summarize("")


@patch("speechsum.summarize.engine.torch")
def test_summarize_cuda_auto_detect(mock_torch):
    mock_torch.cuda.is_available.return_value = True

    from speechsum.summarize.engine import Summarizer

    summarizer = Summarizer(model_name="fake-model")
    assert summarizer.device == "cuda"


@patch("speechsum.summarize.engine.AutoTokenizer")
@patch("speechsum.summarize.engine.AutoModelForSeq2SeqLM")
@patch("speechsum.summarize.engine.torch")
def test_model_load_error(mock_torch, mock_auto_model, _mock_tokenizer):
    mock_torch.cuda.is_available.return_value = False
    mock_auto_model.from_pretrained.side_effect = RuntimeError("Model not found")

    from speechsum.summarize.engine import Summarizer

    summarizer = Summarizer(model_name="nonexistent-model")
    with pytest.raises(ModelLoadError, match="Failed to load"):
        summarizer.summarize("test text")


@patch("speechsum.summarize.engine.torch")
def test_summarize_device_override(mock_torch):
    mock_torch.cuda.is_available.return_value = True

    from speechsum.summarize.engine import Summarizer

    summarizer = Summarizer(model_name="fake-model", device="cpu")
    assert summarizer.device == "cpu"
