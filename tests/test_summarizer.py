from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from speechsum.exceptions import ModelLoadError, SummaryError


@patch("speechsum.summarize.engine.torch")
@patch("speechsum.summarize.engine.pipeline")
def test_summarize_success(mock_pipeline, mock_torch):
    mock_torch.cuda.is_available.return_value = False

    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"summary_text": "This is a summary."}]
    mock_pipeline.return_value = mock_pipe

    from speechsum.summarize.engine import Summarizer

    summarizer = Summarizer(model_name="fake-model")
    result = summarizer.summarize("This is a long text that should be summarized into something shorter.")

    assert result == "This is a summary."
    mock_pipe.assert_called_once()


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


@patch("speechsum.summarize.engine.torch")
@patch("speechsum.summarize.engine.pipeline")
def test_model_load_error(mock_pipeline, mock_torch):
    mock_torch.cuda.is_available.return_value = False
    mock_pipeline.side_effect = RuntimeError("Model not found")

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
