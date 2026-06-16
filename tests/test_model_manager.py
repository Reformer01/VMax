from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from speechsum.exceptions import ModelNotFoundError


def test_installed_models_empty(tmp_path):
    with patch("speechsum.stt.model_manager.settings") as mock_settings:
        mock_settings.model_dir = tmp_path
        from speechsum.stt.model_manager import installed_models

        models = installed_models()
        assert models == []


def test_installed_models_with_model(tmp_path):
    model_dir = tmp_path / "vosk-model-small-en-us-0.15"
    model_dir.mkdir(parents=True)
    (model_dir / "am").mkdir()

    with patch("speechsum.stt.model_manager.settings") as mock_settings:
        mock_settings.model_dir = tmp_path
        from speechsum.stt.model_manager import installed_models

        models = installed_models()
        assert len(models) == 1
        assert models[0]["name"] == "vosk-model-small-en-us-0.15"


@patch("speechsum.stt.model_manager.settings")
def test_ensure_model_not_found(mock_settings):
    mock_settings.model_dir = Path("/nonexistent")
    mock_settings.vosk_model_name = "test-model"
    mock_settings.vosk_model_path = None

    from speechsum.stt.model_manager import ensure_model

    with pytest.raises(ModelNotFoundError, match="not found"):
        ensure_model()


@patch("speechsum.stt.model_manager.settings")
def test_ensure_model_found_in_config(mock_settings):
    mock_settings.model_dir = Path("/nonexistent")
    mock_settings.vosk_model_name = "test-model"
    mock_settings.vosk_model_path = str(Path.cwd())

    from speechsum.stt.model_manager import ensure_model

    result = ensure_model()
    assert result == Path.cwd()


@patch("speechsum.stt.model_manager.settings")
def test_download_model_network_error(mock_settings):
    mock_settings.model_dir = Path("/nonexistent")
    mock_settings.vosk_model_name = "test-model"
    mock_settings.vosk_model_url = "https://example.com/model.zip"

    from speechsum.stt.model_manager import download_model

    with pytest.raises(ModelNotFoundError, match="Failed to download"):
        download_model()
