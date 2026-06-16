from __future__ import annotations

import pytest

from speechsum.output.json_export import export_json
from speechsum.output.markdown import export_markdown
from speechsum.pipeline import PipelineResult


@pytest.fixture
def sample_result():
    return PipelineResult(
        source="test.wav",
        duration_seconds=10.5,
        transcription="hello world this is a test",
        words=[
            {"word": "hello", "start": 0.0, "end": 0.5, "conf": 1.0},
            {"word": "world", "start": 0.6, "end": 1.0, "conf": 0.95},
        ],
        summary="test summary",
        model_info={
            "stt": "vosk-model-small",
            "summarization": "bart-large-cnn",
        },
    )


def test_export_json(sample_result):
    text = export_json(sample_result)
    assert '"transcription": "hello world this is a test"' in text
    assert '"summary": "test summary"' in text
    assert '"source": "test.wav"' in text


def test_export_json_to_file(sample_result, tmp_path):
    output = tmp_path / "result.json"
    export_json(sample_result, output_path=output)
    assert output.exists()
    content = output.read_text()
    assert "hello world" in content


def test_export_markdown(sample_result):
    text = export_markdown(sample_result)
    assert "# Speech Recognition" in text
    assert "hello world this is a test" in text
    assert "test summary" in text
    assert "| Word | Start | End | Confidence |" in text


def test_export_markdown_to_file(sample_result, tmp_path):
    output = tmp_path / "report.md"
    export_markdown(sample_result, output_path=output)
    assert output.exists()
    content = output.read_text()
    assert "# Speech Recognition" in content
    assert "test summary" in content


def test_export_json_no_summary():
    result = PipelineResult(
        source="test.wav",
        duration_seconds=5.0,
        transcription="hello",
        summary=None,
    )
    text = export_json(result)
    assert '"summary": null' in text


def test_export_markdown_no_timestamps():
    result = PipelineResult(
        source="test.wav",
        duration_seconds=5.0,
        transcription="hello",
        summary="brief summary",
    )
    text = export_markdown(result)
    assert "### Word Timestamps" not in text
