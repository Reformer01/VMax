"""End-to-end smoke test — exercises real audio loading, output, and CLI."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from speechsum.audio.loader import load_audio, load_audio_info
from speechsum.output.json_export import export_json
from speechsum.output.markdown import export_markdown
from speechsum.pipeline import PipelineResult

TMP = Path("tmp")
TMP.mkdir(exist_ok=True)

SR = 16000
DURATION = 2.0
NUM_SAMPLES = int(SR * DURATION)


def generate_test_wav(path: Path) -> None:
    t = np.linspace(0, DURATION, NUM_SAMPLES, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 0.3 * np.iinfo(np.int16).max).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(samples.tobytes())


def test_audio_loading():
    wav_path = TMP / "test_smoke.wav"
    generate_test_wav(wav_path)
    assert wav_path.exists()

    info = load_audio_info(str(wav_path))
    assert abs(info["duration_seconds"] - DURATION) < 0.1
    assert info["frame_rate"] == SR

    audio = load_audio(str(wav_path))
    assert len(audio) == NUM_SAMPLES
    assert audio.dtype == np.float32
    assert np.max(np.abs(audio)) <= 1.0

    wav_path.unlink()
    print("  audio_loading: OK")


def test_json_output():
    result = PipelineResult(
        source="test.wav",
        duration_seconds=3.0,
        transcription="hello world this is a test",
        words=[{"word": "hello", "start": 0.0, "end": 0.5, "conf": 1.0}],
        summary="test summary",
        model_info={"stt": "test", "summarization": "test"},
    )
    text = export_json(result)
    assert "hello world" in text
    assert "test summary" in text

    out = TMP / "test_out.json"
    export_json(result, output_path=out)
    assert out.exists()
    assert "hello world" in out.read_text()
    out.unlink()
    print("  json_output: OK")


def test_markdown_output():
    result = PipelineResult(
        source="test.wav",
        duration_seconds=3.0,
        transcription="hello world",
        summary="test summary",
    )
    text = export_markdown(result)
    assert "Speech Recognition" in text
    assert "hello world" in text

    out = TMP / "test_out.md"
    export_markdown(result, output_path=out)
    assert out.exists()
    assert "hello world" in out.read_text()
    out.unlink()
    print("  markdown_output: OK")


def test_pipeline_result_dict():
    result = PipelineResult(
        source="test.wav",
        duration_seconds=5.0,
        transcription="hello",
        summary="sum",
    )
    d = result.to_dict()
    assert d["source"] == "test.wav"
    assert d["transcription"] == "hello"
    assert d["summary"] == "sum"
    assert d["duration_seconds"] == 5.0
    print("  pipeline_result_dict: OK")


if __name__ == "__main__":
    print("Smoke tests:")
    test_audio_loading()
    test_json_output()
    test_markdown_output()
    test_pipeline_result_dict()
    print("All smoke tests passed!")
