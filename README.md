# speechsum

Speech recognition and summarization system. Transcribes audio/video files and microphone input using Vosk (offline STT), then summarizes the transcription with Hugging Face transformer models.

## Features

- **Transcribe** audio files (WAV, MP3, FLAC, OGG, M4A)
- **Extract** audio from video files (MP4, MOV, AVI, MKV, WebM)
- **Record** from microphone and transcribe in real-time
- **Summarize** transcriptions using BART/T5 models (chunk-aware for long audio)
- **Multiple output formats** — console (Rich), JSON, Markdown report
- **Web UI** — dark glassmorphic interface with drag-and-drop upload
- **Offline STT** — Vosk runs locally, no API keys needed
- **GPU auto-detect** — falls back to CPU if CUDA unavailable

## Quick Start

```bash
pip install -e ".[web,dev]"

# Download a Vosk model first
speechsum download

# Transcribe and summarize
speechsum summarize lecture.mp3

# Record from microphone
speechsum record --duration 120

# Launch web UI
speechsum serve
# Then open http://127.0.0.1:8080
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `speechsum transcribe <file>` | Transcribe audio/video file |
| `speechsum summarize <file>` | Transcribe + summarize |
| `speechsum record [--duration 60]` | Record mic → transcribe → summarize |
| `speechsum models` | List installed Vosk models |
| `speechsum download [--url URL]` | Download a Vosk model |
| `speechsum serve [--port 8080]` | Launch web UI |
| `speechsum config show` | Show current configuration |

### Output Options

All transcribe/summarize commands support `--output` (console/json/markdown) and `--output-file`:

```bash
speechsum summarize podcast.mp3 --output json --output-file result.json
speechsum summarize podcast.mp3 --output markdown --output-file report.md
speechsum summarize podcast.mp3 -o markdown -o report.md
```

### Configuration

Set via environment variables (prefix `SPEECHSUM_`) or `.env` file:

```bash
SPEECHSUM_SUMMARIZATION_MODEL=facebook/bart-large-cnn
SPEECHSUM_VOSK_MODEL_PATH=/path/to/vosk-model
SPEECHSUM_SAMPLE_RATE=16000
SPEECHSUM_DEFAULT_OUTPUT_FORMAT=markdown
```

## Project Structure

```
src/speechsum/
├── cli.py              # Click CLI entry point
├── config.py           # Pydantic settings
├── exceptions.py       # Typed exception hierarchy
├── pipeline.py         # Orchestrator
├── audio/
│   ├── loader.py       # Load audio files
│   ├── extractor.py    # Extract audio from video
│   └── recorder.py     # Microphone capture
├── stt/
│   ├── engine.py       # Vosk transcription
│   └── model_manager.py# Model download/cache
├── summarize/
│   ├── engine.py       # HuggingFace summarization
│   └── chunker.py      # Text chunking
└── output/
    ├── console.py      # Rich terminal output
    ├── json_export.py  # JSON serialization
    ├── markdown.py     # Markdown reports
    └── web/            # FastAPI web app
        ├── app.py
        ├── templates/
        └── static/
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=src/speechsum
ruff check src/speechsum/ tests/
mypy src/speechsum/
```

## Dependencies

- **vosk** — Offline speech-to-text
- **transformers + torch** — Summarization models
- **pydub + moviepy** — Audio/video processing
- **sounddevice** — Microphone capture
- **click + rich** — CLI and terminal output
- **fastapi + uvicorn** — Web UI
- **pydantic + structlog** — Config and logging

## License

MIT
