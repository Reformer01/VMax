# Setup Guide

Step-by-step instructions to install and run speechsum on Windows.

---

## Prerequisites

| Dependency | Version | Why |
|------------|---------|-----|
| Python | 3.10 – 3.12 | Project tested on 3.12 |
| pip | latest | Package installer |
| ffmpeg | any recent | Audio/video processing (pydub + moviepy) |
| Git | any | Version control (optional) |

---

## 1. Install Python

1. Download from [python.org](https://www.python.org/downloads/)
2. **Check "Add Python to PATH"** during installation
3. Verify:

```powershell
python --version
pip --version
```

---

## 2. Install ffmpeg

speechsum uses `pydub` and `moviepy` which need ffmpeg. The easiest way:

```powershell
pip install imageio-ffmpeg
```

This installs a bundled ffmpeg binary. No manual PATH setup needed.

To verify:

```powershell
python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
```

---

## 3. Clone or Copy the Project

```powershell
# If you have the repo URL:
git clone https://github.com/Reformer01/VMax.git
cd VMax

# Or if you already have the folder, cd into it:
cd C:\Users\NGFEP\VMax
```

---

## 4. Install speechsum

### A. Install with all features (recommended)

```powershell
pip install -e ".[web,dev]"
```

This installs:
- **Core**: vosk, transformers, torch, pydub, moviepy, sounddevice
- **CLI**: click, rich
- **Web**: fastapi, uvicorn, jinja2
- **Dev**: pytest, mypy, ruff, pre-commit

### B. Base install only (no web UI)

```powershell
pip install -e ".[dev]"
```

---

## 5. Download a Vosk Speech Model

speechsum needs a Vosk model for transcription. The small English model (~40 MB) is good for testing:

```powershell
python -m speechsum download
```

This downloads `vosk-model-small-en-us-0.15` to the `models/` folder.

**For better accuracy**, download the large model manually:
1. Go to https://alphacephei.com/vosk/models
2. Download `vosk-model-en-us-0.22.zip` (~1.8 GB)
3. Extract to `models/vosk-model-en-us-0.22`
4. Configure it:

```powershell
# Use --model flag each time, or set env var:
$env:SPEECHSUM_VOSK_MODEL_PATH = "$pwd\models\vosk-model-en-us-0.22"
```

---

## 6. Verify Installation

```powershell
# Check CLI
python -m speechsum --help

# Check config
python -m speechsum config show

# Check installed models
python -m speechsum models
```

You should see:

```
sample_rate: 16000
vosk_model_name: vosk-model-small-en-us-0.15
summarization_model: facebook/bart-large-cnn
```

---

## 7. Quick Test

Run the end-to-end test with a generated speech sample:

```powershell
# Generate test audio
pip install gTTS
python -c "
from gtts import gTTS
gTTS('The quick brown fox jumps over the lazy dog. Python is a powerful programming language used for machine learning and data science.', lang='en').save('tmp\sample.mp3')
import subprocess, imageio_ffmpeg as ffmpeg
subprocess.run([ffmpeg.get_ffmpeg_exe(), '-i', 'tmp\sample.mp3', '-ar', '16000', '-ac', '1', '-y', 'tmp\sample.wav'])
"

# Transcribe it
python -m speechsum transcribe tmp\sample.wav
```

---

## 8. First Summarization

The first time you summarize, Hugging Face downloads the BART model (~1.5 GB):

```powershell
python -m speechsum transcribe tmp\sample.wav --output json
```

This downloads `facebook/bart-large-cnn` to `%USERPROFILE%\.cache\huggingface\hub\`.

**Note**: Subsequent runs use the cached model and are much faster.

---

## 9. Run the Test Suite

```powershell
pytest tests/ -v
```

Expected output: **52 passed**.

---

## 10. Usage Examples

### Transcribe an audio file

```powershell
python -m speechsum transcribe lecture.mp3
```

### Transcribe + summarize with Markdown report

```powershell
python -m speechsum transcribe podcast.mp3 --output markdown --output-file report.md
```

### Record from microphone

```powershell
python -m speechsum record --duration 60
```

### Record and summarize

```powershell
python -m speechsum record --duration 120 --no-summarize
```

### Transcribe video file

```powershell
python -m speechsum transcribe meeting.mp4
```

### Launch web UI

```powershell
python -m speechsum serve
# Open http://127.0.0.1:8080
```

---

## 11. Configuration

Set via environment variables (prefix `SPEECHSUM_`) or `.env` file:

```powershell
# Use a different summarization model (smaller = faster)
$env:SPEECHSUM_SUMMARIZATION_MODEL = "philschmid/bart-large-cnn-samsum"

# Use the large Vosk model for better accuracy
$env:SPEECHSUM_VOSK_MODEL_PATH = "C:\Users\NGFEP\VMax\models\vosk-model-en-us-0.22"

# Set default output format
$env:SPEECHSUM_DEFAULT_OUTPUT_FORMAT = "json"
```

Create a `.env` file in the project root to persist settings:

```
SPEECHSUM_SUMMARIZATION_MODEL=facebook/bart-large-cnn
SPEECHSUM_DEFAULT_OUTPUT_FORMAT=markdown
```

---

## 12. Development Commands

```powershell
# Run all tests with coverage
pytest tests/ -v --cov=src/speechsum

# Lint check
ruff check src/speechsum/ tests/

# Type check
mypy src/speechsum/ --ignore-missing-imports

# Run pre-commit hooks on all files
pre-commit run --all-files
```

---

## 13. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `No module named 'speechsum'` | Package not installed | `pip install -e .` |
| `ModelNotFoundError` | No Vosk model | `python -m speechsum download` |
| `TranscriptionError: empty result` | Audio has no speech | Use a file with actual speech content |
| `ModelLoadError: Unknown task` | Wrong transformers version | Already fixed in this repo |
| ffmpeg errors | ffmpeg not found | `pip install imageio-ffmpeg` |
| CUDA out of memory | GPU insufficient | System falls back to CPU automatically |
| Slow first run | Downloading ~1.5 GB model | Expected — cached after first run |
| `speechsum` command not found | Scripts not in PATH | Use `python -m speechsum` instead |
