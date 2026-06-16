from __future__ import annotations

import uuid
from pathlib import Path

import structlog
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from speechsum.pipeline import Pipeline

logger = structlog.get_logger()

HERE = Path(__file__).resolve().parent
TEMPLATES_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"

app = FastAPI(title="speechsum", version="0.1.0")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled_error", path=request.url.path)
    return JSONResponse({"error": str(exc)}, status_code=500)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_request: Request, exc: StarletteHTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),  # noqa: B008
    summarize: bool = Form(True),  # noqa: B008
):
    if not file.filename:
        return JSONResponse({"error": "No file provided"}, status_code=400)

    ext = Path(file.filename).suffix.lower()
    supported = {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
        ".aiff",
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
    }
    if ext not in supported:
        return JSONResponse(
            {"error": f"Unsupported format: {ext}"},
            status_code=400,
        )

    temp_dir = Path("tmp")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / f"{uuid.uuid4().hex}{ext}"

    try:
        content = await file.read()
        temp_path.write_bytes(content)

        logger.info("processing_upload", file=file.filename, size=len(content))
        pipeline = get_pipeline()
        result = pipeline.run(source=temp_path, summarize=summarize)
        return JSONResponse(result.to_dict())
    except Exception as e:
        logger.exception("processing_failed", file=file.filename)
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def run_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    import uvicorn

    logger.info("web_server_starting", host=host, port=port)
    uvicorn.run(app, host=host, port=port, log_level="info")
