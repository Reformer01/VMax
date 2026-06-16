from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests
import structlog

from speechsum.config import settings
from speechsum.exceptions import ModelNotFoundError

logger = structlog.get_logger()


def installed_models() -> list[dict]:
    models = []
    model_dir = settings.model_dir
    if not model_dir.exists():
        return models
    for entry in model_dir.iterdir():
        if entry.is_dir() and (entry / "am" ).exists():
            models.append({
                "path": str(entry),
                "name": entry.name,
                "size_mb": _dir_size_mb(entry),
            })
    return models


def ensure_model(model_name: str | None = None) -> Path:
    name = model_name or settings.vosk_model_name
    model_dir = settings.model_dir
    model_path = model_dir / name

    if model_path.exists():
        logger.info("vosk_model_found", path=str(model_path))
        return model_path

    if settings.vosk_model_path and Path(settings.vosk_model_path).exists():
        logger.info("vosk_model_from_config", path=settings.vosk_model_path)
        return Path(settings.vosk_model_path)

    logger.warning("vosk_model_not_found", name=name)
    raise ModelNotFoundError(
        f"Vosk model '{name}' not found at {model_path}. "
        f"Download it from {settings.vosk_model_url} "
        f"and extract it to {model_dir}/"
    )


def download_model(
    url: str | None = None,
    model_name: str | None = None,
    force: bool = False,
) -> Path:
    url = url or settings.vosk_model_url
    name = model_name or settings.vosk_model_name
    model_dir = settings.model_dir
    model_path = model_dir / name

    if model_path.exists() and not force:
        logger.info("vosk_model_exists", path=str(model_path))
        return model_path

    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("vosk_model_downloading", url=url)
    try:
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ModelNotFoundError(f"Failed to download model from {url}: {e}") from e

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(path=model_dir)
    except zipfile.BadZipFile as e:
        raise ModelNotFoundError(f"Downloaded file is not a valid zip: {e}") from e

    if not model_path.exists():
        alt_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        if alt_dirs:
            alt = max(alt_dirs, key=lambda d: _dir_size_mb(d))
            logger.info("vosk_model_renamed", from_path=str(alt), to_path=str(model_path))
            alt.rename(model_path)

    logger.info("vosk_model_downloaded", path=str(model_path))
    return model_path


def _dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 1)
