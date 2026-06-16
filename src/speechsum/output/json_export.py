from __future__ import annotations

import json
from pathlib import Path

from speechsum.pipeline import PipelineResult


def export_json(result: PipelineResult, output_path: str | Path | None = None) -> str:
    data = result.to_dict()

    text = json.dumps(data, indent=2, ensure_ascii=False)

    if output_path:
        path = Path(output_path)
        path.write_text(text, encoding="utf-8")

    return text
