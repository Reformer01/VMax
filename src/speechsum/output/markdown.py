from __future__ import annotations

from datetime import datetime
from pathlib import Path

from speechsum.pipeline import PipelineResult


def export_markdown(result: PipelineResult, output_path: str | Path | None = None) -> str:
    lines: list[str] = [
        "# Speech Recognition & Summarization Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Source:** {result.source}",
        f"**Duration:** {result.duration_seconds:.1f}s",
        "",
        "## Model Information",
        "",
    ]

    for key, value in result.model_info.items():
        lines.append(f"- **{key}:** {value}")

    lines.extend([
        "",
        "## Transcription",
        "",
        result.transcription,
        "",
    ])

    if result.words:
        lines.extend([
            "",
            "### Word Timestamps",
            "",
            "| Word | Start | End | Confidence |",
            "|------|-------|-----|------------|",
        ])
        for w in result.words:
            lines.append(
                f"| {w.get('word', '')} | {w.get('start', 0):.2f}s | "
                f"{w.get('end', 0):.2f}s | {w.get('conf', 0):.2f} |"
            )
        lines.append("")

    if result.summary:
        lines.extend([
            "",
            "## Summary",
            "",
            result.summary,
            "",
        ])

    text = "\n".join(lines)

    if output_path:
        path = Path(output_path)
        path.write_text(text, encoding="utf-8")

    return text
