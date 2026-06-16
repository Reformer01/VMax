from __future__ import annotations

import click

from speechsum.config import settings
from speechsum.output.console import console, print_models, print_result
from speechsum.output.json_export import export_json
from speechsum.output.markdown import export_markdown
from speechsum.pipeline import Pipeline, PipelineResult
from speechsum.stt.model_manager import download_model, installed_models

_SUPPORTED_OUTPUTS = ["console", "json", "markdown"]


@click.group()
@click.version_option(version="0.1.0")
def main():
    """speechsum — Speech recognition and summarization system."""


@main.command()
@click.argument("source", type=click.Path(exists=True))
@click.option("--no-summarize", is_flag=True, help="Skip summarization")
@click.option("--output", "-o", type=click.Choice(_SUPPORTED_OUTPUTS), default=None)
@click.option("--output-file", type=click.Path(), default=None)
@click.option("--model", "-m", help="Path to Vosk model directory")
def transcribe(source: str, no_summarize: bool, output: str | None, output_file: str | None, model: str | None) -> None:
    """Transcribe and optionally summarize an audio or video file."""
    pipeline = Pipeline(stt_model_path=model)
    result = pipeline.run(source=source, summarize=not no_summarize)
    _handle_output(result, output, output_file)


@main.command()
@click.option("--duration", "-d", type=int, default=None, help="Recording duration in seconds")
@click.option("--no-summarize", is_flag=True, help="Skip summarization")
@click.option("--output", "-o", type=click.Choice(_SUPPORTED_OUTPUTS), default=None)
@click.option("--output-file", type=click.Path(), default=None)
def record(duration: int | None, no_summarize: bool, output: str | None, output_file: str | None) -> None:
    """Record from microphone, transcribe, and summarize."""
    pipeline = Pipeline()
    result = pipeline.run_from_recording(duration=duration, summarize=not no_summarize)
    _handle_output(result, output, output_file)


@main.command()
@click.option("--url", default=None, help="Download URL (default: alphacephei.com)")
@click.option("--model-name", default=None, help="Model name/folder name")
@click.option("--force", is_flag=True, help="Re-download even if exists")
def download(url: str | None, model_name: str | None, force: bool) -> None:
    """Download a Vosk speech recognition model."""
    path = download_model(url=url, model_name=model_name, force=force)
    click.echo(f"Model downloaded to: {path}")


@main.command()
def models() -> None:
    """List installed Vosk models."""
    models_list = installed_models()
    print_models(models_list)


@main.command()
@click.option("--port", "-p", type=int, default=8080, help="Web server port")
@click.option("--host", type=str, default="127.0.0.1", help="Web server host")
def serve(port: int, host: str) -> None:
    """Launch the web UI."""
    try:
        from speechsum.output.web.app import run_server
        run_server(host=host, port=port)
    except ImportError as e:
        click.echo(
            "Web dependencies not installed. Run: pip install speechsum[web]",
            err=True,
        )
        raise click.Abort() from e


@main.group()
def config():
    """View or modify configuration."""


@config.command(name="show")
def config_show():
    """Show current configuration."""
    import json

    from pydantic import BaseModel

    if isinstance(settings, BaseModel):
        click.echo(settings.model_dump_json(indent=2))
    else:
        click.echo(json.dumps(vars(settings), indent=2, default=str))


def _handle_output(
    result: PipelineResult,
    output_format: str | None,
    output_file: str | None,
) -> None:
    fmt = output_format or settings.default_output_format

    if fmt == "console":
        print_result(result)
        return

    if fmt == "json":
        text = export_json(result, output_path=output_file)
        if not output_file:
            console.print(text)
        else:
            click.echo(f"JSON written to: {output_file}")
        return

    if fmt == "markdown":
        text = export_markdown(result, output_path=output_file)
        if not output_file:
            console.print(text)
        else:
            click.echo(f"Markdown written to: {output_file}")
        return
