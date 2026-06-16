from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from speechsum.pipeline import PipelineResult

console = Console()


def print_transcription(result: PipelineResult) -> None:
    console.print()
    console.print(Rule("[bold blue]Transcription"))
    console.print()
    console.print(Panel(
        result.transcription,
        title=f"Source: {result.source}",
        subtitle=f"Duration: {result.duration_seconds:.1f}s",
        border_style="blue",
    ))
    if result.words:
        table = Table(title="Word Timestamps", show_lines=True)
        table.add_column("Word", style="cyan")
        table.add_column("Start", style="green")
        table.add_column("End", style="green")
        table.add_column("Confidence", style="yellow")
        for w in result.words:
            table.add_row(
                w.get("word", ""),
                f"{w.get('start', 0):.2f}s",
                f"{w.get('end', 0):.2f}s",
                f"{w.get('conf', 0):.2f}",
            )
        console.print(table)


def print_summary(result: PipelineResult) -> None:
    if not result.summary:
        console.print("[yellow]No summary available[/yellow]")
        return
    console.print()
    console.print(Rule("[bold green]Summary"))
    console.print()
    console.print(Panel(
        result.summary,
        border_style="green",
    ))


def print_result(result: PipelineResult) -> None:
    console.print()
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="bold")
    meta.add_column()
    meta.add_row("Source", result.source)
    meta.add_row("Duration", f"{result.duration_seconds:.1f}s")
    meta.add_row("Transcription", f"{len(result.transcription)} chars")
    if result.summary:
        meta.add_row("Summary", f"{len(result.summary)} chars")

    console.print(Panel(
        meta,
        title="[bold]Pipeline Result[/bold]",
        border_style="white",
    ))

    print_transcription(result)
    if result.summary:
        print_summary(result)


def print_models(models: list[dict]) -> None:
    if not models:
        console.print("[yellow]No installed models found[/yellow]")
        return
    table = Table(title="Installed Vosk Models")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="blue")
    table.add_column("Size", style="green")
    for m in models:
        table.add_row(m["name"], m["path"], f"{m['size_mb']} MB")
    console.print(table)
