"""Console helpers — beautiful, zero-config output via ``rich``."""

from __future__ import annotations

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


def success(message: str) -> None:
    console.print(f"[bold green]✔[/] {message}")


def info(message: str) -> None:
    console.print(f"[bold cyan]ℹ[/] {message}")


def warning(message: str) -> None:
    console.print(f"[bold yellow]⚠[/] {message}")


def error(message: str) -> None:
    console.print(f"[bold red]✘[/] {message}")


def table(rows, title: str | None = None) -> str:
    """Render a two-column (name, value) table and return it as text."""
    tbl = Table(title=title, show_header=True, header_style="bold magenta")
    tbl.add_column("Property", style="cyan", no_wrap=True)
    tbl.add_column("Value", style="white")
    for name, value in rows:
        tbl.add_row(str(name), str(value))
    with console.capture() as capture:
        console.print(tbl)
    return capture.get()


def progress(enabled: bool = True):
    if not enabled:
        class _Null:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def add_task(self, *a, **k):
                return 0

            def advance(self, *a, **k):
                return None

        return _Null()
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
