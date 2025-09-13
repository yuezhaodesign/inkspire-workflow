#cli.py
from __future__ import annotations
import json
import os
import sys
import typer
from pathlib import Path
from typing import Optional

from .pipeline import run, RunArgs

app = typer.Typer(help="Inkspire: generate and post Perusall annotations")

def _echo_ok(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.GREEN)

def _echo_info(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.BLUE)

def _echo_err(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.RED, err=True)

@app.command()
def main(
    course_id: str = typer.Option(..., help="Perusall course ID"),
    document_id: str = typer.Option(..., help="Perusall document ID"),
    assignment_id: Optional[str] = typer.Option(None, help="Perusall assignment ID (required unless --dry-run)"),
    user_id: Optional[str] = typer.Option(None, help="Perusall user ID (required unless --dry-run)"),
    # objectives kept for backwards CLI compatibility but ignored
    objectives: Optional[str] = typer.Option(None, "--objectives", help="[DEPRECATED] Ignored; pipeline uses --kb only."),
    kb: Optional[str] = typer.Option(None, "--kb", help="Knowledge base folder (.txt/.md files)"),
    out: Path = typer.Option(Path("annotations.json"), help="Output preview/results path"),
    dry_run: bool = typer.Option(False, help="Preview JSON only; do not POST to Perusall"),
    provider: Optional[str] = typer.Option(None, help="LLM provider (default from INKSPIRE_PROVIDER or 'openai')"),
    export_get_dir: Optional[Path] = typer.Option(
        None,
        help="(Optional) Write GET artifacts (library JSON, raw textContent JSONs, cleaned text) to this folder",
    ),
):
    # Warn if deprecated objectives flag is used
    if objectives:
        typer.secho("⚠️  --objectives is deprecated and ignored. The pipeline uses only --kb.", fg=typer.colors.YELLOW)

    # Validate live-posting requirements early
    if not dry_run and (not assignment_id or not user_id):
        _echo_err("When not using --dry-run, both --assignment-id and --user-id are required.")
        raise typer.Exit(code=2)

    # Resolve provider with env fallback
    provider = (provider or os.getenv("INKSPIRE_PROVIDER") or "openai").strip().lower()

    args = RunArgs(
        course_id=course_id,
        document_id=document_id,
        assignment_id=assignment_id,
        user_id=user_id,
        objectives_path=objectives,  # kept for struct compatibility; ignored in pipeline.run()
        kb_folder=kb,
        dry_run=dry_run,
        provider=provider,
        out=str(out),
        export_get_dir=str(export_get_dir) if export_get_dir else None,
    )

    try:
        result = run(args)
    except Exception as e:
        _echo_err(f"❌ Run failed: {e}")
        raise typer.Exit(code=1)

    # Ensure parent directory exists
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if dry_run:
        _echo_ok(f"✅ Dry run complete. Wrote preview to: {out}  (items: {len(result)})")
    else:
        _echo_ok(f"✅ Posted {len(result)} annotation(s). Wrote API responses to: {out}")

if __name__ == "__main__":
    app()