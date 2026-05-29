"""Fetch and unpack the BIRD-SQL train and dev splits into ``data/bird/``.

Each split ships per-database SQLite files plus a JSON of
(question, db_id, gold SQL, difficulty) records.

Run:
    python -m data.download_bird
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import requests
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

# Official mirror — see https://bird-bench.github.io/
BIRD_TRAIN_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
BIRD_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bird"
console = Console()


def _stream_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        console.log(f"[dim]Already present:[/] {dest.name}")
        return
    console.log(f"[bold]Downloading[/] {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(dest.name, total=total)
            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
                    progress.advance(task, len(chunk))


def _extract(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    console.log(f"[bold]Extracting[/] {zip_path.name} → {dest_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    # Each split zip nests a *_databases.zip holding the DBs; unwrap it.
    nested = list(dest_dir.rglob("train_databases.zip")) + list(dest_dir.rglob("dev_databases.zip"))
    for inner in nested:
        console.log(f"[bold]Unpacking nested[/] {inner.name}")
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(inner.parent)
        inner.unlink()


def _verify_split(split_dir: Path, split: str) -> None:
    question_file = next(split_dir.rglob(f"{split}.json"), None)
    if question_file is None:
        console.log(f"[red]Could not find {split}.json under {split_dir}[/]")
        return
    rows = json.loads(question_file.read_text())
    db_dirs = [p for p in split_dir.rglob("*") if p.is_dir() and any(p.glob("*.sqlite"))]
    console.log(
        f"[green]{split}[/]: {len(rows)} questions, {len(db_dirs)} databases, json at {question_file.relative_to(DATA_DIR.parent.parent)}"
    )


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_zip = DATA_DIR / "train.zip"
    dev_zip = DATA_DIR / "dev.zip"

    _stream_download(BIRD_TRAIN_URL, train_zip)
    _stream_download(BIRD_DEV_URL, dev_zip)

    _extract(train_zip, DATA_DIR / "train")
    _extract(dev_zip, DATA_DIR / "dev")

    _verify_split(DATA_DIR / "train", "train")
    _verify_split(DATA_DIR / "dev", "dev")

    console.log("[bold green]BIRD data ready.[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
