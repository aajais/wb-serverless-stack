"""Lazy loaders for BIRD train + dev splits, used by the training loop."""

from __future__ import annotations

import functools
import json

from src.config import settings


@functools.cache
def load_bird_train() -> list[dict]:
    train_dir = settings.bird_data_dir / "train"
    candidates = list(train_dir.rglob("train.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No train.json under {train_dir} — run `python -m data.download_bird` first."
        )
    rows = json.loads(candidates[0].read_text())
    for r in rows:
        r["sql"] = r.get("SQL") or r.get("sql")
        r.setdefault("difficulty", "unknown")
    return rows


@functools.cache
def load_bird_dev_full() -> list[dict]:
    dev_dir = settings.bird_data_dir / "dev"
    candidates = list(dev_dir.rglob("dev.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No dev.json under {dev_dir} — run `python -m data.download_bird` first."
        )
    rows = json.loads(candidates[0].read_text())
    for r in rows:
        r["sql"] = r.get("SQL") or r.get("sql")
        r.setdefault("difficulty", "unknown")
    return rows
