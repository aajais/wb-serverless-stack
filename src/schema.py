"""Render a SQLite database's schema as CREATE TABLE text (+ sample rows) for the prompt."""

from __future__ import annotations

import functools
import sqlite3
from pathlib import Path

from src.config import settings


def _find_db(db_id: str, split: str = "train") -> Path:
    root = settings.bird_data_dir / split
    candidates = list(root.rglob(f"{db_id}.sqlite"))
    if not candidates:
        other = "dev" if split == "train" else "train"
        candidates = list((settings.bird_data_dir / other).rglob(f"{db_id}.sqlite"))
    if not candidates:
        raise FileNotFoundError(
            f"No SQLite file for db_id={db_id!r} under {settings.bird_data_dir}"
        )
    return candidates[0]


@functools.lru_cache(maxsize=128)
def render_schema(db_id: str, split: str = "train", include_sample_rows: int = 3) -> str:
    """Return CREATE TABLE statements (verbatim, no indexes/triggers) plus sample rows.

    Sample rows help the model disambiguate enum-like columns.
    """
    db_path = _find_db(db_id, split)
    parts: list[str] = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for t in tables:
            create_sql = (t["sql"] or "").strip()
            if not create_sql:
                continue
            parts.append(create_sql + ";")
            if include_sample_rows > 0:
                try:
                    rows = conn.execute(
                        f'SELECT * FROM "{t["name"]}" LIMIT {include_sample_rows}'
                    ).fetchall()
                except sqlite3.Error:
                    rows = []
                if rows:
                    cols = rows[0].keys()
                    header = " | ".join(cols)
                    body = "\n".join(" | ".join(_fmt(r[c]) for c in cols) for r in rows)
                    parts.append(
                        f"-- sample rows from {t['name']}:\n-- {header}\n"
                        + "\n".join(f"-- {line}" for line in body.splitlines())
                    )
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _fmt(v: object) -> str:
    if v is None:
        return "NULL"
    s = str(v).replace("\n", " ")
    return s[:60] + ("…" if len(s) > 60 else "")


def db_path_for(db_id: str, split: str = "train") -> Path:
    return _find_db(db_id, split)
