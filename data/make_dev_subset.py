"""Build the fixed 200-row held-out scoring set.

Stratified over difficulty ∈ {simple, moderate, challenging} and over databases.
Seed = 0 so every checkpoint scores against the *same* rows.

Run:
    python -m data.make_dev_subset
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_DIR = REPO_ROOT / "data" / "bird" / "dev"
OUTPUT_PATH = REPO_ROOT / "data" / "dev_200.jsonl"

TARGET_PER_BUCKET = {"simple": 80, "moderate": 80, "challenging": 40}
SEED = 0


def _load_dev_rows() -> list[dict]:
    candidates = list(DEV_DIR.rglob("dev.json"))
    if not candidates:
        raise FileNotFoundError(f"No dev.json under {DEV_DIR} — run data/download_bird.py first.")
    rows = json.loads(candidates[0].read_text())
    # BIRD-dev fields: question_id, db_id, question, evidence, SQL, difficulty.
    # Normalize the gold SQL key to lowercase ``sql``.
    for r in rows:
        r["sql"] = r.get("SQL") or r.get("sql")
    return rows


def _stratified_sample(rows: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    by_diff: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_diff[r["difficulty"]].append(r)

    picked: list[dict] = []
    for difficulty, target in TARGET_PER_BUCKET.items():
        pool = by_diff.get(difficulty, [])
        if not pool:
            console.log(f"[yellow]No rows for difficulty={difficulty}[/]")
            continue
        # Secondary stratification: spread the picks across databases.
        by_db: dict[str, list[dict]] = defaultdict(list)
        for r in pool:
            by_db[r["db_id"]].append(r)
        for db_rows in by_db.values():
            rng.shuffle(db_rows)
        # Round-robin one row per database until we hit the target.
        flat = []
        i = 0
        db_queues = list(by_db.values())
        while len(flat) < target and any(db_queues):
            q = db_queues[i % len(db_queues)]
            if q:
                flat.append(q.pop())
            i += 1
            db_queues = [q for q in db_queues if q]
            if not db_queues:
                break
        picked.extend(flat[:target])

    rng.shuffle(picked)
    return picked


def _report(rows: list[dict]) -> None:
    by_diff: dict[str, int] = defaultdict(int)
    by_db: dict[str, int] = defaultdict(int)
    for r in rows:
        by_diff[r["difficulty"]] += 1
        by_db[r["db_id"]] += 1

    table = Table(title="dev_200 composition")
    table.add_column("difficulty")
    table.add_column("count", justify="right")
    for d, n in sorted(by_diff.items()):
        table.add_row(d, str(n))
    table.add_row("[bold]total[/]", f"[bold]{len(rows)}[/]")
    console.print(table)
    console.log(f"Distinct databases sampled: {len(by_db)}")


def main() -> int:
    rows = _load_dev_rows()
    console.log(f"Loaded {len(rows)} BIRD-dev rows")

    picked = _stratified_sample(rows)
    _report(picked)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        for r in picked:
            f.write(json.dumps(r) + "\n")
    console.log(f"[green]Wrote {len(picked)} rows →[/] {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
