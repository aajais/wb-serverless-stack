"""Smoke test the reward function on hand-picked (model_sql, ref_sql) pairs.

Runs in-process via ``FT_SD_LOCAL=1`` so no sandbox creds are required.
"""

from __future__ import annotations

import asyncio
import os
import sys

from rich.console import Console
from rich.table import Table

console = Console()


PAIRS: list[dict] = [
    # ----- correct -----
    {
        "name": "exact-match identical SQL",
        "db_id": "california_schools",
        "model_sql": "SELECT COUNT(*) FROM schools WHERE State = 'CA'",
        "ref_sql": "SELECT COUNT(*) FROM schools WHERE State = 'CA'",
        "expected_reward": 1,
    },
    {
        "name": "equivalent SQL via different alias",
        "db_id": "california_schools",
        "model_sql": "SELECT COUNT(*) AS n FROM schools s WHERE s.State = 'CA'",
        "ref_sql": "SELECT COUNT(*) FROM schools WHERE State = 'CA'",
        "expected_reward": 1,
    },
    {
        # Same rows, different sort order: multiset compare ignores ordering.
        "name": "ORDER BY direction doesn't matter for multiset compare",
        "db_id": "california_schools",
        "model_sql": "SELECT CDSCode FROM schools WHERE State='CA' ORDER BY CDSCode DESC",
        "ref_sql": "SELECT CDSCode FROM schools WHERE State='CA' ORDER BY CDSCode ASC",
        "expected_reward": 1,
    },
    # ----- wrong -----
    {
        "name": "wrong filter column",
        "db_id": "california_schools",
        "model_sql": "SELECT COUNT(*) FROM schools WHERE State = 'NY'",
        "ref_sql": "SELECT COUNT(*) FROM schools WHERE State = 'CA'",
        "expected_reward": 0,
    },
    {
        "name": "syntactically broken model SQL",
        "db_id": "california_schools",
        "model_sql": "SELEC COUN(*) FORM schools",
        "ref_sql": "SELECT COUNT(*) FROM schools",
        "expected_reward": 0,
    },
]


async def run_one(pair: dict) -> dict:
    from src.reward import score_sql

    r = await score_sql(pair["model_sql"], pair["ref_sql"], pair["db_id"], split="train")
    return {
        **pair,
        "actual_reward": r["reward"],
        "model_err": r["model_error"],
        "ref_err": r["ref_error"],
    }


async def main() -> int:
    os.environ["FT_SD_LOCAL"] = "1"

    results = []
    for pair in PAIRS:
        try:
            results.append(await run_one(pair))
        except Exception as e:  # noqa: BLE001
            results.append({**pair, "actual_reward": "ERR", "model_err": str(e), "ref_err": None})

    table = Table(title="smoke_reward results")
    table.add_column("case")
    table.add_column("expected", justify="right")
    table.add_column("actual", justify="right")
    table.add_column("pass?", justify="center")
    table.add_column("model_err", overflow="fold")
    table.add_column("ref_err", overflow="fold")

    passed = 0
    for r in results:
        ok = r["actual_reward"] == r["expected_reward"]
        passed += int(ok)
        table.add_row(
            r["name"],
            str(r["expected_reward"]),
            str(r["actual_reward"]),
            "[green]✓[/]" if ok else "[red]✗[/]",
            str(r["model_err"]) if r["model_err"] else "",
            str(r["ref_err"]) if r["ref_err"] else "",
        )
    console.print(table)
    console.print(f"[bold]{passed}/{len(results)} pass[/]")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
