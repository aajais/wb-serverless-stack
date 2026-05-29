"""SQL runner that executes *inside* the sandbox.

Pushed in once per sandbox, then invoked per rollout with
``--db <path>.sqlite --sql-file /tmp/q.sql``. Prints one JSON line on stdout:
``{"rows": [...], "error": null|str, "elapsed_ms": float}``. SQL comes from a
file (never shell-interpolated); a hard timeout is enforced via conn.interrupt().
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import threading
import time
from pathlib import Path


def _normalize_row(row: tuple) -> tuple:
    out = []
    for v in row:
        if isinstance(v, bytes):
            out.append(v.hex())
        elif isinstance(v, float):
            # round to avoid float-noise mismatches between equivalent queries
            out.append(round(v, 6))
        else:
            out.append(v)
    return tuple(out)


def run_query(db_path: str, sql: str, timeout_sec: float) -> dict:
    started = time.monotonic()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout_sec)
    timer = threading.Timer(timeout_sec, conn.interrupt)
    timer.daemon = True
    timer.start()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        raw_rows = cur.fetchall()
        rows = [_normalize_row(r) for r in raw_rows]
        return {
            "rows": rows,
            "row_count": len(rows),
            "error": None,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        }
    except sqlite3.OperationalError as e:
        # Most timeouts arrive as OperationalError("interrupted")
        msg = str(e)
        if "interrupted" in msg.lower():
            msg = f"timeout after {timeout_sec}s"
        return {
            "rows": [],
            "row_count": 0,
            "error": msg,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        }
    except sqlite3.Error as e:
        return {
            "rows": [],
            "row_count": 0,
            "error": f"{type(e).__name__}: {e}",
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        }
    finally:
        timer.cancel()
        conn.close()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--sql-file", required=True)
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        print(
            json.dumps(
                {"rows": [], "row_count": 0, "error": f"db not found: {db_path}", "elapsed_ms": 0.0}
            )
        )
        return 0

    sql = Path(args.sql_file).read_text().strip()
    if not sql:
        print(json.dumps({"rows": [], "row_count": 0, "error": "empty SQL", "elapsed_ms": 0.0}))
        return 0

    print(json.dumps(run_query(str(db_path), sql, args.timeout), default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
