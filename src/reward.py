"""Execution-based reward: run model and reference SQL, reward 1 iff row multisets match.

Queries run inside a wandb Sandbox via ``SandboxPool``. ``FT_SD_LOCAL=1`` runs
sqlite3 in-process instead, for smoke tests with no sandbox creds.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import weave

from src.config import settings
from src.sandbox_pool import HELPER_REMOTE, current_pool
from src.schema import db_path_for

_LOCAL_MODE = os.environ.get("FT_SD_LOCAL", "0") == "1"


def _normalize_row(row: tuple) -> tuple:
    out: list[Any] = []
    for v in row:
        if isinstance(v, bytes):
            out.append(v.hex())
        elif isinstance(v, float):
            out.append(round(v, 6))
        else:
            out.append(v)
    return tuple(out)


def _run_local(db_path: Path, sql: str, timeout: float) -> dict:
    """In-process sqlite execution; same return shape as the in-sandbox helper."""
    started = time.monotonic()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
    timer = threading.Timer(timeout, conn.interrupt)
    timer.daemon = True
    timer.start()
    try:
        rows = [_normalize_row(r) for r in conn.execute(sql).fetchall()]
        return {
            "rows": rows,
            "row_count": len(rows),
            "error": None,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        }
    except sqlite3.OperationalError as e:
        msg = str(e)
        if "interrupted" in msg.lower():
            msg = f"timeout after {timeout}s"
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


def _row_multiset(rows: list[tuple]) -> Counter:
    return Counter(tuple(str(v) if v is not None else None for v in r) for r in rows)


def result_sets_match(model_rows: list[tuple], ref_rows: list[tuple]) -> bool:
    return _row_multiset(model_rows) == _row_multiset(ref_rows)


@weave.op
async def score_sql(
    model_sql: str,
    ref_sql: str,
    db_id: str,
    split: str = "train",
) -> dict:
    """Execute both queries, compare result sets, return reward ∈ {0,1}."""
    started = time.monotonic()
    timeout = float(settings.sql_timeout_sec)

    if _LOCAL_MODE:
        db_path = db_path_for(db_id, split=split)
        model_result = (
            _run_local(db_path, model_sql, timeout)
            if model_sql
            else {
                "rows": [],
                "row_count": 0,
                "error": "empty model SQL",
                "elapsed_ms": 0.0,
            }
        )
        ref_result = _run_local(db_path, ref_sql, timeout)
    else:
        pool = current_pool()
        if pool is None:
            raise RuntimeError(
                "score_sql called outside a SandboxPool. Wrap the caller in "
                "`async with SandboxPool(...)` or set FT_SD_LOCAL=1."
            )
        model_result, ref_result = await _run_via_pool(pool, model_sql, ref_sql, db_id, timeout)

    model_rows = model_result.get("rows", [])
    ref_rows = ref_result.get("rows", [])
    model_err = model_result.get("error")
    ref_err = ref_result.get("error")

    non_trivial = ref_err is None and len(ref_rows) > 0

    if model_err is not None or ref_err is not None:
        reward = 0
    else:
        reward = 1 if result_sets_match(model_rows, ref_rows) else 0

    return {
        "reward": reward,
        "non_trivial": non_trivial,
        "model_row_count": model_result.get("row_count", 0),
        "ref_row_count": ref_result.get("row_count", 0),
        "model_rows": model_rows[:50],
        "ref_rows": ref_rows[:50],
        "model_error": model_err,
        "ref_error": ref_err,
        "model_elapsed_ms": model_result.get("elapsed_ms"),
        "ref_elapsed_ms": ref_result.get("elapsed_ms"),
        "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
    }


async def _run_via_pool(
    pool: Any, model_sql: str, ref_sql: str, db_id: str, timeout: float
) -> tuple[dict, dict]:
    async with pool.checkout() as sb:
        # First rollout per (sandbox, db_id) pays the artifact download; later
        # ones reuse the cached copy. A failure means the registry/artifact is
        # missing, so surface it as reward 0 rather than crashing the run.
        try:
            db_path = await pool.ensure_db(sb, db_id)
        except Exception as e:  # noqa: BLE001
            err = {
                "rows": [],
                "row_count": 0,
                "error": f"ensure_db({db_id!r}) failed: {e}",
                "elapsed_ms": 0.0,
            }
            return err, err

        await asyncio.gather(
            sb.write_file("/tmp/q_model.sql", (model_sql or "").encode("utf-8")),
            sb.write_file("/tmp/q_ref.sql", (ref_sql or "").encode("utf-8")),
        )
        cmd_base = ["python", HELPER_REMOTE, "--db", db_path, "--timeout", str(timeout)]
        model_proc, ref_proc = await asyncio.gather(
            sb.exec(cmd_base + ["--sql-file", "/tmp/q_model.sql"]),
            sb.exec(cmd_base + ["--sql-file", "/tmp/q_ref.sql"]),
        )

    return _parse_helper(model_proc), _parse_helper(ref_proc)


def _parse_helper(proc_result: Any) -> dict:
    """Decode the single JSON line printed by ``run_sql.py``."""
    stdout = (proc_result.stdout or "").strip()
    if not stdout:
        stderr = (proc_result.stderr or "").strip()
        return {
            "rows": [],
            "row_count": 0,
            "error": f"sandbox: no stdout; stderr={stderr[:200]!r}",
            "elapsed_ms": 0.0,
        }
    try:
        return json.loads(stdout.splitlines()[-1])
    except (ValueError, IndexError) as e:
        return {
            "rows": [],
            "row_count": 0,
            "error": f"sandbox parse: {e}; raw={stdout[:200]!r}",
            "elapsed_ms": 0.0,
        }
