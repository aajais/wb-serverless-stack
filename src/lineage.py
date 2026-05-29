"""Declare BIRD-dataset → wandb-run lineage via ``run.use_artifact``.

Each split is one aggregated Registry artifact (uploaded by
``data.registry_uploader upload-dataset``). Calling ``use_artifact`` once per
split records a lineage edge, an "Input Artifacts" entry on the run, and the
run in the Registry "Consumers" tab. It's metadata only — no download; the
sandbox pulls each DB later per-entry via ``SandboxPool.ensure_db``.

The actual ``db_ids`` a run touches are recorded in run config under
``bird.<split>.db_ids`` for later analysis, separate from the per-split edge.
"""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rich.console import Console

from src.config import settings

console = Console()


@functools.cache
def _api() -> Any:
    """Process-wide ``wandb.Api`` for Registry artifact lookups."""
    import wandb  # type: ignore[import-not-found]

    return wandb.Api(api_key=os.environ.get("WANDB_API_KEY"))


def declare_db_lineage(
    run: Any,
    split: str,
    db_ids: Iterable[str],
    *,
    artifact_type: str = "dataset",
) -> str | None:
    """Declare the split's aggregated BIRD artifact as an input to ``run``.

    Args:
        run: An active ``wandb.Run`` (e.g. ``model._get_wandb_run()`` for ART,
            or the return of ``wandb.init`` for the baseline).
        split: ``"dev"`` or ``"train"`` — picks the right Registry collection.
        db_ids: Iterable of distinct ``db_id`` strings the run will actually
            sample from. Recorded under ``run.config['bird'][split]['db_ids']``
            for downstream analysis; not used for the lineage edge itself
            (the whole split artifact is the input).
        artifact_type: Type the artifact was uploaded as. Must match the
            ``--type`` used in ``upload-dataset`` (default ``dataset``).

    Returns:
        The qualified artifact path that was declared, or None if Registry
        isn't configured or the lookup fails (non-fatal — useful for ad-hoc
        smoke runs).
    """
    qualified = settings.registry_dataset_artifact_path(split)
    if qualified is None:
        return None

    # Record touched db_ids even if the artifact lookup later fails, so the
    # run config still reflects what data it consumed.
    touched = sorted(set(db_ids))
    try:
        bird_cfg = dict(getattr(run, "config", {}).get("bird", {}) or {})
        bird_cfg[split] = {"db_ids": touched, "qualified": qualified}
        run.config["bird"] = bird_cfg
    except Exception:  # noqa: BLE001
        pass  # config update is best-effort

    # Registry paths (wandb-registry-<name>/<collection>:<ver>) can't be passed
    # as strings to run.use_artifact (the parser reads the first segment as an
    # entity). Resolve to an Artifact via api.artifact() first, then pass that.
    api = _api()
    try:
        art = api.artifact(qualified, type=artifact_type)
        run.use_artifact(art)
    except Exception as e:  # noqa: BLE001
        console.log(
            f"[yellow][lineage] use_artifact({qualified!r}) failed (non-fatal):[/] "
            f"{type(e).__name__}: {e}"
        )
        return None

    console.log(
        f"[lineage] declared {qualified!r} as input to "
        f"run={getattr(run, 'name', '?')!r} ({len(touched)} db_ids touched)"
    )
    return qualified


def db_ids_from_dev_subset() -> list[str]:
    """Distinct db_ids referenced by the BIRD-dev-200 jsonl (deterministic)."""
    path: Path = settings.bird_dev_subset
    if not path.exists():
        return []
    out: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.add(json.loads(line)["db_id"])
            except (ValueError, KeyError):
                continue
    return sorted(out)


def db_ids_from_train_rows(rows: Iterable[dict]) -> list[str]:
    """Distinct db_ids referenced by an iterable of BIRD-train rows."""
    return sorted({r["db_id"] for r in rows if "db_id" in r})
