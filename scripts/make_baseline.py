"""Score the base model on BIRD-dev-200 to establish the baseline.

Opens a wandb run so the Weave eval links to it, the BIRD-dev DBs are recorded
as input artifacts, and the summary JSON ships as an output artifact that
downstream training runs can compare against.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import weave
from rich.console import Console

import wandb
from src.config import settings
from src.eval_bird_dev import SQLCopilot, run_dev_scoring
from src.lineage import db_ids_from_dev_subset, declare_db_lineage
from src.sandbox_pool import SandboxPool

console = Console()


async def run_baseline(endpoint: str, model_name: str, out_path: Path) -> dict:
    # Open the wandb run before weave.init(): Weave reads ``wandb.run`` at init
    # time to stamp wb_run_id on every @weave.op call.
    run = wandb.init(
        entity=settings.wandb_entity,
        project=settings.wandb_project,
        job_type="baseline-eval",
        name=f"baseline-{model_name.replace('/', '-')}",
        tags=["baseline", "bird-dev-200"],
        config={
            "endpoint": endpoint,
            "model": model_name,
            "dataset": "bird-dev-200",
            "temperature": 0.0,
            "stack": "weave-evaluation",
        },
    )
    weave.init(settings.weave_project)

    # Record every BIRD-dev DB this run consumes as lineage; edges show in the
    # run's "Used Artifacts" tab and each DB's Registry "Consumers" view.
    declare_db_lineage(run, split="dev", db_ids=db_ids_from_dev_subset())

    copilot = SQLCopilot(endpoint=endpoint, model_name=model_name, temperature=0.0)
    console.log(f"[bold]Baseline pass[/] model={model_name} @ {endpoint}")
    async with SandboxPool(size=settings.sandbox_pool_size, split="dev"):
        # Tag the eval with the wandb run id so it lands in this run's "Weave"
        # section in the workspace.
        with weave.attributes({"wandb-run-id": run.id, "stack": "baseline"}):
            summary = await run_dev_scoring(
                copilot,
                step=0,
                weave_name=f"bird-dev-200-baseline-{model_name.replace('/', '-')}",
            )

    # Flatten the summary into scalar metrics for the workspace charts.
    scorer = summary.get("execution_match", {}) if isinstance(summary, dict) else {}
    flat = {
        f"baseline/execution_match/{field}/mean": float(scorer[field]["mean"])
        for field in ("exact", "non_trivial", "had_error")
        if isinstance(scorer.get(field), dict) and "mean" in scorer[field]
    }
    if flat:
        run.summary.update(flat)

    # Ship as an output artifact so training runs can use_artifact() it and
    # make the comparison explicit in the lineage graph.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    art = wandb.Artifact(
        f"baseline-{model_name.replace('/', '-')}",
        type="evaluation",
        metadata={"model": model_name, "endpoint": endpoint, **flat},
    )
    art.add_file(str(out_path))
    run.log_artifact(art)

    run.finish()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=settings.eval_endpoint)
    ap.add_argument("--model", default=settings.eval_model)
    ap.add_argument("--out", type=Path, default=settings.baseline_out_path)
    args = ap.parse_args()

    summary = asyncio.run(run_baseline(args.endpoint, args.model, args.out))
    console.log(f"[green]Saved baseline summary →[/] {args.out}")
    console.print_json(json.dumps(summary, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
