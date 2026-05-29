"""Mid-training callback that runs ``run_dev_scoring`` at step boundaries.

Logs via ``model.log(split="val", ...)`` so ART prefixes the metrics with
``val/*`` and plots them on the ``training_step`` x-axis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console

from src.config import settings
from src.eval_bird_dev import SQLCopilot, run_dev_scoring
from src.sandbox_pool import SandboxPool

if TYPE_CHECKING:
    import art  # noqa: F401

console = Console()


async def run_dev_scoring_step(model: art.TrainableModel, step: int) -> dict[str, float]:
    """Run the held-out scoring pass for the current step and log via ART."""
    import weave  # local import: train_serverless already called weave.init()

    client = model.openai_client()
    endpoint = str(getattr(client, "base_url", "")) or settings.wandb_training_url

    copilot = SQLCopilot(
        endpoint=endpoint,
        model_name=model.get_inference_name(),
        temperature=0.0,
    )

    # Tag the eval with the wandb run id so it appears under this run's
    # workspace Weave panel (the Models ↔ Weave linking pattern).
    wandb_run = model._get_wandb_run()  # noqa: SLF001
    eval_attrs = {"stack": "held-out-scoring", "step": step}
    if wandb_run is not None:
        eval_attrs["wandb-run-id"] = wandb_run.id

    console.log(f"[bold]Held-out scoring[/] step={step} endpoint={endpoint}")
    # Nested dev pool stacks on the outer train pool; __aexit__ restores it.
    async with SandboxPool(size=settings.sandbox_pool_size, split="dev"):
        with weave.attributes(eval_attrs):
            summary = await run_dev_scoring(
                copilot, step=step, weave_name=f"bird-dev-200-step-{step}"
            )

    # Weave summary shape: {scorer_name: {field: {"mean": x, ...}}}.
    scorer = summary.get("execution_match", {})
    flat = {
        f"execution_match/{field}/mean": float(scorer[field]["mean"])
        for field in ("exact", "non_trivial", "had_error")
        if field in scorer and "mean" in scorer[field]
    }
    await model.log(trajectories=None, split="val", metrics=flat, step=step)
    console.log({f"val/{k}": round(v, 4) for k, v in flat.items()})
    return flat
