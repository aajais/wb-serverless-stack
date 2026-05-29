"""Headline training entrypoint — GRPO on W&B Serverless RL via ART.

Run:
    python -m src.train_serverless --max-steps 500 --eval-every 25

Per step: sample prompts, gather rollouts in parallel, run one GRPO step,
periodically score against held-out BIRD-dev-200.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import uuid

# Silence gRPC fork warnings from cwsandbox when ART forks. Must be set before
# any grpc import (pulled in transitively by art/weave/wandb).
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "true")
os.environ.setdefault("GRPC_POLL_STRATEGY", "poll")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import art  # type: ignore[import-not-found]  # noqa: E402
import weave  # noqa: E402
from rich.console import Console  # noqa: E402

import wandb  # noqa: E402
from src.config import settings
from src.data_loader import load_bird_train
from src.eval_bird_dev import load_dev_subset
from src.eval_callback import run_dev_scoring_step
from src.lineage import (  # noqa: E402
    db_ids_from_dev_subset,
    db_ids_from_train_rows,
    declare_db_lineage,
)
from src.rollout import configure_inference, rollout  # noqa: E402
from src.sandbox_pool import SandboxPool  # noqa: E402

console = Console()


async def train(max_steps: int, eval_every: int, seed: int = 0, data_source: str = "train") -> None:
    # Unique name per invocation → fresh ART model + fresh W&B run.
    run_name = f"{settings.sql_model_name}-{uuid.uuid4().hex[:8]}"
    model = art.TrainableModel(
        name=run_name,
        project=settings.wandb_project,
        entity=settings.wandb_entity,
        base_model=settings.base_model,
    )
    # Set before register() so this lands in the run's wandb.init(config=...).
    model.update_wandb_config(
        {
            "base_model": settings.base_model,
            "max_steps": max_steps,
            "eval_every": eval_every,
            "rollouts_per_prompt": settings.rollouts_per_prompt,
            "prompts_per_step": settings.prompts_per_step,
            "learning_rate": settings.learning_rate,
            "temperature": settings.sampling_temperature,
            "sandbox_pool_size": settings.sandbox_pool_size,
            "stack": "serverless-rl-art",
            "data_source": data_source,
        }
    )

    backend = art.ServerlessBackend(
        api_key=settings.wandb_api_key, base_url=settings.wandb_training_url
    )
    await model.register(backend)

    # Force the wandb run to exist now. ART creates it lazily on the first
    # non-empty _log_metrics call, which may not happen before we need it.
    art_run = model._get_wandb_run()  # noqa: SLF001 — only stable way to force init now
    # ART registers its run with reinit="create_new", which does NOT install it
    # as the wandb.run global. Both Weave (links traces to runs via wb_run_id)
    # and the cwsandbox WandbReporter read wandb.run; without this, traces get
    # wb_run_id=null and cwsandbox/* metrics are dropped.
    if art_run is not None:
        wandb.run = art_run

    # weave.init must come after wandb.run is set so traces capture the run_id.
    weave.init(settings.weave_project)

    # dev200 = plumbing smoke (dev DBs only); train = the real run.
    if data_source == "dev200":
        train_rows, rollout_split = load_dev_subset(), "dev"
    elif data_source == "train":
        train_rows, rollout_split = load_bird_train(), "train"
    else:
        raise ValueError(f"unknown --data-source: {data_source!r}")

    # Declare the run's BIRD split artifacts as inputs (metadata only, no
    # download). Dev is declared too because the held-out scoring step uses it.
    if art_run is not None:
        declare_db_lineage(art_run, split=rollout_split, db_ids=db_ids_from_train_rows(train_rows))
        if rollout_split != "dev":
            declare_db_lineage(art_run, split="dev", db_ids=db_ids_from_dev_subset())

    rng = random.Random(seed)
    start_step = await model.get_step()
    console.log(
        f"[bold]Starting at step={start_step}[/] with {len(train_rows)} rows "
        f"(source={data_source}, split={rollout_split})"
    )

    # Stash the openai client + inference name module-locally so they never
    # become @weave.op args. Weave can't serialize the patched AsyncOpenAI;
    # passing it as an op arg silently empties inputs:{} on every trace.
    configure_inference(model.openai_client(), model.get_inference_name())

    pool_size = settings.prompts_per_step * settings.rollouts_per_prompt
    async with SandboxPool(size=pool_size, split=rollout_split) as train_pool:
        for step in range(start_step, max_steps):
            prompts = rng.sample(train_rows, settings.prompts_per_step)
            console.log(
                f"[step {step}] {len(prompts)} prompts × {settings.rollouts_per_prompt} rollouts"
            )

            with weave.attributes({"step": step}):
                groups = await art.gather_trajectory_groups(
                    (
                        art.TrajectoryGroup(
                            rollout(prompts[i], step, split=rollout_split)
                            for _ in range(settings.rollouts_per_prompt)
                        )
                        for i in range(len(prompts))
                    ),
                    pbar_desc=f"step {step}",
                    max_exceptions=settings.rollouts_per_prompt,
                )

            result = await backend.train(model, groups, learning_rate=settings.learning_rate)
            await model.log(groups, metrics=result.metrics, step=result.step, split="train")
            # Flush cwsandbox/* counters onto the same training_step row.
            train_pool.log_metrics(step=result.step)

            if eval_every > 0 and step > 0 and step % eval_every == 0:
                try:
                    await run_dev_scoring_step(model, step)
                except Exception as e:  # noqa: BLE001
                    console.log(f"[red]Held-out scoring failed at step {step}: {e}[/]")

    # Final scoring pass — let exceptions surface.
    await run_dev_scoring_step(model, max_steps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=settings.train_max_steps)
    ap.add_argument("--eval-every", type=int, default=settings.eval_every_n_steps)
    ap.add_argument("--seed", type=int, default=settings.train_seed)
    ap.add_argument(
        "--data-source",
        choices=["train", "dev200"],
        default=settings.data_source,
        help="train = full BIRD-train (real run). dev200 = the 200-row dev subset, plumbing smoke only.",
    )
    args = ap.parse_args()
    asyncio.run(train(args.max_steps, args.eval_every, args.seed, args.data_source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
