"""Fallback rollout+reward harness for a local vLLM policy (NOT a GRPO loop).

Run (in two shells):
    vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --port 8000 \\
        --enable-lora --max-loras 2 --max-lora-rank 32
    python -m src.train_local --max-steps 100

Drives rollouts by hand and dumps (prompt, completion, reward) JSONL batches
for a separate TRL shim to optimize, so this script stays importable without
GPU deps. Same Weave + Sandbox plumbing as the serverless path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

import weave
from openai import AsyncOpenAI
from rich.console import Console

import wandb
from src.config import settings
from src.data_loader import load_bird_train
from src.eval_bird_dev import SQLCopilot, run_dev_scoring
from src.lineage import (
    db_ids_from_dev_subset,
    db_ids_from_train_rows,
    declare_db_lineage,
)
from src.prompts import SYSTEM_PROMPT, build_user_prompt, extract_sql
from src.reward import score_sql
from src.sandbox_pool import SandboxPool
from src.schema import render_schema

console = Console()

LOCAL_BASE_URL = settings.vllm_base_url
LOCAL_API_KEY = settings.vllm_api_key
LOCAL_MODEL = settings.local_model


async def _generate_completion(
    client: AsyncOpenAI, question: str, db_id: str, evidence: str | None
) -> str:
    schema_text = render_schema(db_id, split="train")
    user_msg = build_user_prompt(question, schema_text, evidence=evidence)
    completion = await client.chat.completions.create(
        model=LOCAL_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=settings.max_completion_tokens,
        temperature=settings.sampling_temperature,
    )
    return completion.choices[0].message.content or ""


@weave.op(tracing_sample_rate=1.0)
async def local_rollout(client: AsyncOpenAI, row: dict, step: int) -> dict:
    with weave.attributes(
        {
            "step": step,
            "db_id": row["db_id"],
            "difficulty": row.get("difficulty", "unknown"),
            "split": "train",
            "stack": "trl-local",
        }
    ):
        raw = await _generate_completion(client, row["question"], row["db_id"], row.get("evidence"))
        model_sql = extract_sql(raw)
        result = await score_sql(model_sql, row["sql"], row["db_id"], split="train")
        return {
            "question_id": row.get("question_id"),
            "db_id": row["db_id"],
            "model_sql": model_sql,
            "raw_completion": raw,
            "reward": float(result["reward"]),
            "non_trivial": result["non_trivial"],
            "model_error": result["model_error"],
        }


async def gather_rollouts_for_step(
    client: AsyncOpenAI, prompts: list[dict], step: int, n_per_prompt: int
) -> list[dict]:
    tasks = []
    for row in prompts:
        for _ in range(n_per_prompt):
            tasks.append(local_rollout(client, row, step))
    return await asyncio.gather(*tasks)


def _write_grpo_batch(rollouts: list[dict], out_path: Path) -> None:
    """Dump rollouts as JSONL for the out-of-process ``trl_grpo_step.py`` shim.

    TRL's GRPOTrainer consumes a (prompt, completion, reward) dataset; keeping
    that step out-of-process lets this script stay importable without GPU deps.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rollouts:
            f.write(json.dumps(r) + "\n")


async def train(max_steps: int, eval_every: int, seed: int = 0) -> None:
    run = wandb.init(
        entity=settings.wandb_entity,
        project=settings.wandb_project,
        name=f"{settings.sql_model_name}-local",
        config={
            "base_model": LOCAL_MODEL,
            "stack": "trl-vllm-local",
            "max_steps": max_steps,
            "eval_every": eval_every,
            "rollouts_per_prompt": settings.rollouts_per_prompt,
            "prompts_per_step": settings.prompts_per_step,
        },
    )
    weave.init(settings.weave_project)

    client = AsyncOpenAI(base_url=LOCAL_BASE_URL, api_key=LOCAL_API_KEY)

    train_rows = load_bird_train()
    declare_db_lineage(run, split="train", db_ids=db_ids_from_train_rows(train_rows))
    declare_db_lineage(run, split="dev", db_ids=db_ids_from_dev_subset())
    rng = random.Random(seed)
    rollout_dir = Path("./out/rollouts")

    pool_size = settings.prompts_per_step * settings.rollouts_per_prompt
    async with SandboxPool(size=pool_size, split="train"):
        for step in range(max_steps):
            prompts = rng.sample(train_rows, settings.prompts_per_step)
            console.log(f"[step {step}] gathering rollouts")
            rollouts = await gather_rollouts_for_step(
                client, prompts, step, settings.rollouts_per_prompt
            )

            rewards = [r["reward"] for r in rollouts]
            reward_mean = sum(rewards) / max(len(rewards), 1)
            wandb.log(
                {
                    "train/reward_mean": reward_mean,
                    "train/reward_nonzero_pct": sum(1 for r in rewards if r > 0)
                    / max(len(rewards), 1),
                    "train/step": step,
                },
                step=step,
            )
            batch_path = rollout_dir / f"step_{step:05d}.jsonl"
            _write_grpo_batch(rollouts, batch_path)
            console.log(f"[step {step}] reward_mean={reward_mean:.3f} → wrote {batch_path}")

            if eval_every > 0 and step > 0 and step % eval_every == 0:
                copilot = SQLCopilot(
                    endpoint=LOCAL_BASE_URL,
                    model_name=LOCAL_MODEL,
                    api_key=LOCAL_API_KEY,
                    temperature=0.0,
                )
                # Nested dev pool; __aexit__ restores the train pool.
                async with SandboxPool(size=settings.sandbox_pool_size, split="dev"):
                    summary = await run_dev_scoring(
                        copilot, step=step, weave_name=f"bird-dev-200-local-step-{step}"
                    )
                scorer = summary.get("execution_match", {}) if isinstance(summary, dict) else {}
                val_metrics = {
                    f"val/execution_match/{field}/mean": float(scorer[field]["mean"])
                    for field in ("exact", "non_trivial", "had_error")
                    if isinstance(scorer.get(field), dict) and "mean" in scorer[field]
                }
                if val_metrics:
                    wandb.log({**val_metrics, "train/step": step}, step=step)
                console.log(summary)

    wandb.finish()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=settings.train_local_max_steps)
    ap.add_argument("--eval-every", type=int, default=settings.eval_every_n_steps)
    ap.add_argument("--seed", type=int, default=settings.train_seed)
    args = ap.parse_args()
    asyncio.run(train(args.max_steps, args.eval_every, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
