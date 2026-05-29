# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A demo that GRPO-finetunes an open-weights model to write SQL against BIRD-SQL schemas, using **execution-based rewards**: the reward function literally runs both the model's SQL and the gold SQL inside a wandb Sandbox and compares result sets. There is no reward model and no LLM judge — the sandbox *is* the reward signal. The stack is W&B Serverless RL (via OpenPipe ART) + Weave (tracing + held-out scoring) + wandb Sandboxes (execution).

## Commands

```bash
# Install (Python >=3.11; uv is the assumed package manager)
uv venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env          # fill in WANDB_API_KEY + WANDB_ENTITY at minimum

# Data: download BIRD (~30GB train + 1.4GB dev), then build the fixed dev subset
python -m data.download_bird
python -m data.make_dev_subset        # writes data/dev_200.jsonl (deterministic, seed=0)

# Publish BIRD DBs to W&B Registry (one aggregated artifact per split; required for training)
python -m data.registry_uploader upload-dataset --collection bird-dev   --root ./data/bird/dev
python -m data.registry_uploader upload-dataset --collection bird-train --root ./data/bird/train

# Smoke tests (no Serverless RL access needed; smoke_reward needs no sandbox creds at all)
python -m scripts.smoke_reward        # 5 known pairs, runs in-process via FT_SD_LOCAL=1
python -m scripts.smoke_eval          # ~10-row Weave scoring pass against any endpoint

# Baseline number for the base model on dev-200
python -m scripts.make_baseline --endpoint https://api.inference.wandb.ai/v1 --model Qwen/Qwen3-30B-A3B-Instruct-2507

# Headline training run
python -m src.train_serverless --max-steps 500 --eval-every 25
python -m src.train_serverless --data-source dev200 --max-steps 200   # plumbing smoke (dev DBs only)

# Run the orchestrator inside a sandbox instead of locally (heavy GPU work still on the fleet)
python scripts/run_in_sandbox.py -- --data-source dev200 --max-steps 200 --eval-every 25

# Lint (ruff; line-length 100, selects E/F/I/W/UP, ignores E501)
ruff check . && ruff format .
```

There is **no pytest suite**. "Tests" here are the `scripts/smoke_*.py` entrypoints. Run them after touching the reward function or scoring harness.

## Architecture

The data flow for one training step: sample BIRD rows → `rollout()` renders schema + prompts the policy → `extract_sql()` pulls the fenced SQL → `score_sql()` executes model+gold SQL in a sandbox and multiset-compares rows → reward ∈ {0,1} fills an `art.Trajectory` → ART runs the GRPO step. Periodically the same machinery scores a frozen 200-row dev subset.

Key modules (`src/`):
- `config.py` — single `settings` singleton (frozen dataclass, dotenv-backed). **Every tunable is here**; CLI flags override env which overrides defaults. Read this first.
- `reward.py` — `score_sql` (`@weave.op`); multiset row comparison after normalization (bytes→hex, floats→round(6), all→str). `FT_SD_LOCAL=1` bypasses the sandbox and runs sqlite3 in-process.
- `sandbox_pool.py` — `SandboxPool` async context manager holding warm sandboxes for the whole run; lazy per-DB artifact pulls.
- `rollout.py` — the W&B/Weave/Sandbox integration point; one trajectory per call.
- `train_serverless.py` — headline GRPO loop (ART `ServerlessBackend`).
- `train_local.py` — fallback; a rollout+reward *harness only* (writes `(prompt, completion, reward)` JSONL to `out/rollouts/`), NOT a full GRPO loop — the optimizer step lives outside this repo.
- `eval_bird_dev.py` / `eval_callback.py` — shared `weave.Evaluation` scoring harness; `eval_callback` runs it mid-training and logs `val/*` metrics on ART's `training_step` axis.
- `schema.py` — renders SQLite schema as CREATE TABLE text + sample rows (LRU-cached per db_id).
- `lineage.py` — declares per-split BIRD artifact as a wandb run input (metadata only, no download).
- `sandbox_runtime/run_sql.py` — runs *inside* the sandbox; prints one JSON line per query. Pushed in at pool boot.

## Non-obvious invariants — do not "clean these up"

These are load-bearing workarounds. Several have detailed comments at the code site; preserve the behavior.

- **Keep the `AsyncOpenAI` client out of every `@weave.op` signature.** `rollout.configure_inference()` stashes the client + inference model name module-locally on purpose. Weave can't serialize the patched client; passing it as an op arg silently drops `inputs` from every trace.
- **`wandb.run = art_run` is set manually** in `train_serverless.train`. ART creates its run with `reinit="create_new"`, which does NOT install it as the `wandb.run` global. Weave trace→run linking AND the cwsandbox `WandbReporter` both read `wandb.run`; without this, traces get `wb_run_id=null` and `cwsandbox/*` metrics vanish.
- **`weave.init()` must come after** `wandb.run` is assigned.
- **`SANDBOX_MAX_LIFETIME_SEC` must exceed the run's wall-clock time.** The pool holds sandboxes for the whole run; the SDK default lifetime is short and will reap them mid-run → gRPC "Socket closed".
- **`current_pool()` is a module-global** that `score_sql` reads to choose pool vs. local execution. The active pool is set/restored by `SandboxPool.__enter/__aexit__` (supports nesting — the dev scoring pool stacks on the train pool).
- **Sandbox close errors ("Failed to stop N sandbox(es)") are intentionally suppressed** so they don't mask the real error or the run summary.
- **Registry paths can't be passed as strings to `run.use_artifact`** — resolve via `api.artifact(qualified)` first, then pass the object (see `lineage.py`).
- **Each split is ONE aggregated artifact** holding every DB as an internal entry at `<db_id>/<db_id>.sqlite`. Sandboxes pull a single entry per rollout via `Artifact.get_entry(name).download(...)`, never the whole multi-GB blob.
- **Weave rollout tracing is sampled**: 100% for the first 10 steps, then `WEAVE_ROLLOUT_SAMPLE_RATE` (default 0.25) thereafter.

## Conventions

- All entrypoints are run as modules (`python -m src.train_serverless`, `python -m data.download_bird`), never as file paths — imports assume the package root is on the path.
- `.env` is loaded exactly once, by `config.py`, via `load_dotenv()`. Don't call it elsewhere.
- Async throughout the reward/rollout/eval path. Sandbox I/O is `await`ed and parallelized with `asyncio.gather`.
