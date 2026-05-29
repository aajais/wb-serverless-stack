# SQL Copilot: Fully Serverless GRPO finetune on BIRD-SQL with execution rewards

GRPO-finetune an open model (Qwen3-30B-A3B) to write SQL that **actually runs and returns the right rows**. The reward is execution-based: a [wandb Sandbox](https://docs.wandb.ai/sandboxes) runs both the model's SQL and the gold SQL and compares the result sets, no reward model, no LLM judge. The full stack is **W&B Serverless RL (via [ART](https://github.com/OpenPipe/ART)) + [Weave](https://weave-docs.wandb.ai/) + Sandboxes + [Registry](https://docs.wandb.ai/registry)**.

**Fully serverless: you never touch a GPU.** Training, inference, and the execution reward all run on managed W&B infra, and even the orchestrator can run inside a serverless sandbox, so nothing has to run on your machine.

This README is the **clone-and-run guide**. For *why* the stack is built this way and how the pieces fit together, read the walkthrough: [`docs/narrative.md`](docs/narrative.md).

---

## Prerequisites

- **Python ≥ 3.11** and [`uv`](https://docs.astral.sh/uv/) (the assumed package manager).
- A **W&B account** with an API key —> get yours at [wandb.ai/authorize](https://wandb.ai/authorize).
- **~32 GB free disk** for the BIRD dataset (~30 GB train + 1.4 GB dev).
- **Serverless RL access** for the headline training run. The smoke tests and baseline scoring work without it.

---

## 1. Clone and install

```bash
git clone <this-repo> && cd ft-sd-demo
uv venv && source .venv/bin/activate
uv pip install -e .
```

## 2. Configure credentials

```bash
cp .env.example .env
```

Open `.env` and fill in at minimum:

| Variable | What to put |
|---|---|
| `WANDB_API_KEY` | Your key from [wandb.ai/authorize](https://wandb.ai/authorize) |
| `WANDB_ENTITY` | Your W&B team/username |
| `WANDB_PROJECT` | Defaults to `sql-copilot-bird` — change if you like |

Every other knob (step counts, learning rate, sandbox pool size, registry names) has a sane default in `.env.example`. `.env` is loaded once, by `src/config.py`; CLI flags override env, which overrides defaults.

## 3. Get the data

```bash
python -m data.download_bird      # ~30GB train + 1.4GB dev, into data/bird/
python -m data.make_dev_subset    # writes data/dev_200.jsonl (deterministic, seed=0)
```

The dev subset is fully reproducible — `seed=0` yields the same 200 rows on any machine.

## 4. Publish the data to W&B Registry

Training sandboxes pull each database from the Registry on demand. Publish each split **once** as a single aggregated artifact:

```bash
python -m data.registry_uploader upload-dataset --collection bird-dev   --root ./data/bird/dev
python -m data.registry_uploader upload-dataset --collection bird-train --root ./data/bird/train
```

Set `BIRD_REGISTRY_NAME` in `.env` to your registry.

## 6. Get a baseline number

Score the base model on dev-200 before training, so you have something to compare against:

```bash
python -m scripts.make_baseline --endpoint https://api.inference.wandb.ai/v1 --model Qwen/Qwen3-30B-A3B-Instruct-2507
python -m scripts.smoke_eval     # optional: quick ~10-row sanity pass
```

The checked-in `out/baseline.json` recorded **49% exact-match** (5% error rate) for the base model.

## 7. Train

```bash
# Headline run (needs Serverless RL)
python -m src.train_serverless --max-steps 500 --eval-every 25

# Dev DBs only, fast, validates the whole loop
python -m src.train_serverless --data-source dev200 --max-steps 200
```

Want it to survive closing your laptop? Run the same orchestrator inside serverless sandbox (the GPU work still happens on the CW fleet):

```bash
python scripts/run_in_sandbox.py -- --data-source dev200 --max-steps 200 --eval-every 25
```

Follow the run live in your W&B workspace, Weave **Traces** shows every rollout, and Weave **Evals** compares held-out passes step-over-step.

---

## Local fallback

`src/train_local.py` is a **rollout + reward harness** (not a full GRPO loop): it reuses the same Weave + Sandbox + eval plumbing but writes `(prompt, completion, reward)` JSONL to `out/rollouts/` for your own optimizer to consume.

```bash
# Shell 1 — serve the policy
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --port 8000 --enable-lora --max-loras 2 --max-lora-rank 32
# Shell 2
python -m src.train_local --max-steps 100
```

---

## Notes for running this yourself

- **All entrypoints run as modules** (`python -m src.train_serverless`), never as file paths, imports assume the package root is on the path.
- **`SANDBOX_MAX_LIFETIME_SEC` must exceed your run's wall-clock time** (default 4h). Pooled sandboxes live for the whole run; if the backend reaps one mid-training you get a cryptic gRPC `"Socket closed"`.

---

## Stack at a glance

| Layer | Choice | Notes |
|---|---|---|
| Training | [W&B Serverless RL](https://docs.wandb.ai/serverless-rl) via [ART](https://github.com/OpenPipe/ART) | Runs on CoreWeave GPUs |
| Base model (headline) | `Qwen/Qwen3-30B-A3B-Instruct-2507` | On the Serverless RL [supported list](https://docs.wandb.ai/serverless-rl/available-models) |
| Base model (fallback) | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | Used by `src/train_local.py` (TRL + vLLM) |
| Execution / reward | [wandb Sandboxes](https://docs.wandb.ai/sandboxes) | sqlite3 in-process inside the sandbox |
| Data | [W&B Registry](https://docs.wandb.ai/registry) | One artifact per split, lazy per-DB entry pulls |
| Tracing + held-out scoring | [Weave](https://weave-docs.wandb.ai/) | `@weave.op` rollouts + `weave.Evaluation` for dev-200 |

**Want the full story of how and why these clip together?** → [`docs/narrative.md`](docs/narrative.md)
