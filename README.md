# Finetune an open model end-to-end on W&B + CoreWeave without any infra to stand up

This is a guided tour of the **W&B + CoreWeave product stack** for post-training: a complete RL finetune where the training, inference, execution, data, and observability are each a managed product, and **you never touch a GPU**. Nothing has to run on your machine, the orchestrator can even run inside a serverless sandbox.

The stack:

- **[W&B Serverless RL](https://docs.wandb.ai/serverless-rl)** (via [ART](https://github.com/OpenPipe/ART)) —> the GRPO training loop, on CoreWeave GPUs.
- **W&B Inference** —> serves each new checkpoint for rollouts; the endpoint auto-updates as training advances.
- **[W&B Serverless Sandboxes](https://docs.wandb.ai/sandboxes)** —> safe execution of arbitrary model output, with fleet observability for free.
- **[W&B Registry](https://docs.wandb.ai/registry)** —> versioned datasets/artifacts and automatic lineage.
- **[Weave](https://weave-docs.wandb.ai/)** —> traces every rollout and runs held-out evals you can compare step-over-step.

The **example workload** is SQL: we GRPO-finetune Qwen3-30B-A3B to write SQL that *actually runs and returns the right rows*, because that makes the reward unambiguous a Sandbox runs both the model's SQL and the gold SQL and compares result sets, so there's no reward model and no LLM judge. Any execution-checkable task would exercise the same stack.

This README is the **clone-and-run guide**. For *why* the stack is built this way and how the products fit together, read the walkthrough: [`docs/narrative.md`](docs/narrative.md).

---

## Stack at a glance

Every layer below is a managed W&B/CoreWeave product you wire nothing up and provision no hardware:

| Layer | Product | Notes |
|---|---|---|
| Training | [W&B Serverless RL](https://docs.wandb.ai/serverless-rl) via [ART](https://github.com/OpenPipe/ART) | GRPO loop runs on CoreWeave GPUs |
| Rollout inference | W&B Inference | Serves each checkpoint; endpoint auto-updates as training advances |
| Base model | `Qwen/Qwen3-30B-A3B-Instruct-2507` | On the Serverless RL [supported list](https://docs.wandb.ai/serverless-rl/available-models) |
| Execution / reward | [Serverless Sandboxes](https://docs.wandb.ai/sandboxes) | Runs arbitrary model output safely; sqlite3 in-process for this example |
| Data | [W&B Registry](https://docs.wandb.ai/registry) | One artifact per split, lazy per-DB entry pulls, automatic lineage |
| Tracing + held-out scoring | [Weave](https://weave-docs.wandb.ai/) | `@weave.op` rollouts + `weave.Evaluation` for dev-200 |

---

## Prerequisites

- **Python ≥ 3.11**.
- A **W&B account** with an API key —> get yours at [wandb.ai/authorize](https://wandb.ai/authorize).

---

## 1. Clone and install

```bash
git clone <this-repo> && cd serverless-stack-workflow
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
| `WANDB_PROJECT` | Defaults to `sql-copilot-bird`, change if you like |

Every other knob (step counts, learning rate, sandbox pool size, registry names) has a sane default in `.env.example`. `.env` is loaded once, by `src/config.py`; CLI flags override env, which overrides defaults.

## 3. Get the data

```bash
python -m data.download_bird      # ~30GB train + 1.4GB dev, into data/bird/
python -m data.make_dev_subset    # writes data/dev_200.jsonl (deterministic, seed=0)
```

## 4. Publish the data to W&B Registry

**W&B Registry** versions your datasets and records lineage automatically. Training sandboxes pull each database from the Registry on demand. Publish each split **once** as a single aggregated artifact:

```bash
python -m data.registry_uploader upload-dataset --collection bird-dev   --root ./data/bird/dev
python -m data.registry_uploader upload-dataset --collection bird-train --root ./data/bird/train
```

Set `BIRD_REGISTRY_NAME` in `.env` to your registry.

## 5. Get a baseline number

Score the base model on dev-200 through **W&B Inference** and **Weave** before training, so you have something to compare against:

```bash
python -m scripts.make_baseline --endpoint https://api.inference.wandb.ai/v1 --model Qwen/Qwen3-30B-A3B-Instruct-2507
python -m scripts.smoke_eval     # optional: quick ~10-row sanity pass
```

The checked-in `out/baseline.json` recorded **49% exact-match** (5% error rate) for the base model.

## 6. Train

**W&B Serverless RL** drives the GRPO loop on CoreWeave GPUs, **W&B Inference** serves each new checkpoint, **Sandboxes** execute the reward, and **Weave** traces every rollout:

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

**Want the full story of how the product stack fits together?** → [`docs/narrative.md`](docs/narrative.md)
