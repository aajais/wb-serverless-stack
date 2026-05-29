# SQL copilot RL on BIRD-SQL — the walkthrough

A draft of the blog / customer-pitch narrative. The arc is dataset → reward
function → first rollouts in Weave → training curve in W&B → held-out curve
climbing → final drill-down.

> Replace bracketed `[…]` with the real run URLs and screenshot links before
> publishing.

---

## 1. The setup, in one paragraph

Every data-driven company has a "SQL copilot" project on the roadmap. The honest version of that project is: can a model take a question in English and a messy 30-table schema, and write SQL that *runs and returns the right rows*? Not SQL that compiles. Not SQL that looks correct. SQL that, when executed against the real database, returns the same result set as the analyst's gold query. That is the only definition of correctness an end user cares about.

That's exactly what BIRD-SQL measures: 12,751 questions across 95 real-world SQLite databases — healthcare, finance, education, sports — paired with gold SQL and a "did it execute to the same answer" scoring rule. BIRD-dev sits around 65-70% for frontier models. Plenty of headroom.

## 2. Why this is a perfect RL workload

Most RL post-training projects fight the same boss: building a non-cheatable reward function. LLM judges drift. Reward models miss-rank. Heuristics overfit. **SQL doesn't have this problem.** The reward is *the result set comparison*. Two queries, run them both, did they return the same rows? Done. Reward ∈ {0, 1}.

The catch: you need somewhere safe to actually *run* arbitrary model-generated SQL against arbitrary databases, thousands of times per training step. That's the sandbox.

## 3. Architecture in one diagram

```
       ┌────────────────── train_serverless.py ──────────────────┐
       │                                                          │
       │   for each step:                                         │
       │     sample N prompts from BIRD-train                     │
       │     for each prompt: K rollouts                          │
       │                                                          │
       │     each rollout:                                        │
       │       ─▶ Serverless RL endpoint generates SQL            │
       │       ─▶ wandb Sandbox runs model SQL + ref SQL          │
       │       ─▶ multiset-compare → reward ∈ {0, 1}              │
       │       ─▶ Weave records the whole rollout as a trace      │
       │                                                          │
       │     backend.train(model, groups)  ── GRPO step           │
       │                                                          │
       │     every 25 steps:                                      │
       │       run held-out Weave Evaluation over dev-200         │
       │       log dev/exact/mean to W&B                          │
       └──────────────────────────────────────────────────────────┘
```

## 4. The reward function, in 60 seconds

Open `src/reward.py`. The whole reward is one `@weave.op` named `score_sql`.

```python
@weave.op
async def score_sql(model_sql, ref_sql, db_id, split="train"):
    # 1. Borrow a warm sandbox from the pool
    # 2. Drop both SQL strings into /tmp/q_*.sql
    # 3. Invoke the in-sandbox helper (sqlite3 + JSON stdout)
    # 4. Multiset-compare the two row lists
    # 5. Return {reward: 0|1, model_rows, ref_rows, ...}
```

Two design choices worth flagging:

- **`mount_paths={...: "/data/dbs"}`** — the BIRD database folder is ~1GB and the same for every rollout. We mount it once when each sandbox is created; we never re-upload.
- **`SandboxPool(size=12)`** — sandbox cold start is 5–15 seconds. At 32 rollouts/step × 500 steps we'd otherwise burn months in cold starts alone. A small pool of long-lived sandboxes amortises that cost over the run.

## 5. First rollouts in the Weave Traces UI

After step 0, open the Weave Traces tab and look for ops named `rollout`. Each row is one trajectory. The columns we care about:

- `attributes.step` — the training step
- `attributes.db_id` — which BIRD database
- `attributes.difficulty` — `simple` / `moderate` / `challenging`
- `output.reward` — 0 or 1

Filter by `step == 0 AND reward == 0` to see what the model gets wrong at the start. Expand one row and you see:

- The exact user message we showed the model (schema + question)
- The completion (in full, including any explanation it tried to add despite our system prompt)
- The extracted SQL
- A nested `score_sql` op with both result sets side by side
- The reference SQL's result set, for contrast

This is the part that's hard to do without Weave: at step 0 the model is wrong in *many* different ways. At step 500 it's still wrong in a few — but a *characteristic* few. Slicing by `db_id` and `difficulty` turns the failure modes into something a human can reason about.

`[Insert screenshot: docs/screenshots/weave_traces_filtered.png]`

## 6. Training health in the W&B Workspace

Open the W&B run. Default panels you should see:

| Panel | What it's telling you |
|---|---|
| `train/reward_mean` | The headline. Should climb. |
| `train/loss` | GRPO policy loss. Smooth-ish, decreasing trend. |
| `train/kl` | KL to the reference policy. Should stay bounded (~0.05–0.5). If it explodes, lower the learning rate. |
| `train/completion_length` | Are completions getting shorter (good) or pathologically long (bad)? |
| `dev/execution_match/exact/mean` | The generalisation curve. The headline of the headline. |

The relationship to watch: `train/reward_mean` climbs first, `dev/.../mean` lags by a few hundred steps, then climbs too. If `train` rises but `dev` doesn't, you're memorising. (Possible with BIRD's 95 databases.)

`[Insert screenshot: docs/screenshots/wandb_workspace.png]`

## 7. Held-out scoring in the Weave Evals tab

Every 25 steps we re-run the same 200-row BIRD-dev subset against the auto-deployed Serverless RL endpoint for that step. Each pass shows up as its own row in Weave Evals, named `bird-dev-200-step-{N}`.

Comparison is what this UI is for. Pick the baseline pass and any later pass; Weave highlights:

- Per-row deltas (which questions flipped 0→1, which flipped 1→0)
- Per-scorer summary stats
- Per-attribute slices (`difficulty`, `db_id`)

The "what is the model actually learning" story comes out of here. Healthcare schemas tend to win first; finance schemas with their gnarly date semantics tend to lag.

`[Insert screenshot: docs/screenshots/weave_evals_compare.png]`

## 8. The drill-down ("what's it still getting wrong?")

At step 500, take the Weave Evals view, filter to `exact == 0`, sort by `db_id`. You'll see clusters. Pick the heaviest cluster (probably a finance DB with timestamp arithmetic) and open three rows. Almost always you'll see:

- The model used the wrong join key (BIRD has a lot of `subject_id`-vs-`id`-vs-`patient_no` ambiguity)
- The model summed instead of counted
- The model picked the right table but the wrong year filter

That's the prompt-engineering / DPO / multi-turn-retry opportunity. The next phase of the demo.

## 9. What we'd do next

- **Schema linking**: rather than dump 30 tables of schema text, retrieve the top-K tables for each question. Cuts prompt cost ~10×, often *improves* accuracy.
- **Multi-turn with execution feedback**: let the model see "your query errored with `no such column foo`" and retry. Same sandbox primitive, multiple `@weave.op` calls per trajectory.
- **DPO on the rollout pairs**: every step produces K rollouts per prompt with reward ∈ {0,1}. The (winner, loser) pairs are free DPO data. Easier supervision than GRPO.

## 10. Reproducing this

Everything is in the repo. `python -m src.train_serverless` works on Serverless RL (preview access required). `python -m src.train_local` works on a 2×H100 box with vLLM. Both write to the same W&B project, both use the same reward function, both use the same Weave traces.

See [`README.md`](../README.md) for the quickstart.
