"""ART rollout coroutine: render schema, prompt the policy, score the SQL, fill a Trajectory.

This is the W&B/Weave/Sandbox integration point. Past step 10, rollouts are
sampled into the traced ``@weave.op`` variant to limit trace volume.

The OpenAI client and inference model name are stashed module-locally via
``configure_inference`` instead of passed as args: Weave can't serialize the
AsyncOpenAI client, and a @weave.op arg it fails to encode silently drops every
trace's ``inputs``, leaving empty rows in the UI.
"""

from __future__ import annotations

import random
from typing import Any

import art  # type: ignore[import-not-found]
import weave
from openai import AsyncOpenAI

from src.config import settings
from src.prompts import SYSTEM_PROMPT, build_user_prompt, extract_sql
from src.reward import score_sql
from src.schema import render_schema

_CLIENT: AsyncOpenAI | None = None
_INFERENCE_MODEL_NAME: str | None = None


def configure_inference(client: AsyncOpenAI, inference_model_name: str) -> None:
    """Stash the OpenAI client + inference model name; called once per run.

    Keeps the AsyncOpenAI handle out of the @weave.op argument list so trace
    inputs survive Weave's encoder (see module docstring).
    """
    global _CLIENT, _INFERENCE_MODEL_NAME
    _CLIENT = client
    _INFERENCE_MODEL_NAME = inference_model_name


async def _do_rollout(
    row: dict,
    step: int,
    split: str,
) -> art.Trajectory:
    if _CLIENT is None or _INFERENCE_MODEL_NAME is None:
        raise RuntimeError("rollout.configure_inference(...) must be called before rollout()")
    db_id = row["db_id"]
    question = row["question"]
    evidence = row.get("evidence")
    difficulty = row.get("difficulty", "unknown")
    ref_sql = row["sql"]

    schema_text = render_schema(db_id, split=split)
    user_msg = build_user_prompt(question, schema_text, evidence=evidence)

    traj = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        metadata={
            "question_id": row.get("question_id"),
            "db_id": db_id,
            "difficulty": difficulty,
            "split": split,
            "step": step,
        },
        reward=0.0,
    )

    completion = await _CLIENT.chat.completions.create(
        model=_INFERENCE_MODEL_NAME,
        messages=traj.messages(),
        max_tokens=settings.max_completion_tokens,
        temperature=settings.sampling_temperature,
    )
    choice = completion.choices[0]
    traj.messages_and_choices.append(choice)

    raw_text: str = choice.message.content or ""
    model_sql = extract_sql(raw_text)

    result: dict[str, Any] = await score_sql(model_sql, ref_sql, db_id, split=split)

    traj.reward = float(result["reward"])
    traj.metrics["non_trivial"] = float(result["non_trivial"])
    traj.metrics["had_model_error"] = float(result["model_error"] is not None)
    traj.metrics["had_ref_error"] = float(result["ref_error"] is not None)
    traj.metrics["model_row_count"] = float(result["model_row_count"])
    traj.metrics["ref_row_count"] = float(result["ref_row_count"])
    traj.metrics["sql_chars"] = float(len(model_sql))

    # Surface key fields on the trace so the Weave Traces table is readable
    # without expanding each row.
    traj.metadata["model_sql"] = model_sql[:2000]
    traj.metadata["reward"] = result["reward"]

    return traj


@weave.op
async def traced_rollout(
    row: dict,
    step: int,
    split: str = "train",
) -> art.Trajectory:
    """@weave.op variant: captures the rollout as a Weave trace.

    Tags the trace with step/db_id/difficulty/split/question_id for slicing later.
    """
    with weave.attributes(
        {
            "step": step,
            "db_id": row["db_id"],
            "difficulty": row.get("difficulty", "unknown"),
            "split": split,
            "question_id": row.get("question_id"),
        }
    ):
        return await _do_rollout(row, step, split)


async def rollout(row: dict, step: int, split: str = "train") -> art.Trajectory:
    """Trace every rollout for the first 10 steps, then sample at
    ``WEAVE_ROLLOUT_SAMPLE_RATE``.
    """
    sample_rate = 1.0 if step <= 10 else settings.weave_rollout_sample_rate
    if random.random() < sample_rate:
        return await traced_rollout(row, step, split)
    return await _do_rollout(row, step, split)
