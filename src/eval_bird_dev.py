"""Weave scoring harness over the fixed 200-row BIRD-dev subset.

Used by ``scripts/make_baseline.py`` and ``src/eval_callback.py``. Both feed the
same ``weave.Evaluation`` so the Evals tab can compare them directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import weave
from openai import AsyncOpenAI

from src.config import settings
from src.prompts import SYSTEM_PROMPT, build_user_prompt, extract_sql
from src.reward import score_sql
from src.schema import render_schema


def load_dev_subset() -> list[dict]:
    """Read the 200-row dev set produced by data/make_dev_subset.py."""
    path: Path = settings.bird_dev_subset
    if not path.exists():
        raise FileNotFoundError(
            f"Dev subset not found at {path}. Run `python -m data.make_dev_subset` first."
        )
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


class SQLCopilot(weave.Model):
    """Wraps an OpenAI-compatible chat endpoint into a weave.Model."""

    endpoint: str
    model_name: str
    _api_key: str = settings.eval_api_key
    max_tokens: int = 512
    temperature: float = 0.0  # greedy at scoring time

    # Built once per instance, not per dev-200 row. Typed Any because
    # weave.Model's pydantic base rejects AsyncOpenAI as a declared field.
    _client: Any = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            object.__setattr__(
                self, "_client", AsyncOpenAI(base_url=self.endpoint, api_key=self._api_key)
            )
        return self._client

    @weave.op
    async def predict(self, question: str, db_id: str, evidence: str | None = None) -> str:
        schema_text = render_schema(db_id, split="dev")
        user_msg = build_user_prompt(question, schema_text, evidence=evidence)
        completion = await self._get_client().chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return extract_sql(completion.choices[0].message.content or "")


@weave.op
async def execution_match(output: str, sql: str, db_id: str) -> dict:
    """Weave scorer: run model SQL + reference SQL and compare row sets."""
    result = await score_sql(output, sql, db_id, split="dev")
    return {
        "exact": int(result["reward"] == 1),
        "non_trivial": bool(result["non_trivial"]),
        "had_error": bool(result["model_error"] is not None),
    }


async def run_dev_scoring(
    sql_copilot: SQLCopilot,
    *,
    step: int | None = None,
    weave_name: str = "bird-dev-200",
) -> dict[str, Any]:
    """Score ``sql_copilot`` against the 200-row dev subset and return Weave's summary."""
    eval_obj = weave.Evaluation(
        name=weave_name,
        dataset=load_dev_subset(),
        scorers=[execution_match],
    )
    attrs = {"step": step} if step is not None else {}
    with weave.attributes(attrs):
        return await eval_obj.evaluate(sql_copilot)
