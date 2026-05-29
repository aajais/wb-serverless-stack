"""Run a ~10-row Weave scoring pass to sanity-check the scoring harness."""

from __future__ import annotations

import argparse
import asyncio
import json

import weave
from rich.console import Console

from src.config import settings
from src.eval_bird_dev import SQLCopilot, execution_match, load_dev_subset

console = Console()


async def run(endpoint: str, model_name: str, n: int) -> None:
    weave.init(settings.weave_project)
    dataset = load_dev_subset()[:n]
    console.log(f"Scoring {len(dataset)} rows against {model_name} @ {endpoint}")

    copilot = SQLCopilot(endpoint=endpoint, model_name=model_name, temperature=0.0)
    evaluation = weave.Evaluation(name="bird-dev-smoke", dataset=dataset, scorers=[execution_match])
    summary = await evaluation.evaluate(copilot)
    console.print_json(json.dumps(summary, default=str))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=settings.eval_endpoint)
    ap.add_argument("--model", default=settings.eval_model)
    ap.add_argument("-n", type=int, default=settings.smoke_eval_n)
    args = ap.parse_args()
    asyncio.run(run(args.endpoint, args.model, args.n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
