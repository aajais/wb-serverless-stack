"""Centralized config — loads .env once and exposes typed accessors."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env(key: str, default: str | None = None) -> str:
    val = os.environ.get(key, default)
    if val is None:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return val


@dataclass(frozen=True)
class Settings:
    # W&B
    wandb_entity: str = field(default_factory=lambda: _env("WANDB_ENTITY"))
    wandb_project: str = field(default_factory=lambda: _env("WANDB_PROJECT", "sql-copilot-bird"))
    wandb_training_url: str = field(
        default_factory=lambda: _env("WANDB_TRAINING_URL", "https://api.training.wandb.ai/v1")
    )
    wandb_api_key: str | None = field(default_factory=lambda: os.environ.get("WANDB_API_KEY"))

    # Run
    sql_model_name: str = field(default_factory=lambda: _env("SQL_MODEL_NAME", "sql-copilot-001"))
    base_model: str = field(
        default_factory=lambda: _env("BASE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    )
    train_max_steps: int = field(
        default_factory=lambda: int(os.environ.get("TRAIN_MAX_STEPS", 500))
    )
    eval_every_n_steps: int = field(
        default_factory=lambda: int(os.environ.get("EVAL_EVERY_N_STEPS", 25))
    )
    rollouts_per_prompt: int = field(
        default_factory=lambda: int(os.environ.get("ROLLOUTS_PER_PROMPT", 8))
    )
    prompts_per_step: int = field(
        default_factory=lambda: int(os.environ.get("PROMPTS_PER_STEP", 4))
    )
    learning_rate: float = field(
        default_factory=lambda: float(os.environ.get("LEARNING_RATE", 1e-5))
    )
    sampling_temperature: float = field(
        default_factory=lambda: float(os.environ.get("SAMPLING_TEMPERATURE", 1.0))
    )
    max_completion_tokens: int = field(
        default_factory=lambda: int(os.environ.get("MAX_COMPLETION_TOKENS", 512))
    )
    train_seed: int = field(default_factory=lambda: int(os.environ.get("TRAIN_SEED", 0)))
    data_source: str = field(default_factory=lambda: _env("DATA_SOURCE", "train"))

    # Sandbox
    sandbox_pool_size: int = field(
        default_factory=lambda: int(os.environ.get("SANDBOX_POOL_SIZE", 12))
    )
    sql_timeout_sec: int = field(default_factory=lambda: int(os.environ.get("SQL_TIMEOUT_SEC", 5)))
    sandbox_container_image: str = field(
        default_factory=lambda: _env("SANDBOX_CONTAINER_IMAGE", "python:3.11")
    )
    # Server-side lifetime for pooled sandboxes. Must exceed the run's
    # wall-clock time: the pool holds every sandbox for the whole run, and the
    # SDK default (None, typically short) would kill them mid-run → "Socket closed".
    sandbox_max_lifetime_sec: int = field(
        default_factory=lambda: int(os.environ.get("SANDBOX_MAX_LIFETIME_SEC", 14400))
    )
    # Cap concurrent boots; too many simultaneous pip installs + gRPC streams
    # trip transient UNAVAILABLE during warm-up.
    sandbox_boot_concurrency: int = field(
        default_factory=lambda: int(os.environ.get("SANDBOX_BOOT_CONCURRENCY", 8))
    )

    # W&B Registry holding BIRD DBs, pulled into sandboxes on demand. Split-aware
    # collections, with BIRD_REGISTRY_COLLECTION as a single-collection fallback.
    bird_registry_name: str | None = field(
        default_factory=lambda: os.environ.get("BIRD_REGISTRY_NAME") or None
    )
    bird_registry_collection: str | None = field(
        default_factory=lambda: os.environ.get("BIRD_REGISTRY_COLLECTION") or None
    )
    bird_registry_dev_collection: str | None = field(
        default_factory=lambda: os.environ.get("BIRD_REGISTRY_DEV_COLLECTION") or None
    )
    bird_registry_train_collection: str | None = field(
        default_factory=lambda: os.environ.get("BIRD_REGISTRY_TRAIN_COLLECTION") or None
    )
    bird_registry_version: str = field(
        default_factory=lambda: _env("BIRD_REGISTRY_VERSION", "latest")
    )

    # Weave
    weave_rollout_sample_rate: float = field(
        default_factory=lambda: float(os.environ.get("WEAVE_ROLLOUT_SAMPLE_RATE", 0.25))
    )

    # Data
    bird_data_dir: Path = field(
        default_factory=lambda: Path(_env("BIRD_DATA_DIR", str(REPO_ROOT / "data" / "bird")))
    )
    bird_dev_subset: Path = field(
        default_factory=lambda: Path(
            _env("BIRD_DEV_SUBSET", str(REPO_ROOT / "data" / "dev_200.jsonl"))
        )
    )

    # Eval / baseline endpoint. Used by scripts/make_baseline.py, smoke_eval.py,
    # and eval_bird_dev.py. ``eval_model`` falls back to ``base_model``.
    eval_endpoint: str = field(
        default_factory=lambda: _env("EVAL_ENDPOINT", "https://api.inference.wandb.ai/v1")
    )
    eval_model: str = field(
        default_factory=lambda: (
            os.environ.get("EVAL_MODEL")
            or os.environ.get("BASE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
        )
    )
    eval_api_key: str = field(
        default_factory=lambda: (
            os.environ.get("EVAL_API_KEY") or os.environ.get("WANDB_API_KEY", "EMPTY")
        )
    )
    eval_name: str = field(default_factory=lambda: _env("EVAL_NAME", "bird-dev-200-baseline"))
    smoke_eval_n: int = field(default_factory=lambda: int(os.environ.get("SMOKE_EVAL_N", 10)))
    baseline_out_path: Path = field(
        default_factory=lambda: Path(
            _env("BASELINE_OUT_PATH", str(REPO_ROOT / "out" / "baseline.json"))
        )
    )

    # vLLM / local-train fallback. ``local_model`` falls back to ``base_model``.
    vllm_base_url: str = field(
        default_factory=lambda: _env("VLLM_BASE_URL", "http://localhost:8000/v1")
    )
    vllm_api_key: str = field(default_factory=lambda: _env("VLLM_API_KEY", "EMPTY"))
    local_model: str = field(
        default_factory=lambda: (
            os.environ.get("LOCAL_MODEL")
            or os.environ.get("BASE_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
        )
    )
    train_local_max_steps: int = field(
        default_factory=lambda: int(os.environ.get("TRAIN_LOCAL_MAX_STEPS", 100))
    )

    @property
    def weave_project(self) -> str:
        return f"{self.wandb_entity}/{self.wandb_project}"

    def registry_collection_for(self, split: str) -> str | None:
        """Registry collection name for ``split`` ∈ {"dev","train"}, or None.

        Split-specific env vars win, falling back to ``BIRD_REGISTRY_COLLECTION``.
        """
        if split == "dev" and self.bird_registry_dev_collection:
            return self.bird_registry_dev_collection
        if split == "train" and self.bird_registry_train_collection:
            return self.bird_registry_train_collection
        return self.bird_registry_collection or None

    def registry_dataset_artifact_path(self, split: str) -> str | None:
        """Fully-qualified Registry path for the split's aggregated BIRD artifact.

        Returns ``wandb-registry-<name>/<collection>:<version>``, or None if
        Registry env vars aren't set. One artifact per split holds every DB as
        an entry (``<db_id>/<db_id>.sqlite``); callers pull individual entries
        via ``Artifact.get_entry(...).download(...)``.
        """
        collection = self.registry_collection_for(split)
        if not (self.bird_registry_name and collection):
            return None
        return f"wandb-registry-{self.bird_registry_name}/{collection}:{self.bird_registry_version}"

    @staticmethod
    def registry_db_entry_name(db_id: str) -> str:
        """Artifact-relative entry path for a given ``db_id`` inside the split artifact."""
        return f"{db_id}/{db_id}.sqlite"


settings = Settings()
