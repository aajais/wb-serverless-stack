"""Run the serverless-RL training driver INSIDE a single cwsandbox.

Uploads the repo + .env into one long-lived sandbox, installs the project, and
runs the driver there. The heavy GPU work (inference + GRPO step) still runs on
the W&B serverless-RL fleet; this only moves the lightweight orchestrator off
your laptop.

Fire-and-forget: ``submit`` launches a detached install+train job and returns in
seconds. The sandbox keeps running server-side up to ``max_lifetime_seconds``;
reattach later by id to follow logs or stop it.

Usage:
    # Submit (fire-and-forget). Anything after ``--`` is forwarded to the trainer.
    python scripts/run_in_sandbox.py
    python scripts/run_in_sandbox.py -- --data-source dev200 --max-steps 200 --eval-every 25

    # Reattach to a submitted run by sandbox id:
    python scripts/run_in_sandbox.py --logs <sandbox_id>
    python scripts/run_in_sandbox.py --stop <sandbox_id>
"""

from __future__ import annotations

import argparse
import io
import os
import shlex
import sys
import tarfile
import time
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from wandb.sandbox import Sandbox

REPO = Path(__file__).resolve().parent.parent
REMOTE_DIR = "/workspace/ft-sd-demo"
TARBALL_REMOTE = "/tmp/repo.tar.gz"
PULL_REMOTE = "/tmp/pull_data.py"
LOG_REMOTE = "/tmp/train.log"
EXIT_REMOTE = "/tmp/train.exit"

# Uploaded into the sandbox and run at bootstrap. Pull the split's DBs from
# the W&B Registry into ./data/bird/<split>
PULL_SCRIPT = b"""import sys
import wandb
from src.config import settings

split = sys.argv[1]
path = settings.registry_dataset_artifact_path(split)
if not path:
    raise SystemExit(f"no registry artifact path for split={split!r}")
dest = str(settings.bird_data_dir / split)
print(f"[pull_data] {path} -> {dest}", flush=True)
api = wandb.Api(overrides={"entity": settings.wandb_entity})
api.artifact(path).download(root=dest)
print("[pull_data] done", flush=True)
"""

# Default train args; override anything after ``--``.
DEFAULT_TRAIN_ARGS = ["--data-source", "dev200", "--max-steps", "100", "--eval-every", "25"]

# Never ship these into the sandbox (huge / irrelevant / regenerated).
EXCLUDE_PREFIXES = (
    ".venv",
    ".git",
    "out",
    "wandb",
    ".mypy_cache",
    ".pytest_cache",
    "data/bird",  # multi-GB sqlite blobs; pulled lazily from the Registry instead
)

# The driver sandbox lives for the whole job and spawns its own SandboxPool
# workers. Pinned to 4h to match the worker pool's SANDBOX_MAX_LIFETIME_SEC
# default (config.py) so both tear down on the same clock.
DRIVER_LIFETIME_SEC = 14400


def make_tarball() -> bytes:
    """Tar the repo (minus heavy/irrelevant dirs) into an in-memory .tar.gz."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:

        def _filter(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
            rel = ti.name[2:] if ti.name.startswith("./") else ti.name
            if "__pycache__" in rel or any(
                rel == p or rel.startswith(p + "/") or f"/{p}/" in rel for p in EXCLUDE_PREFIXES
            ):
                return None
            return ti

        tar.add(REPO, arcname=".", filter=_filter)
    return buf.getvalue()


def env_for_sandbox() -> dict[str, str]:
    """Forward .env values (+ live shell overrides) so config.py resolves."""
    env = {k: v for k, v in dotenv_values(REPO / ".env").items() if v is not None}
    for k in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    if not env.get("WANDB_API_KEY"):
        sys.exit("WANDB_API_KEY not found in .env or environment.")
    return env


def run(sb, cmd: str, *, cwd: str | None = None, timeout: float | None = None):
    """Blocking exec helper -> (returncode, stdout, stderr)."""
    proc = sb.exec(["bash", "-lc", cmd], cwd=cwd)
    res = proc.result(timeout=timeout)
    return (
        proc.returncode,
        res.stdout_bytes.decode("utf-8", "replace"),
        res.stderr_bytes.decode("utf-8", "replace"),
    )


def _split_for(train_args: list[str]) -> str:
    """Which BIRD split the driver needs locally for schema rendering.

    Mirrors train_serverless's data-source -> split mapping: everything except an
    explicit ``--data-source train`` reads from the dev split.
    """
    if "--data-source" in train_args:
        i = train_args.index("--data-source")
        if i + 1 < len(train_args) and train_args[i + 1] == "train":
            return "train"
    return "dev"


def submit(train_args: list[str]) -> int:
    """Boot a sandbox, upload the repo, kick off a detached install+train, exit.

    Uses ``Sandbox.run(...)`` rather than ``Session(...).sandbox()``. 
    A session-less ``Sandbox.run`` sandbox is not tracked by any session, 
    so it survives process exit and lives up to ``max_lifetime_seconds``. 
    Reattach by id via ``--logs``/``--stop``.
    """
    train_cmd = "python -u -m src.train_serverless " + " ".join(shlex.quote(a) for a in train_args)

    print("[sandbox] booting (detached, session-less)...")
    sb = Sandbox.run(
        container_image="python:3.11",
        environment_variables=env_for_sandbox(),
        max_lifetime_seconds=DRIVER_LIFETIME_SEC,
        tags=["driver", "serverless-rl"],
    ).wait()

    print("[sandbox] uploading repo tarball...")
    sb.write_file(TARBALL_REMOTE, make_tarball()).result()
    sb.write_file(PULL_REMOTE, PULL_SCRIPT).result()
    print(f"[sandbox] booted: {sb.sandbox_id}")

    # extract -> install -> pull DBs -> train runs detached (nohup) inside the
    # sandbox, so submit returns in seconds instead of blocking for the whole run.
    split = _split_for(train_args)
    bootstrap = (
        f"mkdir -p {REMOTE_DIR} && tar xzf {TARBALL_REMOTE} -C {REMOTE_DIR} && "
        f"cd {REMOTE_DIR} && pip install -q -e . && "
        f"python {PULL_REMOTE} {split} && {train_cmd}"
    )
    launch = (
        f"rm -f {EXIT_REMOTE} && "
        f"nohup bash -lc {shlex.quote(bootstrap + f'; echo $? > {EXIT_REMOTE}')} "
        f"> {LOG_REMOTE} 2>&1 & disown"
    )
    run(sb, launch)

    sid = sb.sandbox_id
    print(
        "\n[sandbox] training submitted (detached). This script is done.\n"
        f"  sandbox id : {sid}\n"
        f"  stream logs: python scripts/run_in_sandbox.py --logs {sid}\n"
        f"  stop it    : python scripts/run_in_sandbox.py --stop {sid}\n"
    )
    return 0


def logs(sandbox_id: str) -> int:
    """Reattach to a running sandbox and stream the training log until it exits."""
    sb = Sandbox.from_id(sandbox_id).result()
    offset = 0
    while True:
        _, chunk, _ = run(sb, f"tail -c +{offset + 1} {LOG_REMOTE} 2>/dev/null")
        if chunk:
            sys.stdout.write(chunk)
            sys.stdout.flush()
            offset += len(chunk.encode("utf-8", "replace"))
        done, code, _ = run(sb, f"cat {EXIT_REMOTE} 2>/dev/null")
        if done == 0 and code.strip():
            print("\n" + "-" * 60 + f"\n[sandbox] training exited with code {code.strip()}")
            return int(code.strip() or 1)
        time.sleep(5)


def stop(sandbox_id: str) -> int:
    """Stop (terminate) a running sandbox."""
    sb = Sandbox.from_id(sandbox_id).result()
    sb.stop().result()
    print(f"[sandbox] stopped {sandbox_id}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--logs", metavar="SANDBOX_ID", help="Stream logs from a submitted sandbox.")
    ap.add_argument("--stop", metavar="SANDBOX_ID", help="Stop a running sandbox.")
    ap.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Args after `--` forwarded to src.train_serverless (submit mode).",
    )
    args = ap.parse_args()

    # Export .env into this process so the cwsandbox client authenticates with
    # the project's WANDB_API_KEY (via wandb's auth resolver). 
    load_dotenv(REPO / ".env", override=True)

    if args.logs:
        return logs(args.logs)
    if args.stop:
        return stop(args.stop)

    train_args = args.train_args
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    return submit(train_args or DEFAULT_TRAIN_ARGS)


if __name__ == "__main__":
    raise SystemExit(main())
