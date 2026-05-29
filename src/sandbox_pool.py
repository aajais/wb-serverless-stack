"""Pool of warm ``wandb.Sandbox`` instances with lazy per-DB artifact pulls.

At pool boot each sandbox gets ``wandb`` plus two runner scripts installed.
The .sqlite files are pulled on demand the first time a rollout hits a given
``db_id`` on a given sandbox (see ``ensure_db``); later rollouts on the same
(sandbox, db_id) pair hit the local cache. Pulling per-DB instead of the whole
split artifact keeps boot fast since most rollouts touch a few small DBs.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

from rich.console import Console

from src.config import settings

console = Console()

_HELPER_LOCAL = Path(__file__).parent / "sandbox_runtime" / "run_sql.py"
HELPER_REMOTE = "/opt/run_sql.py"
PULLER_REMOTE = "/opt/pull_db.py"

# Active pool. ``score_sql`` reads this to choose pool vs. local execution.
# Set/restored by SandboxPool.__aenter__/__aexit__ (nesting supported).
_POOL: SandboxPool | None = None


def current_pool() -> SandboxPool | None:
    return _POOL


def _render_puller_script() -> str:
    """Source of the in-sandbox puller that downloads a single DB entry.

    Resolves the split's aggregated artifact, then downloads exactly one entry
    via ``Artifact.get_entry(name).download(root=root)`` rather than the whole
    multi-GB blob. Prints ``pulled <path-to-sqlite>`` on success.

    Args (sys.argv):
      [1] fully qualified Registry path (e.g. wandb-registry-bird_ds/bird-train:latest)
      [2] artifact-relative entry name (e.g. california_schools/california_schools.sqlite)
      [3] destination directory inside the sandbox.
    """
    return (
        "import os, sys, traceback\n"
        "import wandb\n"
        "qualified = sys.argv[1]\n"
        "entry_name = sys.argv[2]\n"
        "root = sys.argv[3]\n"
        "entity = os.environ.get('WANDB_ENTITY')\n"
        "try:\n"
        "    os.makedirs(root, exist_ok=True)\n"
        "    api = wandb.Api(overrides={'entity': entity})\n"
        "    art = api.artifact(qualified)\n"
        "    entry = art.get_entry(entry_name)\n"
        "    path = entry.download(root=root)\n"
        "    print('pulled ' + str(path))\n"
        "except Exception as e:\n"
        "    print('pull-failed:', type(e).__name__, str(e), file=sys.stderr)\n"
        "    traceback.print_exc()\n"
        "    sys.exit(1)\n"
    )


class SandboxPool:
    """Lifecycle-managed pool of pre-warmed sandboxes with lazy DB pulls.

    Args:
        size: Number of sandboxes to keep warm. Bounds concurrent ``score_sql``
            calls. Use ``prompts_per_step * rollouts_per_prompt`` for training.
        split: ``"dev"`` or ``"train"``. Picks the Registry collection to pull
            per-DB artifacts from.
    """

    def __init__(self, size: int, split: str):
        if size < 1:
            raise ValueError(f"SandboxPool size must be >=1, got {size}")
        self.size = size
        self.split = split
        self.collection = settings.registry_collection_for(split)
        if not (settings.bird_registry_name and self.collection):
            raise RuntimeError(
                f"No Registry collection configured for split={split!r}. Set "
                f"BIRD_REGISTRY_NAME and BIRD_REGISTRY_{split.upper()}_COLLECTION "
                f"(or BIRD_REGISTRY_COLLECTION as a single-collection fallback) in .env."
            )
        self.mount_path = f"/data/bird/{split}"
        # Per-sandbox cache: sandbox_id → {db_id: absolute_path_inside_sandbox}.
        # Rollouts on the same (sandbox, db_id) skip the download.
        self._sandbox_db_cache: dict[str, dict[str, str]] = {}
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._stack: AsyncExitStack | None = None
        self._session: Any = None
        self._prev_pool: SandboxPool | None = None
        self._metric_defined: bool = False

    async def __aenter__(self) -> SandboxPool:
        global _POOL
        from wandb.sandbox import SandboxDefaults, Session  # type: ignore[attr-defined]

        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        # Pass W&B creds via env rather than writing the key into the sandbox FS.
        sandbox_env = {
            "WANDB_API_KEY": settings.wandb_api_key or "",
            "WANDB_ENTITY": settings.wandb_entity,
        }
        # max_lifetime_seconds must exceed the run's wall-clock time: the pool
        # holds every sandbox for the whole run, and the SDK default (None,
        # backend-controlled and typically short) would reap them mid-run →
        # grpc "Socket closed". Tune via SANDBOX_MAX_LIFETIME_SEC.
        defaults = SandboxDefaults(
            container_image=settings.sandbox_container_image,
            environment_variables=sandbox_env,
            max_lifetime_seconds=settings.sandbox_max_lifetime_sec,
        )
        # report_to=["wandb"] forces the wandb reporter to attach. The default
        # (None) auto-detects at construction time and silently disables
        # reporting when no wandb run is active yet — but ART creates its run
        # lazily. Forcing it makes log_metrics() bind to whatever run is active
        # at flush time.
        self._session = await self._stack.enter_async_context(
            Session(defaults=defaults, report_to=["wandb"])
        )
        console.log(
            f"[sandbox-pool] booting {self.size} sandbox(es) for split={self.split!r} "
            f"collection={self.collection!r} "
            f"(lifetime={settings.sandbox_max_lifetime_sec}s, "
            f"boot_concurrency={settings.sandbox_boot_concurrency}, "
            f"DBs lazy-pulled per rollout)"
        )

        helper_bytes = _HELPER_LOCAL.read_bytes()
        puller_bytes = _render_puller_script().encode("utf-8")
        # Bound concurrent boots; too many simultaneous pip installs + gRPC
        # streams trip transient UNAVAILABLE during warm-up.
        boot_sem = asyncio.Semaphore(max(1, settings.sandbox_boot_concurrency))

        async def _bootstrap_one(idx: int) -> Any:
            async with boot_sem:
                sb = self._session.sandbox()
                await asyncio.gather(
                    sb.write_file(HELPER_REMOTE, helper_bytes),
                    sb.write_file(PULLER_REMOTE, puller_bytes),
                )
                install = await sb.exec(["pip", "install", "-q", "wandb"])
                if install.returncode not in (0, None):
                    raise RuntimeError(
                        f"sandbox[{idx}] pip install wandb failed (rc={install.returncode}): "
                        f"{(install.stderr or '')[:500]}"
                    )
                return sb

        try:
            sandboxes = await asyncio.gather(*[_bootstrap_one(i) for i in range(self.size)])
        except Exception:
            # Tear the session down, but never let a cleanup error mask the
            # real bootstrap failure. The cwsandbox close path commonly raises
            # ``Failed to stop N sandbox(es)``, which would otherwise replace
            # the informative original exception.
            try:
                await self._stack.__aexit__(None, None, None)
            except Exception as cleanup_err:  # noqa: BLE001
                console.log(
                    f"[yellow][sandbox-pool] bootstrap cleanup error suppressed "
                    f"(original error re-raised below):[/] {str(cleanup_err)[:200]}"
                )
            finally:
                self._stack = None
                self._session = None
            raise

        console.log(f"[sandbox-pool] {self.size} sandbox(es) ready (no DBs downloaded yet)")

        for sb in sandboxes:
            await self._queue.put(sb)
        self._prev_pool = _POOL
        _POOL = self
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        global _POOL
        _POOL = self._prev_pool
        self._prev_pool = None
        if self._stack is not None:
            try:
                await self._stack.__aexit__(exc_type, exc, tb)
            except Exception as e:  # noqa: BLE001
                # The cwsandbox SDK races on session close (``Failed to stop N
                # sandbox(es)``) when pools are nested or the gRPC stream is torn
                # down mid-poll. Sandboxes are ephemeral and W&B GCs them server
                # side, so suppress this rather than masking the run summary.
                msg = str(e)
                if "Failed to stop" in msg or "sandbox_status" in msg:
                    console.log(
                        f"[yellow][sandbox-pool] non-fatal close error suppressed:[/] {msg[:200]}"
                    )
                else:
                    raise
            finally:
                self._stack = None
        self._session = None

    @asynccontextmanager
    async def checkout(self):
        """Borrow a sandbox, returned to the FIFO queue when the ``async with`` exits.

        FIFO cycling lets each sandbox warm its DB cache organically.
        """
        sb = await self._queue.get()
        try:
            yield sb
        finally:
            await self._queue.put(sb)

    async def ensure_db(self, sb: Any, db_id: str) -> str:
        """Return the absolute path of ``<db_id>.sqlite`` inside ``sb``, pulling once.

        Per-sandbox cache; a no-op when the DB is already present (``test -f``).
        Raises ``RuntimeError`` on pull failure so the rollout fails (reward 0)
        rather than swallowing the error.
        """
        sandbox_id = getattr(sb, "sandbox_id", id(sb))
        cache = self._sandbox_db_cache.setdefault(str(sandbox_id), {})
        if db_id in cache:
            return cache[db_id]

        qualified = settings.registry_dataset_artifact_path(self.split)
        if qualified is None:
            raise RuntimeError(
                f"Registry not configured for split={self.split!r}; cannot ensure {db_id!r}"
            )
        entry_name = settings.registry_db_entry_name(db_id)
        # entry.download(root=db_root) writes to <db_root>/<entry_name>,
        # preserving the in-artifact subdir.
        db_root = self.mount_path
        expected = f"{db_root}/{entry_name}"

        # Fast path: the DB is already on this sandbox's filesystem.
        check = await sb.exec(["test", "-f", expected])
        if check.returncode == 0:
            cache[db_id] = expected
            return expected

        pull = await sb.exec(["python", PULLER_REMOTE, qualified, entry_name, db_root])
        if pull.returncode not in (0, None):
            raise RuntimeError(
                f"ensure_db({db_id!r}) failed (rc={pull.returncode}) "
                f"qualified={qualified!r}\n"
                f"--- stdout ---\n{(pull.stdout or '')[-1500:]}\n"
                f"--- stderr ---\n{(pull.stderr or '')[-1500:]}"
            )

        # Prefer the path the puller printed (``pulled <abs path>``).
        path = expected
        for line in (pull.stdout or "").splitlines():
            line = line.strip()
            if line.startswith("pulled "):
                path = line.removeprefix("pulled ").strip()
                break
        cache[db_id] = path
        return path

    def log_metrics(self, step: int | None = None) -> bool:
        """Flush cwsandbox/* metrics onto ART's ``training_step`` x-axis."""
        if self._session is None:
            return False
        try:
            import wandb  # type: ignore[import-not-found]

            run = getattr(wandb, "run", None)
        except ImportError:
            run = None
        if run is not None and not self._metric_defined:
            # Pin cwsandbox/* to ART's global training_step x-axis so panels
            # line up. Only needs to happen once per run.
            run.define_metric("cwsandbox/*", step_metric="training_step")
            self._metric_defined = True
        if run is not None and step is not None:
            run.log({"training_step": step}, commit=False)
        return self._session.log_metrics(step=step)
