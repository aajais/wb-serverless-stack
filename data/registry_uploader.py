"""Push the BIRD dataset to a W&B Registry collection and pull it back.

Putting BIRD in the Registry makes it a versioned artifact: one canonical
source across machines, versioned splits, and data → run → checkpoint lineage
in the W&B graph.

Workflow: ONE artifact per split (``bird-train``, ``bird-dev``) holding every
``*.sqlite`` as an internal entry at ``<db_id>/<db_id>.sqlite``. Sandboxes pull
a single entry on demand via ``Artifact.get_entry(name).download(root=...)``,
so they never fetch the multi-GB blob. One artifact also means one collection
version regardless of DB count (see ``src/sandbox_pool.py::SandboxPool.ensure_db``).

Usage:

    # Build & link the dev artifact (contains all dev DBs as entries).
    python -m data.registry_uploader upload-dataset \\
        --collection bird-dev \\
        --root ./data/bird/dev

    # Same for train.
    python -m data.registry_uploader upload-dataset \\
        --collection bird-train \\
        --root ./data/bird/train

    # Optional: pull the whole linked artifact into ./data/bird (debug only).
    python -m data.registry_uploader download \\
        --collection bird-dev \\
        --root ./data/bird/dev
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import wandb
from src.config import settings  # loads .env via python-dotenv

DEFAULT_REGISTRY = os.environ.get("BIRD_REGISTRY_NAME", "bird_ds")
DEFAULT_COLLECTION = os.environ.get("BIRD_REGISTRY_COLLECTION", "")
DEFAULT_UPLOAD_PATH = os.environ.get("BIRD_REGISTRY_UPLOAD_PATH", str(settings.bird_data_dir))
DEFAULT_ARTIFACT_TYPE = os.environ.get("BIRD_REGISTRY_ARTIFACT_TYPE", "dataset")
DEFAULT_DOWNLOAD_ROOT = os.environ.get("BIRD_REGISTRY_DOWNLOAD_ROOT", str(settings.bird_data_dir))


class RegistryUploader:
    """Wrapper around ``wandb.Artifact`` plus Registry linking.

    Args:
        registry_name: Registry name, without the ``wandb-registry-`` prefix.
        collection_name: Collection inside the Registry to link new artifact
            versions into.
    """

    def __init__(self, registry_name: str, collection_name: str) -> None:
        self.registry_name = registry_name
        self.collection_name = collection_name
        self.api = wandb.Api(api_key=os.environ.get("WANDB_API_KEY"))

    @property
    def _target_path(self) -> str:
        # Registry-linked artifacts live at wandb-registry-<name>/<collection>.
        return f"wandb-registry-{self.registry_name}/{self.collection_name}"

    def upload_dataset(
        self,
        root: str | Path,
        *,
        artifact_type: str = "dataset",
    ) -> str:
        """Aggregate every ``*.sqlite`` under ``root`` into one artifact and link it.

        Each .sqlite becomes an internal entry at ``<db_id>/<db_id>.sqlite`` so
        callers can pull one at a time via
        ``Artifact.get_entry(name).download(root=...)``. Each call bumps the
        collection's version (vN+1), leaving a single artifact holding all DBs
        for the split.

        Args:
            root: Directory to scan recursively for ``*.sqlite`` files.
            artifact_type: Type assigned to the artifact (must be in the
                collection's accepted types list).

        Returns:
            Qualified artifact path
            (``wandb-registry-<name>/<collection>:<version>``).
        """
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"root does not exist: {root_path}")

        sqlite_files = sorted(root_path.rglob("*.sqlite"))
        if not sqlite_files:
            raise RuntimeError(f"no *.sqlite files under {root_path}")

        total_bytes = sum(p.stat().st_size for p in sqlite_files)
        total_mb = total_bytes / 1024 / 1024
        print(
            f"[registry_uploader] aggregating {len(sqlite_files)} DB(s) "
            f"({total_mb:,.1f} MB) under {root_path} into "
            f"{self.collection_name!r}"
        )

        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            project=os.environ.get("WANDB_PROJECT", "sql-copilot-bird"),
            job_type="upload-dataset",
            name=f"upload-{self.collection_name}",
        )
        try:
            artifact = wandb.Artifact(
                self.collection_name,
                type=artifact_type,
                metadata={
                    "db_count": len(sqlite_files),
                    "total_bytes": total_bytes,
                    "db_ids": sorted({p.stem for p in sqlite_files}),
                },
            )
            for db_path in sqlite_files:
                db_id = db_path.stem
                # In-artifact path must match ``Settings.registry_db_entry_name``
                # so the sandbox puller and lineage helper resolve the same entry.
                entry_name = f"{db_id}/{db_id}.sqlite"
                artifact.add_file(str(db_path), name=entry_name)

            run.log_artifact(artifact)
            artifact.wait()  # block until W&B finalizes the version
            run.link_artifact(artifact, target_path=self._target_path)
            qualified = f"{self._target_path}:{artifact.version or 'latest'}"
            print(f"[registry_uploader] linked → {qualified}")
            return qualified
        finally:
            run.finish()

    def download(self, root: str | Path, *, version: str = "latest") -> Path:
        """Pull the linked artifact version into ``root`` and return that path."""
        qualified = f"{self._target_path}:{version}"
        artifact = self.api.artifact(qualified)
        out = Path(root)
        out.mkdir(parents=True, exist_ok=True)
        downloaded = artifact.download(root=str(out))
        print(f"[registry_uploader] downloaded {qualified} → {downloaded}")
        return Path(downloaded)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    ``--registry`` and ``--collection`` live on each subcommand, NOT the
    top-level parser, to avoid argparse's parent-vs-child default resolution
    footgun. Pass the flags AFTER the subcommand, e.g.:

        python -m data.registry_uploader upload-dataset --collection bird-dev --root ./data/bird/dev
    """

    def _add_shared(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--registry",
            default=DEFAULT_REGISTRY,
            help=f"Registry name (default: $BIRD_REGISTRY_NAME / {DEFAULT_REGISTRY!r})",
        )
        parser.add_argument(
            "--collection",
            default=DEFAULT_COLLECTION or None,
            help="Collection name inside the registry (default: $BIRD_REGISTRY_COLLECTION)",
        )

    p = argparse.ArgumentParser(prog="data.registry_uploader")
    sub = p.add_subparsers(dest="cmd", required=True)

    ud = sub.add_parser(
        "upload-dataset",
        help="Aggregate every *.sqlite under --root into ONE artifact, link to --collection",
    )
    _add_shared(ud)
    ud.add_argument(
        "--root",
        default=DEFAULT_UPLOAD_PATH,
        help="Directory to scan recursively for *.sqlite (default: $BIRD_REGISTRY_UPLOAD_PATH)",
    )
    ud.add_argument(
        "--type",
        default=DEFAULT_ARTIFACT_TYPE,
        help="Artifact type (default: $BIRD_REGISTRY_ARTIFACT_TYPE)",
    )

    dn = sub.add_parser("download", help="Download the linked artifact version into --root (debug)")
    _add_shared(dn)
    dn.add_argument(
        "--root",
        default=DEFAULT_DOWNLOAD_ROOT,
        help="Destination directory (default: $BIRD_REGISTRY_DOWNLOAD_ROOT)",
    )
    dn.add_argument(
        "--version",
        default=settings.bird_registry_version,
        help="Artifact version (default: $BIRD_REGISTRY_VERSION)",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.collection:
        raise SystemExit(
            "error: --collection is required (or set $BIRD_REGISTRY_COLLECTION in .env)"
        )
    uploader = RegistryUploader(args.registry, args.collection)
    if args.cmd == "upload-dataset":
        uploader.upload_dataset(args.root, artifact_type=args.type)
    elif args.cmd == "download":
        uploader.download(args.root, version=args.version)
    else:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
