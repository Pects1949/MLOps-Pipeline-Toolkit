"""DVC-backed data versioning."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml


class DVCVersioning:
    """Thin wrapper around DVC for dataset and pipeline versioning.

    Initialises DVC in *repo_path* on first use if not already present.
    All methods raise ``RuntimeError`` when the underlying DVC command
    exits with a non-zero status.
    """

    def __init__(self, repo_path: str = ".") -> None:
        self.repo_path = Path(repo_path).resolve()
        self._ensure_initialized()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run(self, *args: str) -> str:
        """Run a DVC command and return stdout, raising on failure."""
        result = subprocess.run(
            ["dvc", *args],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"dvc {' '.join(args)} failed:\n{result.stderr.strip()}"
            )
        return result.stdout.strip()

    def _ensure_initialized(self) -> None:
        if not (self.repo_path / ".dvc").exists():
            self._run("init")

    # ------------------------------------------------------------------
    # Dataset tracking
    # ------------------------------------------------------------------

    def add(self, path: str) -> Path:
        """Track *path* with DVC; returns the generated .dvc file path."""
        self._run("add", path)
        dvc_file = Path(path).with_suffix(Path(path).suffix + ".dvc")
        return dvc_file

    def push(self, remote: str | None = None) -> None:
        """Push cached files to the configured (or named) remote."""
        args = ["push"]
        if remote:
            args += ["-r", remote]
        self._run(*args)

    def pull(self, remote: str | None = None) -> None:
        """Pull cached files from the configured (or named) remote."""
        args = ["pull"]
        if remote:
            args += ["-r", remote]
        self._run(*args)

    def add_remote(self, name: str, url: str) -> None:
        """Register a remote storage location."""
        self._run("remote", "add", name, url)

    # ------------------------------------------------------------------
    # Pipeline management
    # ------------------------------------------------------------------

    def create_stage(
        self,
        name: str,
        command: str,
        deps: list[str] | None = None,
        outs: list[str] | None = None,
        params: list[str] | None = None,
    ) -> None:
        """Add a stage to dvc.yaml (creates the file if absent)."""
        dvc_yaml = self.repo_path / "dvc.yaml"
        config: dict[str, Any] = {"stages": {}}
        if dvc_yaml.exists():
            with dvc_yaml.open() as fh:
                config = yaml.safe_load(fh) or config

        config["stages"][name] = {
            "cmd": command,
            **({"deps": deps} if deps else {}),
            **({"outs": outs} if outs else {}),
            **({"params": params} if params else {}),
        }

        with dvc_yaml.open("w") as fh:
            yaml.dump(config, fh, default_flow_style=False, sort_keys=False)

    def reproduce(self, *targets: str) -> None:
        """Re-run any stages whose dependencies have changed."""
        args = ["repro", *targets]
        self._run(*args)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_hash(self, path: str) -> str:
        """Return the DVC-tracked MD5 hash for *path*."""
        dvc_file = Path(f"{path}.dvc")
        if not dvc_file.exists():
            raise FileNotFoundError(f"No DVC file found for {path!r}")
        with dvc_file.open() as fh:
            data = yaml.safe_load(fh)
        outs = data.get("outs", [{}])
        return outs[0].get("md5") or outs[0].get("hash", "unknown")

    def status(self) -> str:
        """Return the raw output of ``dvc status``."""
        return self._run("status")
