"""CI/CD pipeline orchestration and workflow generation."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PipelineStep:
    """A single executable step in the pipeline."""

    name: str
    command: str
    env: dict[str, str] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    continue_on_error: bool = False


class MLOpsPipeline:
    """Define, run, and export MLOps CI/CD pipelines.

    Steps are executed in definition order; ``depends_on`` is validated
    before each step so mis-ordered or missing dependencies are caught
    early.  The same pipeline can be serialised to GitHub Actions or
    GitLab CI YAML so local development and cloud automation stay in sync.

    Usage::

        pipeline = (
            MLOpsPipeline("train-and-deploy")
            .add_step(PipelineStep("test", "pytest tests/"))
            .add_step(PipelineStep("train", "python train.py", depends_on=["test"]))
            .add_step(PipelineStep("deploy", "python deploy.py", depends_on=["train"]))
        )

        pipeline.run()
        pipeline.export_github_actions(".github/workflows/mlops.yml")
    """

    def __init__(self, name: str = "mlops-pipeline") -> None:
        self.name = name
        self.steps: list[PipelineStep] = []

    def add_step(self, step: PipelineStep) -> MLOpsPipeline:
        self.steps.append(step)
        return self

    # ------------------------------------------------------------------
    # Local execution
    # ------------------------------------------------------------------

    def run(self, dry_run: bool = False) -> dict[str, bool]:
        """Execute each step in order, honouring ``depends_on``.

        Returns a mapping of step name → success.  Raises ``RuntimeError``
        on the first failing step (unless ``continue_on_error`` is set).
        """
        completed: dict[str, bool] = {}

        for step in self.steps:
            for dep in step.depends_on:
                if not completed.get(dep):
                    raise RuntimeError(
                        f"Step {step.name!r} requires {dep!r} which has not completed"
                    )

            if dry_run:
                print(f"[DRY RUN] {step.name}: {step.command}")
                completed[step.name] = True
                continue

            print(f"▶ {step.name}")
            env = {**os.environ, **step.env}
            result = subprocess.run(
                step.command, shell=True, env=env, text=True
            )
            ok = result.returncode == 0
            completed[step.name] = ok

            if not ok:
                print(f"  ✗ {step.name} exited with code {result.returncode}")
                if not step.continue_on_error:
                    raise RuntimeError(f"Pipeline failed at step {step.name!r}")
            else:
                print(f"  ✓ {step.name}")

        return completed

    # ------------------------------------------------------------------
    # GitHub Actions export
    # ------------------------------------------------------------------

    def export_github_actions(
        self,
        output_path: str = ".github/workflows/mlops.yml",
        python_version: str = "3.11",
        on_push_branches: list[str] | None = None,
    ) -> Path:
        """Write a GitHub Actions workflow file and return its path."""
        on_branches = on_push_branches or ["main"]

        workflow: dict[str, Any] = {
            "name": self.name,
            "on": {
                "push": {"branches": on_branches},
                "pull_request": {"branches": on_branches},
            },
            "jobs": {
                "pipeline": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v5",
                            "with": {"python-version": python_version},
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt",
                        },
                    ],
                }
            },
        }

        for step in self.steps:
            job_step: dict[str, Any] = {"name": step.name, "run": step.command}
            if step.env:
                job_step["env"] = step.env
            if step.continue_on_error:
                job_step["continue-on-error"] = True
            workflow["jobs"]["pipeline"]["steps"].append(job_step)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            yaml.dump(workflow, fh, default_flow_style=False, sort_keys=False)

        return out

    # ------------------------------------------------------------------
    # GitLab CI export
    # ------------------------------------------------------------------

    def export_gitlab_ci(self, output_path: str = ".gitlab-ci.yml") -> Path:
        """Write a GitLab CI pipeline file and return its path."""
        config: dict[str, Any] = {
            "stages": [s.name for s in self.steps],
            "default": {
                "image": "python:3.11",
                "before_script": ["pip install -r requirements.txt"],
            },
        }

        for step in self.steps:
            job: dict[str, Any] = {
                "stage": step.name,
                "script": [step.command],
            }
            if step.depends_on:
                job["needs"] = step.depends_on
            if step.env:
                job["variables"] = step.env
            config[step.name] = job

        out = Path(output_path)
        with out.open("w") as fh:
            yaml.dump(config, fh, default_flow_style=False, sort_keys=False)

        return out
