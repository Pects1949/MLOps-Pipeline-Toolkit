"""Tests for mlops_toolkit.cicd.pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from mlops_toolkit.cicd.pipeline import MLOpsPipeline, PipelineStep


@pytest.fixture()
def simple_pipeline() -> MLOpsPipeline:
    return (
        MLOpsPipeline("test-pipeline")
        .add_step(PipelineStep("lint", "echo lint"))
        .add_step(PipelineStep("test", "echo test", depends_on=["lint"]))
        .add_step(PipelineStep("train", "echo train", depends_on=["test"]))
    )


class TestMLOpsPipeline:
    def test_dry_run_completes_without_executing(
        self, simple_pipeline: MLOpsPipeline, capsys
    ) -> None:
        results = simple_pipeline.run(dry_run=True)
        assert all(results.values())
        out = capsys.readouterr().out
        assert "DRY RUN" in out

    def test_real_run_executes_steps(self, simple_pipeline: MLOpsPipeline) -> None:
        results = simple_pipeline.run()
        assert results == {"lint": True, "test": True, "train": True}

    def test_dependency_order_enforced(self) -> None:
        p = (
            MLOpsPipeline("dep-test")
            .add_step(PipelineStep("b", "echo b", depends_on=["a"]))
        )
        with pytest.raises(RuntimeError, match="dependency"):
            p.run()

    def test_continue_on_error_does_not_raise(self) -> None:
        p = (
            MLOpsPipeline("err-pipeline")
            .add_step(PipelineStep("fail", "exit 1", continue_on_error=True))
        )
        results = p.run()
        assert results["fail"] is False

    def test_failing_step_raises_by_default(self) -> None:
        p = MLOpsPipeline("fail-pipeline").add_step(PipelineStep("oops", "exit 1"))
        with pytest.raises(RuntimeError, match="failed at step"):
            p.run()

    def test_export_github_actions_creates_file(
        self, simple_pipeline: MLOpsPipeline, tmp_path: Path
    ) -> None:
        out = simple_pipeline.export_github_actions(
            str(tmp_path / ".github/workflows/mlops.yml")
        )
        assert out.exists()
        workflow = yaml.safe_load(out.read_text())
        assert workflow["name"] == "test-pipeline"
        step_names = [s["name"] for s in workflow["jobs"]["pipeline"]["steps"]]
        assert "lint" in step_names

    def test_export_gitlab_ci_creates_file(
        self, simple_pipeline: MLOpsPipeline, tmp_path: Path
    ) -> None:
        out = simple_pipeline.export_gitlab_ci(str(tmp_path / ".gitlab-ci.yml"))
        assert out.exists()
        config = yaml.safe_load(out.read_text())
        assert "lint" in config
        assert "test" in config
        assert config["test"]["needs"] == ["lint"]
