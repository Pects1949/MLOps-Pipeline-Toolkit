"""Tests for mlops_toolkit.experiments.tracking."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mlops_toolkit.experiments.tracking import ExperimentTracker


@pytest.fixture()
def tracker(mlflow_tracking_uri: str) -> ExperimentTracker:
    return ExperimentTracker("test-experiment", tracking_uri=mlflow_tracking_uri)


class TestExperimentTracker:
    def test_run_context_manager_yields_run(self, tracker: ExperimentTracker) -> None:
        with tracker.run("my-run") as run:
            assert run is not None
            assert run.info.run_id

    def test_log_params_inside_run(self, tracker: ExperimentTracker) -> None:
        with tracker.run():
            tracker.log_params({"lr": 0.01, "epochs": 10})

    def test_log_metrics_inside_run(self, tracker: ExperimentTracker) -> None:
        with tracker.run():
            tracker.log_metrics({"accuracy": 0.95})
            tracker.log_metrics({"accuracy": 0.97}, step=1)

    def test_log_model_returns_uri(
        self, tracker: ExperimentTracker, tmp_path
    ) -> None:
        from sklearn.linear_model import LogisticRegression
        import numpy as np

        clf = LogisticRegression()
        clf.fit([[0], [1]], [0, 1])

        with tracker.run("model-run") as run:
            uri = tracker.log_model(clf, "model", flavor="sklearn")

        assert "model" in uri

    def test_search_runs_returns_dataframe(self, tracker: ExperimentTracker) -> None:
        with tracker.run("search-run"):
            tracker.log_metrics({"accuracy": 0.9})

        df = tracker.search_runs()
        assert not df.empty
        assert "run_id" in df.columns

    def test_best_run_raises_when_no_runs(self, mlflow_tracking_uri: str) -> None:
        empty_tracker = ExperimentTracker("empty-experiment", tracking_uri=mlflow_tracking_uri)
        with pytest.raises(ValueError, match="No runs found"):
            empty_tracker.best_run()

    def test_get_run_returns_correct_run(self, tracker: ExperimentTracker) -> None:
        with tracker.run("get-me") as run:
            run_id = run.info.run_id

        fetched = tracker.get_run(run_id)
        assert fetched.info.run_id == run_id
