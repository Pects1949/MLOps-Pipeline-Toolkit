"""MLflow-based experiment tracking."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.entities import Run
from mlflow.tracking import MlflowClient


class ExperimentTracker:
    """Wraps MLflow to provide a concise experiment-tracking API.

    Usage::

        tracker = ExperimentTracker("fraud-detection", tracking_uri="http://mlflow:5000")

        with tracker.run("baseline") as run:
            tracker.log_params({"n_estimators": 100, "max_depth": 5})
            tracker.log_metrics({"accuracy": 0.94, "f1": 0.91})
            tracker.log_model(clf, "model")
    """

    def __init__(
        self,
        experiment_name: str = "default",
        tracking_uri: str = "mlruns",
    ) -> None:
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._client = MlflowClient()

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    @contextmanager
    def run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Generator[Run, None, None]:
        """Context manager that starts / ends an MLflow run."""
        with mlflow.start_run(run_name=run_name, tags=tags) as active_run:
            yield active_run

    # ------------------------------------------------------------------
    # Logging helpers (call inside a ``run()`` context)
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(local_path, artifact_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        flavor: str = "sklearn",
        registered_name: str | None = None,
    ) -> str:
        """Log *model* and optionally register it; returns the model URI."""
        log_fn = getattr(mlflow, flavor, mlflow.sklearn)
        info = log_fn.log_model(
            model,
            artifact_path,
            registered_model_name=registered_name,
        )
        return info.model_uri

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> Run:
        return self._client.get_run(run_id)

    def search_runs(
        self,
        filter_string: str = "",
        order_by: list[str] | None = None,
        max_results: int = 20,
    ):
        """Return a Pandas DataFrame of matching runs."""
        return mlflow.search_runs(
            experiment_names=[self.experiment_name],
            filter_string=filter_string,
            order_by=order_by or ["metrics.accuracy DESC"],
            max_results=max_results,
        )

    def best_run(self, metric: str = "metrics.accuracy", ascending: bool = False) -> Run:
        """Return the run with the best value for *metric*."""
        order = "ASC" if ascending else "DESC"
        runs = mlflow.search_runs(
            experiment_names=[self.experiment_name],
            order_by=[f"{metric} {order}"],
            max_results=1,
        )
        if runs.empty:
            raise ValueError(f"No runs found in experiment {self.experiment_name!r}")
        return self._client.get_run(runs.iloc[0]["run_id"])
