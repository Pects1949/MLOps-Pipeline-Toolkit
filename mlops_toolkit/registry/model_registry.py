"""MLflow Model Registry management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

VALID_STAGES = ("Staging", "Production", "Archived", "None")


@dataclass
class ModelVersion:
    name: str
    version: str
    stage: str
    run_id: str
    description: str = ""


class ModelRegistry:
    """High-level interface to the MLflow Model Registry.

    Handles registration, stage transitions, and loading of versioned
    models without callers needing to build model URIs manually.

    Usage::

        registry = ModelRegistry()
        mv = registry.register("runs:/abc123/model", "fraud-classifier")
        registry.transition(mv.name, mv.version, "Staging")
        model = registry.load("fraud-classifier", stage="Staging")
    """

    def __init__(self, tracking_uri: str = "mlruns") -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, model_uri: str, name: str) -> ModelVersion:
        """Register a model artifact under *name* and return its version."""
        mv = mlflow.register_model(model_uri, name)
        return self._to_model_version(mv)

    def register_from_run(
        self, run_id: str, artifact_path: str, name: str
    ) -> ModelVersion:
        """Convenience wrapper: build the URI from a run ID and artifact path."""
        uri = f"runs:/{run_id}/{artifact_path}"
        return self.register(uri, name)

    # ------------------------------------------------------------------
    # Stage transitions
    # ------------------------------------------------------------------

    def transition(
        self,
        name: str,
        version: str,
        stage: str,
        archive_existing: bool = True,
    ) -> ModelVersion:
        """Move *version* of *name* to *stage*, archiving previous versions."""
        if stage not in VALID_STAGES:
            raise ValueError(f"stage must be one of {VALID_STAGES}, got {stage!r}")
        mv = self._client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )
        return self._to_model_version(mv)

    def promote_to_staging(self, name: str, version: str) -> ModelVersion:
        return self.transition(name, version, "Staging")

    def promote_to_production(self, name: str, version: str) -> ModelVersion:
        return self.transition(name, version, "Production")

    def archive(self, name: str, version: str) -> ModelVersion:
        return self.transition(name, version, "Archived", archive_existing=False)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_latest(self, name: str, stage: str = "Production") -> ModelVersion | None:
        """Return the latest version in *stage*, or None if none exists."""
        versions = self._client.get_latest_versions(name, stages=[stage])
        return self._to_model_version(versions[0]) if versions else None

    def list_models(self) -> list[str]:
        return [m.name for m in self._client.search_registered_models()]

    def list_versions(self, name: str, stage: str | None = None) -> list[ModelVersion]:
        stages = [stage] if stage else list(VALID_STAGES)
        versions = self._client.get_latest_versions(name, stages=stages)
        return [self._to_model_version(v) for v in versions]

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self, name: str, stage: str = "Production") -> Any:
        """Load the latest *stage* version as a ``mlflow.pyfunc.PyFuncModel``."""
        uri = f"models:/{name}/{stage}"
        return mlflow.pyfunc.load_model(uri)

    def load_version(self, name: str, version: str) -> Any:
        uri = f"models:/{name}/{version}"
        return mlflow.pyfunc.load_model(uri)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def annotate(self, name: str, version: str, description: str) -> None:
        self._client.update_model_version(
            name=name, version=version, description=description
        )

    def delete_version(self, name: str, version: str) -> None:
        self._client.delete_model_version(name=name, version=version)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_model_version(mv: Any) -> ModelVersion:
        return ModelVersion(
            name=mv.name,
            version=mv.version,
            stage=mv.current_stage,
            run_id=mv.run_id or "",
            description=mv.description or "",
        )
