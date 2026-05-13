"""Tests for mlops_toolkit.registry.model_registry."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

from mlops_toolkit.experiments.tracking import ExperimentTracker
from mlops_toolkit.registry.model_registry import ModelRegistry, ModelVersion


@pytest.fixture()
def registered_model(mlflow_tracking_uri: str) -> tuple[ModelRegistry, ModelVersion]:
    """Train a tiny model, log it, and register it."""
    tracker = ExperimentTracker("registry-test", tracking_uri=mlflow_tracking_uri)
    clf = LogisticRegression()
    clf.fit([[0], [1]], [0, 1])

    with tracker.run("reg-run") as run:
        run_id = run.info.run_id
        mlflow.sklearn.log_model(clf, "model")

    registry = ModelRegistry(tracking_uri=mlflow_tracking_uri)
    mv = registry.register_from_run(run_id, "model", "test-model")
    return registry, mv


class TestModelRegistry:
    def test_register_creates_version(
        self, registered_model: tuple[ModelRegistry, ModelVersion]
    ) -> None:
        _, mv = registered_model
        assert mv.name == "test-model"
        assert mv.version == "1"

    def test_promote_to_staging(
        self, registered_model: tuple[ModelRegistry, ModelVersion]
    ) -> None:
        registry, mv = registered_model
        promoted = registry.promote_to_staging(mv.name, mv.version)
        assert promoted.stage == "Staging"

    def test_promote_to_production(
        self, registered_model: tuple[ModelRegistry, ModelVersion]
    ) -> None:
        registry, mv = registered_model
        registry.promote_to_staging(mv.name, mv.version)
        promoted = registry.promote_to_production(mv.name, mv.version)
        assert promoted.stage == "Production"

    def test_get_latest_returns_none_for_empty_stage(
        self, registered_model: tuple[ModelRegistry, ModelVersion]
    ) -> None:
        registry, mv = registered_model
        result = registry.get_latest(mv.name, stage="Production")
        assert result is None

    def test_list_models_includes_registered(
        self, registered_model: tuple[ModelRegistry, ModelVersion]
    ) -> None:
        registry, mv = registered_model
        assert mv.name in registry.list_models()

    def test_invalid_stage_raises(
        self, registered_model: tuple[ModelRegistry, ModelVersion]
    ) -> None:
        registry, mv = registered_model
        with pytest.raises(ValueError, match="stage must be one of"):
            registry.transition(mv.name, mv.version, "InvalidStage")

    def test_annotate_sets_description(
        self, registered_model: tuple[ModelRegistry, ModelVersion]
    ) -> None:
        registry, mv = registered_model
        registry.annotate(mv.name, mv.version, "baseline model")
        versions = registry.list_versions(mv.name)
        assert any(v.version == mv.version for v in versions)
