"""End-to-end example: train → track → register → deploy locally.

Requires:
    pip install mlflow scikit-learn

Run:
    python examples/end_to_end.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from mlops_toolkit.deployment.deploy import DeploymentConfig, LocalDeployer
from mlops_toolkit.experiments.tracking import ExperimentTracker
from mlops_toolkit.registry.model_registry import ModelRegistry

TRACKING_URI = "mlruns"
EXPERIMENT = "wine-classification"
MODEL_NAME = "wine-rf"


def train() -> tuple[RandomForestClassifier, dict[str, float], str]:
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {"n_estimators": 120, "max_depth": 6, "random_state": 42}
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_macro": float(f1_score(y_test, preds, average="macro")),
    }
    return clf, metrics, params


def main() -> None:
    # 1. Train
    print("Training model …")
    clf, metrics, params = train()
    print(f"  accuracy={metrics['accuracy']:.4f}  f1={metrics['f1_macro']:.4f}")

    # 2. Track with MLflow
    tracker = ExperimentTracker(EXPERIMENT, tracking_uri=TRACKING_URI)
    with tracker.run("random-forest-baseline") as run:
        tracker.log_params(params)
        tracker.log_metrics(metrics)

        # Save a small feature-importance artefact
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            feature_names = load_wine().feature_names
            for name, imp in zip(feature_names, clf.feature_importances_):
                f.write(f"{name}: {imp:.4f}\n")
            artifact_path = f.name
        tracker.log_artifact(artifact_path, "feature_importance")

        model_uri = tracker.log_model(clf, "model", flavor="sklearn")
        run_id = run.info.run_id

    print(f"  Run logged: {run_id[:8]} | URI: {model_uri}")

    # 3. Register in Model Registry
    registry = ModelRegistry(tracking_uri=TRACKING_URI)
    mv = registry.register_from_run(run_id, "model", MODEL_NAME)
    print(f"  Registered: {mv.name} v{mv.version}")

    mv = registry.promote_to_staging(mv.name, mv.version)
    print(f"  Promoted to: {mv.stage}")

    # 4. Deploy locally (non-blocking print only — starts a server subprocess)
    config = DeploymentConfig(
        model_name=MODEL_NAME,
        model_version=mv.version,
        environment="staging",
    )
    deployer = LocalDeployer()
    print(f"\nTo serve the model locally, run:")
    print(f"  mlflow models serve -m models:/{MODEL_NAME}/Staging --no-conda")
    print(f"\nOr use the deployer programmatically:")
    print(f"  deployer = LocalDeployer()")
    print(f"  endpoint = deployer.deploy(DeploymentConfig('{MODEL_NAME}', '{mv.version}', 'staging'))")

    print("\nDone.")


if __name__ == "__main__":
    main()
