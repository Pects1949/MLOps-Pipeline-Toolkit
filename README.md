# MLOps Pipeline Toolkit

A Python toolkit for building end-to-end MLOps pipelines covering data versioning, experiment tracking, model registry, CI/CD automation, and deployment.

## Features

| Layer | Technology | Module |
|---|---|---|
| Data versioning | DVC | `mlops_toolkit.data` |
| Experiment tracking | MLflow | `mlops_toolkit.experiments` |
| Model registry | MLflow Model Registry | `mlops_toolkit.registry` |
| CI/CD automation | GitHub Actions / GitLab CI | `mlops_toolkit.cicd` |
| Deployment | Local (MLflow serve) + Kubernetes | `mlops_toolkit.deployment` |

## Installation

```bash
pip install -e ".[dev,examples]"
```

## Quick start

```python
from mlops_toolkit.experiments import ExperimentTracker
from mlops_toolkit.registry import ModelRegistry
from mlops_toolkit.deployment import LocalDeployer, DeploymentConfig

# 1. Track an experiment
tracker = ExperimentTracker("my-project")
with tracker.run("baseline") as run:
    tracker.log_params({"n_estimators": 100})
    tracker.log_metrics({"accuracy": 0.94})
    tracker.log_model(clf, "model", flavor="sklearn")
    run_id = run.info.run_id

# 2. Register in model registry
registry = ModelRegistry()
mv = registry.register_from_run(run_id, "model", "my-classifier")
registry.promote_to_staging(mv.name, mv.version)

# 3. Deploy locally
deployer = LocalDeployer()
endpoint = deployer.deploy(
    DeploymentConfig("my-classifier", mv.version, environment="staging")
)
print(f"Model serving at {endpoint}")
```

## CLI

```bash
# Data versioning
mlops data add data/train.csv
mlops data push
mlops data pull --remote s3-remote

# Model registry
mlops registry list
mlops registry versions my-classifier
mlops registry promote my-classifier 1 Production

# Deployment
mlops deploy local my-classifier 1 --port 5001

# CI/CD export
mlops pipeline export-gha
```

## Data versioning (DVC)

```python
from mlops_toolkit.data import DVCVersioning

dvc = DVCVersioning(".")
dvc.add("data/train.csv")          # creates data/train.csv.dvc
dvc.push()                          # upload to remote

# Define a reproducible pipeline
dvc.create_stage(
    "preprocess",
    command="python src/preprocess.py",
    deps=["data/raw.csv", "src/preprocess.py"],
    outs=["data/processed.csv"],
)
dvc.reproduce()
```

## CI/CD pipeline

```python
from mlops_toolkit.cicd import MLOpsPipeline, PipelineStep

pipeline = (
    MLOpsPipeline("train-and-deploy")
    .add_step(PipelineStep("lint",  "ruff check ."))
    .add_step(PipelineStep("test",  "pytest tests/",    depends_on=["lint"]))
    .add_step(PipelineStep("train", "python train.py",  depends_on=["test"]))
    .add_step(PipelineStep("deploy","python deploy.py", depends_on=["train"]))
)

# Run locally
pipeline.run()

# Export to GitHub Actions
pipeline.export_github_actions(".github/workflows/mlops.yml")

# Export to GitLab CI
pipeline.export_gitlab_ci(".gitlab-ci.yml")
```

## Deployment

```python
from mlops_toolkit.deployment import (
    DeploymentConfig, LocalDeployer, KubernetesDeployer, DeploymentManager
)

config = DeploymentConfig("fraud-model", version="3", environment="staging", replicas=2)

manager = DeploymentManager()
manager.register("local", LocalDeployer())
manager.register("k8s",   KubernetesDeployer(namespace="mlops", registry="myregistry.io"))

# Deploy to staging locally
endpoint = manager.deploy("local", config)

# Promote to production on k8s
prod_config = DeploymentConfig("fraud-model", version="3", environment="production", replicas=3)
k8s_uri = manager.deploy("k8s", prod_config)
```

## Running tests

```bash
pytest tests/ -v --cov=mlops_toolkit
```

## Project structure

```
mlops_toolkit/
├── data/           # DVCVersioning — dataset tracking & pipeline stages
├── experiments/    # ExperimentTracker — MLflow runs, params, metrics
├── registry/       # ModelRegistry — register, promote, load versioned models
├── cicd/           # MLOpsPipeline — local runner + GHA/GitLab CI export
├── deployment/     # LocalDeployer, KubernetesDeployer, DeploymentManager
└── cli.py          # Click CLI entry-point (`mlops` command)
examples/
└── end_to_end.py   # Full train → track → register → deploy walkthrough
tests/
└── test_*.py       # pytest suite (one file per module)
.github/workflows/
└── mlops_ci.yml    # CI: lint → test matrix → train-and-register on main
```
