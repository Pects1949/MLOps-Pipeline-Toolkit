"""Command-line interface for the MLOps Pipeline Toolkit."""

from __future__ import annotations

import click

from mlops_toolkit.cicd.pipeline import MLOpsPipeline, PipelineStep
from mlops_toolkit.data.versioning import DVCVersioning
from mlops_toolkit.deployment.deploy import DeploymentConfig, LocalDeployer
from mlops_toolkit.registry.model_registry import ModelRegistry


@click.group()
@click.version_option()
def cli() -> None:
    """MLOps Pipeline Toolkit — data versioning, experiment tracking, and deployment."""


# ---------------------------------------------------------------------------
# data sub-group
# ---------------------------------------------------------------------------

@cli.group()
def data() -> None:
    """DVC-backed data versioning commands."""


@data.command("add")
@click.argument("path")
@click.option("--repo", default=".", show_default=True)
def data_add(path: str, repo: str) -> None:
    """Track PATH with DVC."""
    dvc = DVCVersioning(repo)
    dvc_file = dvc.add(path)
    click.echo(f"Tracked {path!r} → {dvc_file}")


@data.command("push")
@click.option("--remote", default=None)
@click.option("--repo", default=".", show_default=True)
def data_push(remote: str | None, repo: str) -> None:
    """Push cached data to remote storage."""
    DVCVersioning(repo).push(remote)
    click.echo("Push complete.")


@data.command("pull")
@click.option("--remote", default=None)
@click.option("--repo", default=".", show_default=True)
def data_pull(remote: str | None, repo: str) -> None:
    """Pull data from remote storage."""
    DVCVersioning(repo).pull(remote)
    click.echo("Pull complete.")


@data.command("status")
@click.option("--repo", default=".", show_default=True)
def data_status(repo: str) -> None:
    """Show DVC pipeline status."""
    click.echo(DVCVersioning(repo).status())


# ---------------------------------------------------------------------------
# registry sub-group
# ---------------------------------------------------------------------------

@cli.group()
def registry() -> None:
    """MLflow Model Registry commands."""


@registry.command("list")
@click.option("--tracking-uri", default="mlruns", show_default=True)
def registry_list(tracking_uri: str) -> None:
    """List all registered models."""
    models = ModelRegistry(tracking_uri).list_models()
    if not models:
        click.echo("No registered models found.")
    for m in models:
        click.echo(f"  • {m}")


@registry.command("versions")
@click.argument("model_name")
@click.option("--stage", default=None)
@click.option("--tracking-uri", default="mlruns", show_default=True)
def registry_versions(model_name: str, stage: str | None, tracking_uri: str) -> None:
    """List versions of MODEL_NAME."""
    versions = ModelRegistry(tracking_uri).list_versions(model_name, stage)
    for v in versions:
        click.echo(f"  v{v.version}  stage={v.stage}  run={v.run_id[:8]}")


@registry.command("promote")
@click.argument("model_name")
@click.argument("version")
@click.argument("stage", type=click.Choice(["Staging", "Production", "Archived"]))
@click.option("--tracking-uri", default="mlruns", show_default=True)
def registry_promote(model_name: str, version: str, stage: str, tracking_uri: str) -> None:
    """Transition MODEL_NAME VERSION to STAGE."""
    mv = ModelRegistry(tracking_uri).transition(model_name, version, stage)
    click.echo(f"Promoted {mv.name} v{mv.version} → {mv.stage}")


# ---------------------------------------------------------------------------
# deploy sub-group
# ---------------------------------------------------------------------------

@cli.group()
def deploy() -> None:
    """Model deployment commands."""


@deploy.command("local")
@click.argument("model_name")
@click.argument("version")
@click.option("--port", default=5001, show_default=True)
@click.option("--env", default="staging", show_default=True)
def deploy_local(model_name: str, version: str, port: int, env: str) -> None:
    """Serve MODEL_NAME VERSION locally with MLflow."""
    config = DeploymentConfig(
        model_name=model_name, model_version=version, environment=env
    )
    deployer = LocalDeployer(base_port=port)
    endpoint = deployer.deploy(config)
    click.echo(f"Endpoint: {endpoint}")
    click.echo("Press Ctrl+C to stop.")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# pipeline sub-group
# ---------------------------------------------------------------------------

@cli.group()
def pipeline() -> None:
    """CI/CD pipeline commands."""


@pipeline.command("export-gha")
@click.option("--output", default=".github/workflows/mlops.yml", show_default=True)
def pipeline_export_gha(output: str) -> None:
    """Export a sample GitHub Actions workflow to OUTPUT."""
    p = (
        MLOpsPipeline("mlops-example")
        .add_step(PipelineStep("lint", "ruff check ."))
        .add_step(PipelineStep("test", "pytest tests/ -v", depends_on=["lint"]))
        .add_step(PipelineStep("train", "python examples/end_to_end.py", depends_on=["test"]))
    )
    out = p.export_github_actions(output)
    click.echo(f"Workflow written to {out}")


if __name__ == "__main__":
    cli()
