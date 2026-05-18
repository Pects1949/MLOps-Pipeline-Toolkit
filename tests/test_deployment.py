"""Tests for mlops_toolkit.deployment.deploy."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlops_toolkit.deployment.deploy import (
    DeploymentConfig,
    DeploymentManager,
    KubernetesDeployer,
    LocalDeployer,
)


@pytest.fixture()
def config() -> DeploymentConfig:
    return DeploymentConfig(
        model_name="fraud-model",
        model_version="3",
        environment="staging",
        env_vars={"LOG_LEVEL": "INFO"},
    )


class TestLocalDeployer:
    def test_deploy_starts_process(self, config: DeploymentConfig) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None

        with patch("mlops_toolkit.deployment.deploy.subprocess.Popen", return_value=mock_proc):
            deployer = LocalDeployer(base_port=9100)
            endpoint = deployer.deploy(config)

        assert "9100" in endpoint
        assert "/invocations" in endpoint

    def test_status_running(self, config: DeploymentConfig) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 42
        mock_proc.poll.return_value = None

        with patch("mlops_toolkit.deployment.deploy.subprocess.Popen", return_value=mock_proc):
            deployer = LocalDeployer(base_port=9200)
            deployer.deploy(config)
            name = f"{config.model_name}-{config.model_version}-{config.environment}"
            s = deployer.status(name)

        assert s["status"] == "running"
        assert s["pid"] == 42

    def test_undeploy_terminates_process(self, config: DeploymentConfig) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with patch("mlops_toolkit.deployment.deploy.subprocess.Popen", return_value=mock_proc):
            deployer = LocalDeployer(base_port=9300)
            deployer.deploy(config)
            name = f"{config.model_name}-{config.model_version}-{config.environment}"
            deployer.undeploy(name)

        mock_proc.terminate.assert_called_once()

    def test_undeploy_unknown_raises(self) -> None:
        deployer = LocalDeployer()
        with pytest.raises(KeyError):
            deployer.undeploy("nonexistent")

    def test_status_not_found_for_unknown(self) -> None:
        deployer = LocalDeployer()
        s = deployer.status("ghost-deployment")
        assert s["status"] == "not_found"


class TestKubernetesDeployer:
    def test_deploy_calls_kubectl_apply(self, config: DeploymentConfig, tmp_path: Path) -> None:
        with patch("mlops_toolkit.deployment.deploy.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="applied")
            deployer = KubernetesDeployer(namespace="test-ns")
            uri = deployer.deploy(config)

        assert "k8s://" in uri
        assert "test-ns" in uri
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "kubectl"
        assert "apply" in cmd

    def test_manifest_structure(self, config: DeploymentConfig) -> None:
        deployer = KubernetesDeployer(namespace="test-ns", registry="myregistry.io")
        manifest = deployer._build_manifest(config)

        assert manifest["kind"] == "Deployment"
        assert manifest["metadata"]["namespace"] == "test-ns"
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        assert "myregistry.io/fraud-model:3" == container["image"]
        assert container["resources"]["requests"]["cpu"] == config.cpu_request

    def test_undeploy_calls_kubectl_delete(self) -> None:
        with patch("mlops_toolkit.deployment.deploy.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="deleted")
            deployer = KubernetesDeployer()
            deployer.undeploy("fraud-model-staging")

        cmd = mock_run.call_args[0][0]
        assert "delete" in cmd
        assert "fraud-model-staging" in cmd

    def test_status_parses_kubectl_output(self) -> None:
        fake_output = json.dumps({
            "status": {"replicas": 2, "readyReplicas": 2, "availableReplicas": 2}
        })
        with patch("mlops_toolkit.deployment.deploy.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=fake_output)
            deployer = KubernetesDeployer()
            s = deployer.status("fraud-model-staging")

        assert s["ready"] == 2
        assert s["available"] == 2


class TestDeploymentManager:
    def test_dispatch_to_correct_backend(self, config: DeploymentConfig) -> None:
        mock_deployer = MagicMock()
        mock_deployer.deploy.return_value = "http://localhost:5001/invocations"

        manager = DeploymentManager()
        manager.register("mock", mock_deployer)
        endpoint = manager.deploy("mock", config)

        mock_deployer.deploy.assert_called_once_with(config)
        assert endpoint == "http://localhost:5001/invocations"

    def test_unknown_backend_raises(self, config: DeploymentConfig) -> None:
        manager = DeploymentManager()
        with pytest.raises(KeyError, match="Unknown backend"):
            manager.deploy("nonexistent", config)
