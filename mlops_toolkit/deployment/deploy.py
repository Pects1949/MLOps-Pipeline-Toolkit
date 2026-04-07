"""Model deployment backends."""

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeploymentConfig:
    model_name: str
    model_version: str
    environment: str = "staging"
    replicas: int = 1
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "256Mi"
    memory_limit: str = "512Mi"
    env_vars: dict[str, str] = field(default_factory=dict)
    port: int = 8080


class BaseDeployer(ABC):
    """Contract that every deployer must satisfy."""

    @abstractmethod
    def deploy(self, config: DeploymentConfig) -> str:
        """Deploy a model; return an endpoint or resource URI."""

    @abstractmethod
    def undeploy(self, deployment_name: str) -> None:
        """Remove an existing deployment."""

    @abstractmethod
    def status(self, deployment_name: str) -> dict[str, Any]:
        """Return a dict describing the current state of a deployment."""


# ---------------------------------------------------------------------------
# Local deployer — MLflow model server
# ---------------------------------------------------------------------------

class LocalDeployer(BaseDeployer):
    """Serve a model locally with ``mlflow models serve``.

    Useful for smoke-testing before pushing to a real environment.
    """

    def __init__(self, host: str = "127.0.0.1", base_port: int = 5001) -> None:
        self.host = host
        self.base_port = base_port
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._ports: dict[str, int] = {}

    def deploy(self, config: DeploymentConfig) -> str:
        name = self._name(config)
        port = self.base_port + len(self._processes)
        model_uri = f"models:/{config.model_name}/{config.model_version}"
        cmd = [
            "mlflow", "models", "serve",
            "-m", model_uri,
            "-h", self.host,
            "-p", str(port),
            "--no-conda",
        ]
        env = {**os.environ, **config.env_vars}
        proc = subprocess.Popen(cmd, env=env, text=True)
        self._processes[name] = proc
        self._ports[name] = port
        endpoint = f"http://{self.host}:{port}/invocations"
        print(f"Deployment {name!r} started → {endpoint}  (pid {proc.pid})")
        return endpoint

    def undeploy(self, deployment_name: str) -> None:
        proc = self._processes.pop(deployment_name, None)
        if proc is None:
            raise KeyError(f"No active deployment: {deployment_name!r}")
        proc.terminate()
        self._ports.pop(deployment_name, None)
        print(f"Deployment {deployment_name!r} stopped")

    def status(self, deployment_name: str) -> dict[str, Any]:
        proc = self._processes.get(deployment_name)
        if proc is None:
            return {"name": deployment_name, "status": "not_found"}
        running = proc.poll() is None
        port = self._ports.get(deployment_name)
        return {
            "name": deployment_name,
            "status": "running" if running else "stopped",
            "pid": proc.pid,
            "endpoint": f"http://{self.host}:{port}/invocations" if port else None,
        }

    @staticmethod
    def _name(config: DeploymentConfig) -> str:
        return f"{config.model_name}-{config.model_version}-{config.environment}"
