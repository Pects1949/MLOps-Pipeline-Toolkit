"""Model deployment backends."""

from __future__ import annotations

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
