import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """A temporary directory that acts as a fake git+DVC repo root."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".dvc").mkdir()
    return tmp_path


@pytest.fixture()
def mlflow_tracking_uri(tmp_path: Path) -> str:
    """Point MLflow at a throwaway local directory."""
    uri = str(tmp_path / "mlruns")
    os.environ["MLFLOW_TRACKING_URI"] = uri
    return uri
