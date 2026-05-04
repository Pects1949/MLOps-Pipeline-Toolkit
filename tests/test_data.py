"""Tests for mlops_toolkit.data.versioning."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from mlops_toolkit.data.versioning import DVCVersioning


@pytest.fixture()
def dvc(tmp_repo: Path) -> DVCVersioning:
    with patch("mlops_toolkit.data.versioning.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        return DVCVersioning(str(tmp_repo))


class TestDVCVersioning:
    def test_init_skips_init_when_dvc_dir_exists(self, tmp_repo: Path) -> None:
        with patch("mlops_toolkit.data.versioning.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            DVCVersioning(str(tmp_repo))
            # .dvc already exists → dvc init should NOT be called
            mock_run.assert_not_called()

    def test_add_returns_dvc_file_path(self, dvc: DVCVersioning) -> None:
        with patch("mlops_toolkit.data.versioning.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = dvc.add("data/train.csv")
        assert str(result) == "data/train.csv.dvc"
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ["dvc", "add", "data/train.csv"]

    def test_push_without_remote(self, dvc: DVCVersioning) -> None:
        with patch("mlops_toolkit.data.versioning.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            dvc.push()
        assert mock_run.call_args[0][0] == ["dvc", "push"]

    def test_push_with_remote(self, dvc: DVCVersioning) -> None:
        with patch("mlops_toolkit.data.versioning.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            dvc.push("s3-remote")
        assert mock_run.call_args[0][0] == ["dvc", "push", "-r", "s3-remote"]

    def test_run_raises_on_nonzero_exit(self, dvc: DVCVersioning) -> None:
        with patch("mlops_toolkit.data.versioning.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error!")
            with pytest.raises(RuntimeError, match="failed"):
                dvc.pull()

    def test_create_stage_writes_dvc_yaml(self, dvc: DVCVersioning, tmp_repo: Path) -> None:
        dvc.create_stage(
            "preprocess",
            command="python preprocess.py",
            deps=["data/raw.csv"],
            outs=["data/processed.csv"],
        )
        dvc_yaml = tmp_repo / "dvc.yaml"
        assert dvc_yaml.exists()
        import yaml
        config = yaml.safe_load(dvc_yaml.read_text())
        assert "preprocess" in config["stages"]
        assert config["stages"]["preprocess"]["cmd"] == "python preprocess.py"

    def test_get_hash_reads_dvc_file(self, dvc: DVCVersioning, tmp_repo: Path) -> None:
        import yaml
        dvc_content = {"outs": [{"md5": "abc123def456", "path": "data/train.csv"}]}
        (tmp_repo / "data").mkdir()
        (tmp_repo / "data" / "train.csv.dvc").write_text(yaml.dump(dvc_content))
        h = dvc.get_hash(str(tmp_repo / "data" / "train.csv"))
        assert h == "abc123def456"

    def test_get_hash_raises_for_untracked_file(self, dvc: DVCVersioning) -> None:
        with pytest.raises(FileNotFoundError):
            dvc.get_hash("/nonexistent/data.csv")
