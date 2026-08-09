from __future__ import annotations

import stat
import tempfile
from pathlib import Path

import pytest

from src.automation.chatgpt_mcp.settings import (
    GatewaySettings,
    normalize_origin_url,
    normalize_public_base_url,
)


def _git_repo(path: Path) -> Path:
    path.mkdir()
    (path / ".git").mkdir()
    return path


def test_normalize_public_base_url_accepts_https_origin() -> None:
    assert normalize_public_base_url("https://example.test/") == "https://example.test"


@pytest.mark.parametrize(
    "value",
    [
        "http://example.test",
        "https://example.test/mcp",
        "https://example.test?query=1",
        "localhost:8765",
    ],
)
def test_normalize_public_base_url_rejects_non_origin(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_public_base_url(value)


def test_origin_is_fixed_to_tennis_lab() -> None:
    assert normalize_origin_url("https://github.com/Motoki0705/tennis-lab").endswith(
        ".git"
    )
    with pytest.raises(ValueError, match="Motoki0705/tennis-lab"):
        normalize_origin_url("https://github.com/other/repository.git")


def test_settings_reject_control_plane_inside_project(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    with pytest.raises(ValueError, match="control plane"):
        GatewaySettings(
            repo_root=repo,
            state_dir=tmp_path / "state",
            control_dir=repo / ".control",
            public_base_url=None,
        )


def test_ensure_state_creates_stable_private_runtime_state() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory)
        settings = GatewaySettings(
            repo_root=_git_repo(root / "repo"),
            state_dir=root / "state",
            control_dir=root / "control",
            public_base_url="https://example.test",
        )
        settings.ensure_state()
        first = settings.read_owner_secret()
        settings.ensure_state()
        second = settings.read_owner_secret()

        assert first == second
        assert len(first) >= 32
        assert stat.S_IMODE(settings.owner_secret_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(settings.state_dir.stat().st_mode) == 0o700
        assert settings.gpu_lock_file == Path("/var/lib/tennis-lab-actions/gpu.lock")
        assert stat.S_IMODE(settings.job_specs_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(settings.sandbox_jobs_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(settings.revision_workspace_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(settings.trusted_queue_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(settings.git_file_mask_path.stat().st_mode) == 0o400
        assert stat.S_IMODE(settings.git_dir_mask_path.stat().st_mode) == 0o500
        assert "intentionally unavailable" in settings.git_file_mask_path.read_text(
            encoding="utf-8"
        )
