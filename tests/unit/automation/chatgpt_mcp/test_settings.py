from __future__ import annotations

import stat
import tempfile
from pathlib import Path

import pytest

from src.automation.chatgpt_mcp.settings import (
    GatewaySettings,
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


def test_ensure_state_creates_stable_private_runtime_state() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        root = Path(directory)
        settings = GatewaySettings(
            repo_root=_git_repo(root / "repo"),
            state_dir=root / "state",
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
        assert stat.S_IMODE(settings.job_specs_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(settings.sandbox_jobs_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(settings.revision_workspace_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(settings.git_mask_path.stat().st_mode) == 0o400
        assert "intentionally unavailable" in settings.git_mask_path.read_text(
            encoding="utf-8"
        )
