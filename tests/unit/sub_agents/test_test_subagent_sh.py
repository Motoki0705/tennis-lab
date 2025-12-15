from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import src  # noqa: F401


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Repository root not found")


@pytest.mark.unit
def test_test_subagent_help_creates_local_cache_and_prints_usage(tmp_path: Path) -> None:
    repo_root = _repo_root()
    cache_dir = tmp_path / "cache"

    env = os.environ.copy()
    env["CODEX_LOCAL_CACHE_DIR"] = str(cache_dir)

    script = repo_root / "agents_workspace" / "sub_agents" / "test_subagent.sh"

    proc = subprocess.run(
        ["bash", str(script), "-h"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0
    assert "Usage:" in proc.stdout
    assert "Default test command" in proc.stdout

    assert (cache_dir / "xdg_cache").is_dir()
    assert (cache_dir / "uv_cache").is_dir()
    assert (cache_dir / "pre_commit_home").is_dir()
    assert (cache_dir / "fake_home" / ".codex").is_dir()
