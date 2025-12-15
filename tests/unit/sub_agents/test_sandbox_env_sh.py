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
def test_codex_sandbox_env_setup_sets_env_and_creates_dirs(tmp_path: Path) -> None:
    repo_root = _repo_root()
    cache_dir = tmp_path / "cache"

    env = os.environ.copy()
    env["CODEX_LOCAL_CACHE_DIR"] = str(cache_dir)

    sandbox_env_sh = repo_root / "agents_workspace" / "sub_agents" / "sandbox_env.sh"

    subprocess.run(
        [
            "bash",
            "-lc",
            f"source {sandbox_env_sh!s} && codex_sandbox_env_setup",
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )

    assert cache_dir.is_dir()
    assert (cache_dir / "xdg_cache").is_dir()
    assert (cache_dir / "uv_cache").is_dir()
    assert (cache_dir / "pre_commit_home").is_dir()
    assert (cache_dir / "fake_home").is_dir()
    assert (cache_dir / "fake_home" / ".codex").is_dir()
