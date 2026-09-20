"""Executable analysis boundaries preserve arguments and fail before side effects."""

import subprocess
import sys
from pathlib import Path

import pytest

MODULES = (
    "evaluate_run",
)


def test_cli_roots_resolve_shared_worktree_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.slcs.scripts import _paths
    from src.utils.configuration import PathRole

    checkout = tmp_path / "worktree"
    checkout.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    for name in ("data", "ckpt", ".cache", "third_party", "outputs"):
        (shared / name).mkdir()
        (checkout / name).symlink_to(shared / name, target_is_directory=True)
    monkeypatch.setattr(_paths, "PROJECT_ROOT", checkout)
    resolver = _paths.cli_resolver(checkout / "outputs")
    assert resolver.roots.project_root == checkout
    assert resolver.roots.data_root == shared / "data"
    assert resolver.roots.checkpoint_root == shared / "ckpt"
    assert resolver.roots.artifact_root == checkout
    assert resolver.roots.output_root == shared / "outputs"
    assert resolver.roots.cache_root == shared / ".cache"
    assert resolver.roots.external_asset_root == shared / "third_party"
    assert resolver.resolve(PathRole.OUTPUT, "slcs/evaluate/example/run") == (
        shared / "outputs/slcs/evaluate/example/run"
    )


@pytest.mark.parametrize("name", MODULES)
def test_cpu_module_help(name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", f"src.tasks.slcs.scripts.{name}", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--output" in result.stdout
