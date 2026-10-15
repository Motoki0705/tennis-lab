from __future__ import annotations

import subprocess
from pathlib import Path
from typing import cast

import pytest

from src.tasks.base.annotation.paths import AnnotationLayout, Area


def test_worktree_and_main_resolve_same_root(tmp_path: Path) -> None:
    main = tmp_path / "main"
    subprocess.run(["git", "init", "-q", str(main)], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.test",
            "commit",
            "--allow-empty",
            "-qm",
            "base",
        ],
        cwd=main,
        check=True,
    )
    worktree = main / ".claude/worktrees/test"
    subprocess.run(
        ["git", "worktree", "add", "-q", "-b", "test", str(worktree)],
        cwd=main,
        check=True,
    )
    assert AnnotationLayout.production(main) == AnnotationLayout.production(worktree)
    assert (
        AnnotationLayout.production(worktree).root == main / "outputs/chat_annotation"
    )


def test_sibling_root_and_symlink_escape_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fixed"):
        AnnotationLayout(
            tmp_path, tmp_path / "outputs/chat_annotation_other", "production"
        )
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    layout.root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (layout.root / "campaigns").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        layout.path("campaigns", "example", "plan.json")
    assert not list(outside.iterdir())


def test_only_registered_areas_and_filename_components_are_allowed(
    tmp_path: Path,
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    with pytest.raises(ValueError, match="unregistered"):
        layout.path(cast(Area, "migration_20261014"), "x")
    for value in ("..", "/absolute", "a/b", "a\\b", ""):
        with pytest.raises(ValueError, match="component"):
            layout.path("campaigns", value)
    assert (
        layout.path("campaigns", "new", "plan.json")
        == layout.root / "campaigns/new/plan.json"
    )
    assert layout.path("cache", "image.jpg") == layout.root / "cache/image.jpg"


def test_root_symlink_is_rejected_before_resolver_normalizes_it(tmp_path: Path) -> None:
    root = tmp_path / "annotation"
    root.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        AnnotationLayout.testing(root)
