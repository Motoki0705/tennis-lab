"""End-to-end coverage for linked-worktree-only Issue initialization."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / ".agents/skills/issue-subagent-workflow/scripts"


def load_initializer() -> ModuleType:
    module_name = "_issue_subagent_workflow_init_guard"
    path = SCRIPTS / "init_issue_task.py"
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


init = load_initializer()


def git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def create_repository(
    tmp_path: Path, *, detached: bool = False
) -> tuple[Path, Path]:
    primary = tmp_path / "primary"
    primary.mkdir()
    git(primary, "init")
    git(primary, "config", "user.email", "test@example.com")
    git(primary, "config", "user.name", "Test")
    (primary / "tracked.txt").write_text("base\n", encoding="utf-8")
    git(primary, "add", "tracked.txt")
    git(primary, "commit", "-m", "base")

    linked = tmp_path / "linked"
    if detached:
        git(primary, "worktree", "add", "--detach", str(linked), "HEAD")
    else:
        git(primary, "worktree", "add", "-b", "issue-822", str(linked), "HEAD")
    return primary, linked


def issue_payload(number: int = 822) -> dict[str, Any]:
    return {
        "number": number,
        "title": "Linked worktree guard",
        "body": "## Acceptance checklist\n\n- [ ] Initialize safely\n",
        "url": f"https://github.com/example/repo/issues/{number}",
        "state": "OPEN",
        "labels": [],
        "updatedAt": "2026-08-29T00:00:00Z",
    }


def invoke(
    monkeypatch: pytest.MonkeyPatch,
    cwd: Path,
    *arguments: str,
) -> int:
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(sys, "argv", ["init_issue_task.py", "822", *arguments])
    return int(init.main())


def snapshot_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_primary_rejects_before_github_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    primary, _linked = create_repository(tmp_path)
    github = Mock(return_value=issue_payload())
    monkeypatch.setattr(init, "run_gh", github)

    assert invoke(monkeypatch, primary) == 2

    github.assert_not_called()
    assert not (primary / ".codex/tasks").exists()
    error = capsys.readouterr().err
    assert "primary worktree" in error
    assert "Create and enter a dedicated linked worktree" in error
    assert "rerun the initializer" in error


def test_primary_rejection_preserves_preexisting_task_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary, _linked = create_repository(tmp_path)
    task_root = primary / ".codex/tasks"
    sentinel = task_root / "issue-existing/sentinel.bin"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"preserve-existing-output\x00")
    before = snapshot_files(task_root)
    github = Mock(return_value=issue_payload())
    monkeypatch.setattr(init, "run_gh", github)

    assert invoke(monkeypatch, primary) == 2

    github.assert_not_called()
    assert snapshot_files(task_root) == before


def test_primary_cannot_use_explicit_root_as_an_escape_hatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary, _linked = create_repository(tmp_path)
    github = Mock(return_value=issue_payload())
    monkeypatch.setattr(init, "run_gh", github)
    custom_root = primary / "custom/tasks"

    assert invoke(monkeypatch, primary, "--root", str(custom_root)) == 2

    github.assert_not_called()
    assert not custom_root.exists()


def test_linked_default_initializes_only_the_active_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary, linked = create_repository(tmp_path)
    github = Mock(return_value=issue_payload())
    monkeypatch.setattr(init, "run_gh", github)

    assert invoke(monkeypatch, linked) == 0

    github.assert_called_once_with(822, init.DEFAULT_REPO)
    task = linked / ".codex/tasks/issue-822"
    assert (task / "issue.json").is_file()
    assert (task / "issue.md").is_file()
    assert (task / "state.toml").is_file()
    assert (task / "02-planning/checks.json").is_file()
    assert not (primary / ".codex/tasks/issue-822").exists()


def test_linked_default_anchors_at_top_level_from_nested_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _primary, linked = create_repository(tmp_path)
    nested = linked / "nested/directory"
    nested.mkdir(parents=True)
    monkeypatch.setattr(init, "run_gh", Mock(return_value=issue_payload()))

    assert invoke(monkeypatch, nested) == 0

    assert (linked / ".codex/tasks/issue-822/state.toml").is_file()
    assert not (nested / ".codex").exists()


def test_explicit_relative_root_inside_linked_worktree_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _primary, linked = create_repository(tmp_path)
    nested = linked / "nested"
    nested.mkdir()
    github = Mock(return_value=issue_payload())
    monkeypatch.setattr(init, "run_gh", github)

    assert invoke(monkeypatch, nested, "--root", "custom/tasks") == 0

    github.assert_called_once()
    assert (nested / "custom/tasks/issue-822/state.toml").is_file()
    assert not (linked / ".codex").exists()


def test_explicit_absolute_root_inside_linked_worktree_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary, linked = create_repository(tmp_path)
    nested = linked / "nested"
    nested.mkdir()
    task_root = linked / "absolute/tasks"
    github = Mock(return_value=issue_payload())
    monkeypatch.setattr(init, "run_gh", github)

    assert invoke(monkeypatch, nested, "--root", str(task_root)) == 0

    github.assert_called_once()
    assert (task_root / "issue-822/state.toml").is_file()
    assert not (linked / ".codex").exists()
    assert not (primary / ".codex").exists()


def test_linked_rejects_roots_outside_its_canonical_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    primary, linked = create_repository(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    escape = linked / "escape"
    escape.symlink_to(outside, target_is_directory=True)
    github = Mock(return_value=issue_payload())
    monkeypatch.setattr(init, "run_gh", github)
    roots = (
        primary / ".codex/tasks",
        outside / "tasks",
        Path("../outside/dotdot-tasks"),
        escape / "tasks",
    )

    for root in roots:
        assert invoke(monkeypatch, linked, "--root", str(root)) == 2
        assert "--root must resolve within" in capsys.readouterr().err

    github.assert_not_called()
    assert not (primary / ".codex").exists()
    assert not (outside / "tasks").exists()
    assert not (outside / "dotdot-tasks").exists()


def test_linked_rejects_default_root_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _primary, linked = create_repository(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (linked / ".codex").symlink_to(outside, target_is_directory=True)
    github = Mock(return_value=issue_payload())
    monkeypatch.setattr(init, "run_gh", github)

    assert invoke(monkeypatch, linked) == 2

    github.assert_not_called()
    assert not (outside / "tasks").exists()
    assert "default task root must resolve within" in capsys.readouterr().err


def test_linked_refresh_preserves_output_location_and_increments_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary, linked = create_repository(tmp_path)
    github = Mock(return_value=issue_payload())
    monkeypatch.setattr(init, "run_gh", github)

    assert invoke(monkeypatch, linked) == 0
    task = linked / ".codex/tasks/issue-822"
    with (task / "state.toml").open("rb") as handle:
        assert tomllib.load(handle)["attempt"] == 1

    assert invoke(monkeypatch, linked, "--refresh-issue") == 0

    assert github.call_count == 2
    with (task / "state.toml").open("rb") as handle:
        assert tomllib.load(handle)["attempt"] == 2
    assert not (primary / ".codex/tasks/issue-822").exists()


def test_linked_refresh_rejects_issue_directory_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _primary, linked = create_repository(tmp_path)
    task_root = linked / ".codex/tasks"
    task_root.mkdir(parents=True)
    outside_task = tmp_path / "outside-task"
    outside_task.mkdir()
    (outside_task / "sentinel.bin").write_bytes(b"outside-must-remain-unchanged\x00")
    (task_root / "issue-822").symlink_to(outside_task, target_is_directory=True)
    before = snapshot_files(outside_task)
    github = Mock(return_value=issue_payload())
    monkeypatch.setattr(init, "run_gh", github)

    result = invoke(monkeypatch, linked, "--refresh-issue")

    github.assert_not_called()
    assert (result, snapshot_files(outside_task)) == (2, before)


def test_detached_linked_worktree_is_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary, linked = create_repository(tmp_path, detached=True)
    assert git(linked, "branch", "--show-current") == ""
    monkeypatch.setattr(init, "run_gh", Mock(return_value=issue_payload()))

    assert invoke(monkeypatch, linked) == 0

    assert (linked / ".codex/tasks/issue-822/state.toml").is_file()
    assert not (primary / ".codex/tasks/issue-822").exists()


@pytest.mark.parametrize("repository_kind", ["non-git", "bare"])
def test_non_worktree_locations_fail_closed(
    repository_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cwd = tmp_path / repository_kind
    cwd.mkdir()
    if repository_kind == "bare":
        git(cwd, "init", "--bare")
    github = Mock(return_value=issue_payload())
    monkeypatch.setattr(init, "run_gh", github)

    assert invoke(monkeypatch, cwd) == 2

    github.assert_not_called()
    assert not (cwd / ".codex").exists()
    assert "cannot verify the active Git linked worktree" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "expected"),
    [
        (128, "", "forced Git failure", "forced Git failure"),
        (
            0,
            "true\n/tmp/top\n/tmp/git-dir\n",
            "",
            "malformed worktree metadata",
        ),
    ],
)
def test_git_discovery_failure_or_malformed_output_fails_before_side_effects(
    returncode: int,
    stdout: str,
    stderr: str,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    github = Mock(return_value=issue_payload())
    monkeypatch.setattr(init, "run_gh", github)

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["git"],
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    monkeypatch.setattr(init.subprocess, "run", fake_run)

    assert invoke(monkeypatch, cwd) == 2

    github.assert_not_called()
    assert not (cwd / ".codex").exists()
    assert expected in capsys.readouterr().err


def test_initialization_documentation_orders_worktree_before_initializer() -> None:
    skill = (
        ROOT / ".agents/skills/issue-subagent-workflow/SKILL.md"
    ).read_text(encoding="utf-8")
    steps = (
        "Select and fix the GitHub Issue number",
        "Create, enter, and verify a dedicated linked worktree",
        "python .agents/skills/issue-subagent-workflow/scripts/init_issue_task.py",
    )

    offsets = [skill.index(step) for step in steps]
    assert offsets == sorted(offsets)
    assert "[worktree-create skill](../worktree-create/SKILL.md)" in skill
    assert "The omitted `--root` resolves to `.codex/tasks`" in skill
    assert "Do not run the initializer" in skill
    assert "from the primary worktree." in skill
