"""A successful Actions check must match the pinned PR and test merge."""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / ".agents/ops"
HEAD = "a" * 40
BASE = "b" * 40
MERGE = "c" * 40


@pytest.fixture
def workflow_run() -> dict[str, object]:
    return {
        "path": ".github/workflows/knowledge.yml",
        "event": "pull_request",
        "status": "completed",
        "conclusion": "success",
        "head_sha": HEAD,
        "id": 141,
        "run_attempt": 2,
    }


@pytest.fixture
def record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "scope": "records-only",
        "pr_number": 12,
        "head_sha": HEAD,
        "base_sha": BASE,
        "merge_sha": MERGE,
        "run_id": 141,
        "run_attempt": 2,
        "checks": {"kg_validate": "success", "kg_repro_paths": "success"},
        "conclusion": "success",
    }


@pytest.fixture
def knowledge_ci(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(OPS))
    return importlib.import_module("lib.knowledge_ci")


@pytest.fixture
def verify(knowledge_ci: Any) -> Any:
    return knowledge_ci.verify_result


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=ci@example.test", *args],
        cwd=root,
        stderr=subprocess.PIPE,
        text=True,
    ).strip()


def commit_path(root: Path, path: str) -> str:
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("fixture\n", encoding="utf-8")
    git(root, "add", "--", path)
    git(root, "commit", "-qm", f"Add {path!r}")
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-qb", "main")
    commit_path(root, "README.md")
    return root


def test_knowledge_ci_paths_match_the_workflow(knowledge_ci: Any) -> None:
    workflow = yaml.load(
        (ROOT / knowledge_ci.WORKFLOW_PATH).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert (
        tuple(workflow["on"]["pull_request"]["paths"])
        == knowledge_ci.KNOWLEDGE_CI_PATHS
    )


def test_identical_base_requires_no_git_objects(
    tmp_path: Path, knowledge_ci: Any
) -> None:
    knowledge_ci.require_unchanged_base(tmp_path, BASE, BASE)


@pytest.mark.parametrize(
    ("path", "watched"),
    [
        (".agents/logs/parents/1/worker-log.md", False),
        ("knowledge-notes/result.md", False),
        ("knowledge/reports/nested/result\n検証.md", True),
        ("tests/e2e/skills/test_knowledge_example.py", True),
        ("tests/e2e/skills/test_knowledge_nested/example.py", False),
        ("src/tennis_scene/configs/nested/source.yaml", True),
        ("LICENSES.md", True),
        ("licenses.md", False),
    ],
)
def test_base_advance_uses_root_relative_workflow_paths(
    git_repo: Path, knowledge_ci: Any, path: str, watched: bool
) -> None:
    base = git(git_repo, "rev-parse", "HEAD")
    tip = commit_path(git_repo, path)
    if watched:
        with pytest.raises(ValueError, match="main changed knowledge paths") as exc:
            knowledge_ci.require_unchanged_base(git_repo, base, tip)
        assert repr(path) in str(exc.value)
    else:
        knowledge_ci.require_unchanged_base(git_repo, base, tip)


def test_moving_a_watched_file_outside_workflow_paths_counts_as_a_change(
    git_repo: Path, knowledge_ci: Any
) -> None:
    path = "knowledge/summary.md"
    base = commit_path(git_repo, path)
    git(git_repo, "config", "diff.renames", "true")
    git(git_repo, "mv", path, "summary.md")
    git(git_repo, "commit", "-qm", "Move the summary")
    with pytest.raises(ValueError, match="knowledge/summary.md"):
        knowledge_ci.require_unchanged_base(
            git_repo, base, git(git_repo, "rev-parse", "HEAD")
        )


def test_divergent_base_is_rejected_even_with_a_replace_ref(
    git_repo: Path, knowledge_ci: Any
) -> None:
    fork = git(git_repo, "rev-parse", "HEAD")
    base = commit_path(git_repo, "previous.txt")
    git(git_repo, "switch", "-qc", "rewritten", fork)
    tip = commit_path(git_repo, "current.txt")
    git(git_repo, "replace", "--graft", tip, base)
    git(git_repo, "merge-base", "--is-ancestor", base, tip)
    with pytest.raises(ValueError, match="is not an ancestor"):
        knowledge_ci.require_unchanged_base(git_repo, base, tip)


def test_missing_base_objects_are_fetched_without_changing_checkout_or_refs(
    git_repo: Path, tmp_path: Path, knowledge_ci: Any
) -> None:
    local = tmp_path / "checkout"
    git(tmp_path, "clone", "-q", git_repo.as_uri(), str(local))
    checkout = git(local, "rev-parse", "HEAD")
    base = commit_path(git_repo, "notes/first.md")
    tip = commit_path(git_repo, "notes/next.md")
    git(git_repo, "tag", "new-tag")
    git(local, "config", "fetch.prune", "true")
    git(local, "config", "fetch.pruneTags", "true")
    git(local, "update-ref", "refs/remotes/origin/local-only", checkout)
    refs = git(local, "show-ref")
    fetch_head = local / ".git/FETCH_HEAD"
    fetch_head.write_bytes(b"unchanged\n")
    with pytest.raises(subprocess.CalledProcessError):
        git(local, "cat-file", "-t", base)
    with pytest.raises(subprocess.CalledProcessError):
        git(local, "cat-file", "-t", tip)

    knowledge_ci.require_unchanged_base(local, base, tip)

    assert (
        git(local, "cat-file", "-t", base)
        == git(local, "cat-file", "-t", tip)
        == "commit"
    )
    assert git(local, "show-ref") == refs
    assert git(local, "rev-parse", "HEAD") == checkout
    assert git(local, "status", "--porcelain") == ""
    assert fetch_head.read_bytes() == b"unchanged\n"


def test_missing_base_rejects_when_pinned_fetch_fails(
    git_repo: Path, knowledge_ci: Any
) -> None:
    with pytest.raises(ValueError, match=r"cannot verify knowledge paths.*fetch"):
        knowledge_ci.require_unchanged_base(
            git_repo, git(git_repo, "rev-parse", "HEAD"), BASE
        )


def test_missing_tree_rejects_when_diff_is_unavailable(
    git_repo: Path, knowledge_ci: Any
) -> None:
    base = git(git_repo, "rev-parse", "HEAD")
    tip = commit_path(git_repo, "notes/main.md")
    tree = git(git_repo, "rev-parse", f"{tip}^{{tree}}")
    (git_repo / ".git/objects" / tree[:2] / tree[2:]).unlink()
    with pytest.raises(ValueError, match=r"cannot verify knowledge paths.*diff"):
        knowledge_ci.require_unchanged_base(git_repo, base, tip)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("head_sha", "d" * 40, "head_sha does not match"),
        ("base_sha", "d" * 40, "base_sha does not match"),
        pytest.param(
            "merge_sha",
            "d" * 40,
            "merge_sha does not match",
            id="different-test-merge-sha",
        ),
        ("pr_number", 13, "pr_number does not match"),
        ("run_attempt", 1, "run_attempt does not match"),
        ("run_id", 140, "run_id does not match"),
        ("conclusion", "failure", "both knowledge records checks"),
        ("checks", {"kg_validate": "success"}, "both knowledge records checks"),
        (
            "checks",
            {"kg_validate": "success", "kg_repro_paths": "skipped"},
            "both knowledge records checks",
        ),
        ("schema_version", True, "schema version 1 records-only"),
        ("scope", "normal", "schema version 1 records-only"),
    ],
)
def test_rejects_invalid_or_stale_artifact(
    verify: Any,
    record: dict[str, object],
    workflow_run: dict[str, object],
    field: str,
    value: object,
    error: str,
) -> None:
    record[field] = value
    with pytest.raises(ValueError, match=error):
        verify(
            json.dumps(record).encode(),
            workflow_run,
            pr_number=12,
            head_sha=HEAD,
            base_sha=BASE,
            merge_sha=MERGE,
        )


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"path": ".github/workflows/ci.yml"}, "Knowledge library pull_request run"),
        ({"event": "push"}, "Knowledge library pull_request run"),
        ({"head_sha": "d" * 40}, "workflow run HEAD"),
        (
            {"status": "in_progress", "conclusion": None},
            "workflow run has not succeeded",
        ),
        ({"conclusion": "cancelled"}, "workflow run has not succeeded"),
    ],
)
def test_rejects_unsuccessful_or_unrelated_run_even_with_a_successful_artifact(
    verify: Any,
    record: dict[str, object],
    workflow_run: dict[str, object],
    change: dict[str, object],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        verify(
            json.dumps(record).encode(),
            {**workflow_run, **change},
            pr_number=12,
            head_sha=HEAD,
            base_sha=BASE,
            merge_sha=MERGE,
        )


@pytest.mark.parametrize(
    ("data", "error"),
    [
        (None, "missing knowledge records result"),
        (b"{", "invalid knowledge records result JSON"),
    ],
)
def test_rejects_missing_or_unreadable_result(
    verify: Any,
    workflow_run: dict[str, object],
    data: bytes | None,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        verify(
            data,
            workflow_run,
            pr_number=12,
            head_sha=HEAD,
            base_sha=BASE,
            merge_sha=MERGE,
        )
