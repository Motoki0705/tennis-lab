"""Real git histories distinguish base conflict resolution from unrelated edits."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"
sys.path[:0] = [str(OPS), str(OPS / "review")]
workspace: Any = importlib.import_module("review_workspace")


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.mark.parametrize(
    "kind", ["rebase", "merge", "conflict", "extra", "rename", "mode", "stacked"]
)
def test_inheritance_uses_file_deltas_and_the_base_period(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.chdir(root)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.invalid")

    def commit(message: str) -> str:
        git(root, "add", ".")
        git(root, "commit", "-qm", message)
        return git(root, "rev-parse", "HEAD")

    (root / "a.py").write_text("original\n")
    (root / "b.py").write_text("untouched\n")
    start = commit("base")
    if kind == "stacked":
        (root / "dependency.py").write_text("dependency\n")
        start = commit("upstream PR")
    git(root, "checkout", "-qb", "feature")
    (root / "a.py").write_text("own change\n")
    head = commit("own change")
    old = {
        "number": 1,
        "branch": "feature",
        "base_ref": "upstream" if kind == "stacked" else "main",
        "base_sha": start,
        "merge_base_sha": start,
        "head_sha": head,
    }
    git(root, "checkout", "-q", "main")
    (root / ("a.py" if kind == "conflict" else "base.py")).write_text("base change\n")
    base = commit("base moved")
    git(root, "checkout", "-q", "feature")
    if kind == "merge":
        git(root, "merge", "-q", "--no-ff", "-m", "take base", base)
    elif kind == "conflict":
        proc = subprocess.run(
            ["git", "-C", str(root), "merge", base], capture_output=True
        )
        assert proc.returncode == 1
        (root / "a.py").write_text("resolved base and own changes\n")
        commit("resolve")
    else:
        git(root, "rebase", "--onto", base, start)
    if kind == "extra":
        (root / "b.py").write_text("unreviewed\n")
        commit("extra")
    elif kind == "rename":
        git(root, "mv", "b.py", "renamed.py")
        commit("rename unrelated")
    elif kind == "mode":
        (root / "b.py").chmod(0o755)
        commit("mode unrelated")
    current = {
        **old,
        "base_ref": "main",
        "base_sha": base,
        "head_sha": git(root, "rev-parse", "HEAD"),
        "merge_base_sha": base,
    }
    assert workspace.verify_rebased_snapshot(old, current) == (
        kind not in {"extra", "rename", "mode"}
    )
    assert not workspace.verify_rebased_snapshot(old, {**current, "branch": "other"})
