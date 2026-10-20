"""CPU fixtures for safe reference migration after a repository history rewrite."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / ".agents/skills/knowledge-control/scripts/kg_remap_commits.py"
OLD = "abc1234" + "0" * 33
NEW = "def5678" + "0" * 33


@pytest.fixture
def remap(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    return importlib.import_module("kg_remap_commits")


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def initialize(root: Path) -> str:
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "config", "user.name", "fixture")
    (root / "README.md").write_text(f"commit `{OLD}`\n")
    git(root, "add", "README.md")
    git(root, "commit", "-qm", "fixture")
    return git(root, "rev-parse", "HEAD")


def test_full_and_short_refs_and_foreign_urls(remap: Any) -> None:
    resolver = remap.Resolver({OLD: NEW}, {NEW})
    text = (
        f"commit: {OLD}\ncheckout {OLD[:7]}\n"
        f"https://github.com/Motoki0705/tennis-lab/commit/{OLD}\n"
        f"https://github.com/elsewhere/project/commit/{OLD}\n"
        f"https://huggingface.co/owner/model/resolve/{OLD}/file\n"
        f"sha256: {OLD}{'1' * 24}\n"
        f"sha1: {OLD}\n"
        f"config_{OLD}.yaml\n"
    )
    updated, events = remap.transform(text, resolver)
    assert updated.count(NEW) == 3
    assert f"elsewhere/project/commit/{OLD}" in updated
    assert f"resolve/{OLD}/file" in updated
    assert f"sha256: {OLD}{'1' * 24}" in updated
    assert f"sha1: {OLD}" in updated
    assert f"config_{OLD}.yaml" in updated
    assert [row["status"] for row in events] == ["replace"] * 3
    assert remap.transform(updated, resolver)[0] == updated


def test_foreign_link_label_does_not_change(remap: Any) -> None:
    resolver = remap.Resolver({OLD: NEW}, {NEW})
    text = (
        f"[upstream commit `{OLD[:8]}`](https://github.com/elsewhere/code/tree/{OLD})\n"
    )
    assert remap.transform(text, resolver) == (text, [])


def test_ambiguous_prefix_even_if_targets_converge(remap: Any) -> None:
    other = OLD[:7] + "1" * 33
    resolver = remap.Resolver({OLD: NEW, other: NEW}, {NEW})
    updated, events = remap.transform(f"commit: {OLD[:7]}", resolver)
    assert updated == f"commit: {OLD[:7]}"
    assert events[0]["status"] == "ambiguous_old_prefix"


def test_worktree_pin_and_short_github_link(remap: Any) -> None:
    resolver = remap.Resolver({OLD: NEW}, {NEW})
    text = (
        f'git -C "$root" worktree add --detach "$code" {OLD[:8]}\n'
        f"# {OLD[:8]} (the commit that added the code)\n"
        f"https://github.com/Motoki0705/tennis-lab/tree/{OLD[:8]}/src\n"
    )
    updated, events = remap.transform(text, resolver)
    assert updated.count(NEW) == 3
    assert len(events) == 3


def test_unpublished_and_dropped_refs_are_not_guessed(remap: Any) -> None:
    resolver = remap.Resolver({OLD: NEW}, set())
    assert resolver.resolve(OLD) == ("unpublished_target", NEW)
    resolver = remap.Resolver({OLD: "0" * 40}, set())
    assert resolver.resolve(OLD) == ("dropped_commit", None)
    updated, events = remap.transform(
        "git checkout 0123456\ncolor: 0123456\n", resolver
    )
    assert updated == "git checkout 0123456\ncolor: 0123456\n"
    assert [row["status"] for row in events] == ["unknown_commit_reference"]


@pytest.mark.parametrize(
    "content",
    ["", "old new\n", f"old new\n{OLD} bad\n", f"old new\n{OLD} {NEW}\n{OLD} {NEW}\n"],
)
def test_invalid_maps_fail(tmp_path: Path, remap: Any, content: str) -> None:
    path = tmp_path / "commit-map"
    path.write_text(content)
    with pytest.raises(ValueError):
        remap.read_map(path)


def test_protected_evidence_and_binary_files(tmp_path: Path, remap: Any) -> None:
    new = initialize(tmp_path)
    files = {
        "saved.patch": f"+ commit: {OLD}\n".encode(),
        "artifact-manifest.json": f'{{"removed_from_commit": "{OLD}"}}\n'.encode(),
        "binary.dat": b"\0commit: " + OLD.encode(),
        "document.md": f"commit {OLD[:7]}\n".encode(),
    }
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    git(tmp_path, "add", ".")
    changes, report = remap.plan(tmp_path, {OLD: new})
    assert {p.name for p in changes} == {"README.md", "document.md"}
    assert report["counts"]["replace"] == 2
    assert set(report["protected_files"]) == {"saved.patch", "artifact-manifest.json"}
    assert all((tmp_path / name).read_bytes() == data for name, data in files.items())


def test_pinned_run_rebaseline_is_narrow_and_explicit(
    tmp_path: Path, remap: Any
) -> None:
    new = initialize(tmp_path)
    run = tmp_path / "knowledge/runs/run-fixture"
    run.mkdir(parents=True)
    (run / "repro.sh").write_text(f"git checkout {OLD}\n")
    patch = f"+commit: {OLD}\n".encode()
    (run / "uncommitted.patch").write_bytes(patch)
    old_hash = remap.tree_sha256(run)
    policy = tmp_path / "knowledge/run-storage-exceptions.json"
    exceptions: dict[str, Any] = {
        "schema_version": 1,
        "files": [
            {
                "path": "runs/run-fixture/uncommitted.patch",
                "sha256": hashlib.sha256(patch).hexdigest(),
                "reason": "legacy patch",
            }
        ],
        "runs": [
            {"path": "runs/run-fixture", "sha256": old_hash, "reason": "legacy run"}
        ],
        "legacy_manifests": [],
    }
    policy.write_text(json.dumps(exceptions))
    git(tmp_path, "add", ".")
    with pytest.raises(ValueError, match="refresh-run-exceptions"):
        remap.plan(tmp_path, {OLD: new})
    changes, report = remap.plan(tmp_path, {OLD: new}, refresh_run_exceptions=True)
    assert report["run_exception_updates"][0]["changed_files"] == ["repro.sh"]
    updated = json.loads(changes[policy])
    assert updated["files"] == exceptions["files"]
    assert updated["runs"][0]["path"] == exceptions["runs"][0]["path"]
    for path, data in changes.items():
        path.write_bytes(data)
    assert updated["runs"][0]["sha256"] == remap.tree_sha256(run)
    assert (run / "uncommitted.patch").read_bytes() == patch
    (run / "unexpected.txt").write_text("new content")
    with pytest.raises(ValueError, match="differs before"):
        remap.plan(tmp_path, {OLD: new}, refresh_run_exceptions=True)


def test_no_following_working_tree_symlink(tmp_path: Path, remap: Any) -> None:
    new = initialize(tmp_path)
    (tmp_path / "README.md").unlink()
    (tmp_path / "README.md").symlink_to(tmp_path / "other.md")
    with pytest.raises(ValueError, match="symlink"):
        remap.plan(tmp_path, {OLD: new})


def test_cli_dry_run_and_unknown_write_gate(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    new = initialize(root)
    (root / "README.md").write_text(f"commit {OLD}\ncommit 9876abc\n")
    git(root, "add", "README.md")
    git(root, "commit", "-qm", "record unresolved reference")
    mapping = tmp_path / "commit-map"
    mapping.write_text(f"old new\n{OLD} {new}\n")
    report = tmp_path / "report.json"
    args = [
        sys.executable,
        "-B",
        str(SCRIPT),
        "--repo",
        str(root),
        "--commit-map",
        str(mapping),
        "--report",
        str(report),
    ]
    before = (root / "README.md").read_bytes()
    assert subprocess.run(args, capture_output=True).returncode == 2
    assert (root / "README.md").read_bytes() == before
    assert subprocess.run([*args, "--write"], capture_output=True).returncode == 2
    assert (root / "README.md").read_bytes() == before
    assert (
        subprocess.run(
            [*args, "--write", "--allow-unresolved"], capture_output=True
        ).returncode
        == 0
    )
    assert f"commit {new}\ncommit 9876abc\n" == (root / "README.md").read_text()
    assert json.loads(report.read_text())["applied"] is True


def test_write_refuses_existing_tracked_changes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    new = initialize(root)
    path = root / "README.md"
    path.write_text(f"uncommitted work and commit {OLD}\n")
    before = path.read_bytes()
    mapping = tmp_path / "commit-map"
    mapping.write_text(f"old new\n{OLD} {new}\n")
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--repo",
            str(root),
            "--commit-map",
            str(mapping),
            "--report",
            str(tmp_path / "report.json"),
            "--allow-unresolved",
            "--write",
        ],
        capture_output=True,
    )
    assert result.returncode != 0
    assert b"clean tracked checkout" in result.stderr
    assert path.read_bytes() == before


def test_offline_issue_export_only(tmp_path: Path, remap: Any) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    new = initialize(root)
    exported = tmp_path / "issue-bodies"
    exported.mkdir()
    (exported / "issue.md").write_text(f"commit `{OLD[:8]}`\n")
    changes, report = remap.plan(root, {OLD: new}, text_dir=exported)
    assert list(changes.values()) == [f"commit `{new}`\n".encode()]
    assert report["counts"]["replace"] == 1
    mapping = tmp_path / "commit-map"
    mapping.write_text(f"old new\n{OLD} {new}\n")
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--repo",
            str(root),
            "--commit-map",
            str(mapping),
            "--report",
            str(tmp_path / "report.json"),
            "--text-dir",
            "issue-bodies",
            "--write",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert (exported / "issue.md").read_text() == f"commit `{new}`\n"
