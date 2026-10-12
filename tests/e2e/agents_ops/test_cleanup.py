"""End-to-end tests for .agents/ops/cleanup on throwaway git repositories.

Every test builds its own repository + worktrees under ``tmp_path`` and runs
cleanup.py (as a subprocess, or in-process when a test needs to hook it) with a
fake ``gh`` that serves a canned PR list. Scope under test: ``apply-auto`` deletes
only merged branches without a worktree; a worktree is removed only by an
approved ``delete worktree:<path>`` when it is completely clean.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
CLEANUP_DIR = ROOT / ".agents/ops/cleanup"
CLEANUP = CLEANUP_DIR / "cleanup.py"

FAKE_GH = """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$FAKE_GH_ARGS"
[[ -n "${FAKE_GH_FAIL:-}" ]] && { echo "gh: simulated failure" >&2; exit 1; }
cat "$FAKE_GH_PRS"
"""


def _load_classify() -> Any:
    spec = importlib.util.spec_from_file_location(
        "cleanup_classify", CLEANUP_DIR / "classify.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Lab:
    """A temporary main checkout plus helpers to build branches, PRs and worktrees."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.repo = tmp / "repo"
        self.wts = tmp / "wts"
        # Worktree basenames are unique per run: name mentions in any process's argv
        # protect a worktree, so a concurrent run of this file must not collide.
        self.uid = uuid.uuid4().hex[:8]
        self.queue = tmp / "queue"
        self.state = tmp / "state"
        self.bin = tmp / "bin"
        self.prs: list[dict[str, Any]] = []
        self.next_pr = 1
        home = tmp / "home"
        for d in (
            self.repo,
            self.wts,
            self.queue / "jobs",
            self.queue / "running",
            self.bin,
            home,
        ):
            d.mkdir(parents=True)
        gh = self.bin / "gh"
        gh.write_text(FAKE_GH, encoding="utf-8")
        gh.chmod(gh.stat().st_mode | stat.S_IEXEC)
        self.env = {
            **os.environ,
            "HOME": str(home),
            "XDG_STATE_HOME": str(tmp / "xdg-state"),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "FAKE_GH_PRS": str(tmp / "prs.json"),
            "FAKE_GH_ARGS": str(tmp / "gh_args.txt"),
        }
        self.config: dict[str, Any] = {
            "protected_branches": ["main", "ai-env/base"],
            "protected_branch_globs": ["keep/*"],
            "protected_worktree_globs": [str(self.wts / f"{self.uid}-pinned-*")],
            "default_branch_refs": ["refs/heads/main"],
            "stale_days": 14,
            "min_idle_hours": 0,
            "queue_dir": str(self.queue),
            "pr_limit": 100,
        }
        self.git("init", "-q", "-b", "main")
        (self.repo / ".gitignore").write_text(
            "outputs/\n__pycache__/\n.venv\n", encoding="utf-8"
        )
        (self.repo / "README").write_text("base\n", encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-qm", "init")

    # -- git helpers -------------------------------------------------------------
    def git(self, *args: str, cwd: Path | None = None) -> str:
        proc = subprocess.run(
            ["git", "-c", "commit.gpgsign=false", *args],
            cwd=cwd or self.repo,
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert proc.returncode == 0, f"git {args} failed: {proc.stderr}"
        return proc.stdout.strip()

    def tip(self, ref: str) -> str:
        return self.git("rev-parse", ref)

    def branch(self, name: str, commits: int = 1, base: str = "main") -> str:
        """Create ``name`` from ``base`` with ``commits`` new commits; return the tip."""
        self.git("branch", name, base)
        tmp_wt = self.tmp / f"scratch-{name.replace('/', '_')}"
        self.git("worktree", "add", "-q", str(tmp_wt), name)
        for i in range(commits):
            (tmp_wt / f"{name.replace('/', '_')}-{i}.txt").write_text(
                f"{i}\n", encoding="utf-8"
            )
            self.git("add", "-A", cwd=tmp_wt)
            self.git("commit", "-qm", f"{name} {i}", cwd=tmp_wt)
        self.git("worktree", "remove", str(tmp_wt))
        return self.tip(name)

    def merge(self, name: str, *, squash: bool) -> None:
        if squash:
            self.git("merge", "-q", "--squash", name)
            self.git("commit", "-qm", f"squash {name}")
        else:
            self.git("merge", "-q", "--no-ff", "-m", f"merge {name}", name)

    def worktree(
        self, name: str, branch: str | None = None, *, detach: str | None = None
    ) -> Path:
        path = self.wts / self.wt_name(name)
        if detach is not None:
            self.git("worktree", "add", "-q", "--detach", str(path), detach)
        else:
            assert branch is not None
            self.git("worktree", "add", "-q", str(path), branch)
        return path

    def wt_name(self, name: str) -> str:
        return f"{self.uid}-{name}"

    def pr(
        self, branch: str, head: str, state: str = "MERGED", *, cross: bool = False
    ) -> int:
        number = self.next_pr
        self.next_pr += 1
        self.prs.append(
            {
                "number": number,
                "state": state,
                "headRefName": branch,
                "headRefOid": head,
                "mergedAt": "2026-10-01T00:00:00Z" if state == "MERGED" else None,
                "baseRefName": "main",
                "isCrossRepository": cross,
            }
        )
        return number

    # -- running cleanup ---------------------------------------------------------
    def write_inputs(self) -> Path:
        """Write the fake gh PR list and config.toml; return the config path."""
        (self.tmp / "prs.json").write_text(json.dumps(self.prs), encoding="utf-8")
        cfg = self.tmp / "config.toml"
        cfg.write_text(
            "".join(f"{k} = {json.dumps(v)}\n" for k, v in self.config.items()),
            encoding="utf-8",
        )
        return cfg

    def argv(self, *args: str) -> list[str]:
        command, *rest = args
        return [
            command,
            "--repo",
            str(self.repo),
            "--config",
            str(self.write_inputs()),
            "--state-dir",
            str(self.state),
            "--report-json",
            str(self.tmp / "report.json"),
            *rest,
        ]

    def run(
        self, *args: str, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CLEANUP), *self.argv(*args)],
            cwd=cwd or self.repo,
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
        )

    def scan(self, cwd: Path | None = None) -> dict[str, dict[str, Any]]:
        proc = self.run("scan", cwd=cwd)
        assert proc.returncode == 0, proc.stderr
        return {e["id"]: e for e in self.report()["entries"]}

    def report(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(
            (self.tmp / "report.json").read_text(encoding="utf-8")
        )
        return data

    def log(self) -> list[dict[str, Any]]:
        path = self.state / "deleted.jsonl"
        if not path.exists():
            return []
        return [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]

    def has_branch(self, name: str) -> bool:
        return bool(self.git("branch", "--list", name))

    def push_main(self) -> None:
        """Pretend main was pushed: refs/remotes/origin/main = main."""
        self.git("update-ref", "refs/remotes/origin/main", "main")


def codes(entry: dict[str, Any]) -> set[str]:
    return {r["code"] for r in entry["reasons"]}


@pytest.fixture
def lab(tmp_path: Path) -> Lab:
    return Lab(tmp_path)


@pytest.fixture
def background(lab: Lab) -> Iterator[list[subprocess.Popen[bytes]]]:
    procs: list[subprocess.Popen[bytes]] = []
    yield procs
    for p in procs:
        p.kill()
        p.wait()


def _wait_ready(path: Path) -> None:
    import time

    for _ in range(200):
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"helper process did not signal readiness via {path}")


TRAINING_QUEUE = ROOT / ".agents/skills/training-queue/scripts/training_queue.sh"


def _merged_worktree(lab: Lab, name: str) -> Path:
    """Branch feat/<name> merged via PR (tip == PR head), checked out in a worktree."""
    tip = lab.branch(f"feat/{name}")
    lab.merge(f"feat/{name}", squash=True)
    lab.pr(f"feat/{name}", tip)
    return lab.worktree(name, f"feat/{name}")


def _merged_in_repo(lab: Lab, branch: str) -> Path:
    """A PR-merged branch checked out at ``<repo>/.claude/worktrees/<name>``."""
    tip = lab.branch(branch)
    lab.pr(branch, tip)
    path = lab.repo / ".claude/worktrees" / lab.wt_name(branch.split("/", 1)[1])
    lab.git("worktree", "add", "-q", str(path), branch)
    return path


def wt_id(path: Path) -> str:
    return f"worktree:{path}"


def _import_cleanup(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import cleanup.py in-process (same env/cwd as ``lab.run``) so tests can hook it."""
    monkeypatch.syspath_prepend(str(CLEANUP_DIR))
    for name in ("cleanup", "inventory", "classify"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import cleanup  # type: ignore[import-not-found]

    assert Path(cleanup.__file__).resolve() == CLEANUP.resolve()
    for key, value in lab.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(lab.repo)
    return cleanup


def _git_status(lab: Lab, *args: str, cwd: Path) -> int:
    """Run git where failure is expected (rebase stopping); return the exit code."""
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=cwd,
        env=lab.env,
        capture_output=True,
        check=False,
    ).returncode


def _commit(lab: Lab, cwd: Path, rel: str, text: str) -> str:
    (cwd / rel).write_text(text, encoding="utf-8")
    lab.git("add", "-A", cwd=cwd)
    lab.git("commit", "-qm", f"edit {rel}", cwd=cwd)
    return lab.git("rev-parse", "HEAD", cwd=cwd)


def _enqueue(
    lab: Lab, command: str, cwd: Path, name: str, pwd: Path | None = None
) -> str:
    """Enqueue through the real training_queue.sh (never runs the job)."""
    env = {**lab.env, "TRAINING_QUEUE_DIR": str(lab.queue), "PWD": str(pwd or cwd)}
    proc = subprocess.run(
        ["bash", str(TRAINING_QUEUE), "add", command, "--name", name],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    (job,) = sorted((lab.queue / "jobs").glob(f"*_{name}.job"))
    return job.read_text(encoding="utf-8")


# ------------------------------------------------------------- branch classification


def test_merge_detection_regular_squash_and_unmerged(lab: Lab) -> None:
    regular = lab.branch("feat/regular")
    lab.merge("feat/regular", squash=False)
    squash = lab.branch("feat/squash", commits=2)
    lab.merge("feat/squash", squash=True)
    lab.pr("feat/regular", regular)
    lab.pr("feat/squash", squash)
    # Squash merges are invisible to ancestry: only the PR head proves the merge.
    assert lab.git("branch", "--merged", "main", "--list", "feat/squash") == ""

    lab.branch("feat/no-pr")
    closed = lab.branch("feat/closed")
    lab.pr("feat/closed", closed, "CLOSED")
    open_tip = lab.branch("feat/open")
    lab.pr("feat/open", open_tip, "OPEN")
    fork = lab.branch("feat/fork-name")
    lab.pr("feat/fork-name", fork, "MERGED", cross=True)  # a fork's PR proves nothing

    entries = lab.scan()
    assert entries["branch:feat/regular"]["category"] == "auto_delete"
    assert "merged_pr" in codes(entries["branch:feat/regular"])
    assert entries["branch:feat/squash"]["category"] == "auto_delete"
    assert "merged_pr" in codes(entries["branch:feat/squash"])
    assert entries["branch:feat/no-pr"]["category"] == "approval_required"
    assert "no_pr" in codes(entries["branch:feat/no-pr"])
    assert entries["branch:feat/closed"]["category"] == "approval_required"
    assert "pr_closed_unmerged" in codes(entries["branch:feat/closed"])
    assert entries["branch:feat/open"]["category"] == "protected"
    assert "open_pr" in codes(entries["branch:feat/open"])
    assert entries["branch:feat/fork-name"]["category"] == "approval_required"
    assert entries["branch:main"]["category"] == "protected"
    assert entries["branch:feat/squash"]["action"] == "would_delete"
    # scan is a dry run.
    assert lab.has_branch("feat/squash") and lab.log() == []


def test_only_tip_equal_to_merged_pr_head_is_auto(lab: Lab) -> None:
    first = lab.branch("feat/ahead")
    lab.merge("feat/ahead", squash=True)
    lab.pr("feat/ahead", first)
    extra = lab.branch("feat/ahead-extra", base="feat/ahead")
    lab.git("branch", "-f", "feat/ahead", extra)  # one commit after the merged PR head
    lab.git("branch", "-D", "feat/ahead-extra")

    behind = lab.branch("feat/behind")
    pushed_head = lab.branch("feat/behind-more", base="feat/behind")
    lab.git("branch", "-D", "feat/behind-more")  # PR head exists only as an object
    lab.pr("feat/behind", pushed_head)
    assert lab.tip("feat/behind") == behind

    lab.branch("feat/missing")
    lab.pr("feat/missing", "f" * 40)  # head never fetched locally

    entries = lab.scan()
    ahead = entries["branch:feat/ahead"]
    assert ahead["category"] == "approval_required"
    assert "1 commit(s) after merged PR" in str(ahead["reasons"])
    behind_entry = entries["branch:feat/behind"]
    assert behind_entry["category"] == "approval_required"
    assert "behind_merged_pr" in codes(behind_entry)
    missing = entries["branch:feat/missing"]
    assert missing["category"] == "protected" and "undetermined" in codes(missing)


def test_ancestor_of_default_branch_needs_stale_threshold(lab: Lab) -> None:
    lab.branch("old/direct-merge")
    lab.merge("old/direct-merge", squash=False)
    entry = lab.scan()["branch:old/direct-merge"]
    assert entry["category"] == "approval_required"
    assert codes(entry) >= {"ancestor_of_default_branch", "recently_active"}

    lab.config["stale_days"] = 0
    assert lab.scan()["branch:old/direct-merge"]["category"] == "auto_delete"


def test_recently_active_merged_branch_waits_for_min_idle(lab: Lab) -> None:
    tip = lab.branch("feat/fresh")
    lab.merge("feat/fresh", squash=True)
    lab.pr("feat/fresh", tip)
    lab.config["min_idle_hours"] = 1000
    entry = lab.scan()["branch:feat/fresh"]
    assert entry["category"] == "approval_required"
    assert "recently_active" in codes(entry)


# --------------------------------------------- apply-auto never touches worktrees


def test_apply_auto_deletes_only_branches_without_worktrees(lab: Lab) -> None:
    clean = _merged_worktree(lab, "clean")
    (clean / ".venv").symlink_to(lab.tmp)
    free_tip = lab.branch("feat/free")
    lab.pr("feat/free", free_tip)
    detached = lab.worktree("detached", detach="main")
    lab.push_main()

    entries = lab.scan()
    branch = entries["branch:feat/clean"]
    assert branch["category"] == "approval_required"
    assert codes(branch) >= {"merged_pr", "checked_out_in_worktree"}
    for path in (clean, detached):
        e = entries[wt_id(path)]
        assert e["kind"] == "worktree" and e["category"] == "approval_required"
        assert {r["kind"] for r in e["reasons"]} <= {"approval", "info"}, e["reasons"]
    assert entries[wt_id(clean)]["worktree_state"]["disposable"] == [".venv"]
    assert [
        e["id"] for e in lab.report()["entries"] if e["category"] == "auto_delete"
    ] == ["branch:feat/free"]

    proc = lab.run("apply-auto")
    assert proc.returncode == 0, proc.stderr
    assert clean.exists() and detached.exists() and lab.has_branch("feat/clean")
    assert not lab.has_branch("feat/free")
    assert [r["id"] for r in lab.log() if r["phase"] == "done"] == ["branch:feat/free"]


def test_apply_auto_logs_tip_and_branch_is_restorable(lab: Lab) -> None:
    tip = lab.branch("feat/logged")
    lab.pr("feat/logged", tip)
    lab.git("config", "branch.feat/logged.description", "line one\nline two")
    lab.git("config", "branch.feat/logged.remote", "origin")
    lab.git("config", "branch.feat/logged.merge", "refs/heads/feat/logged")
    proc = lab.run("apply-auto")
    assert proc.returncode == 0, proc.stderr
    assert not lab.has_branch("feat/logged")
    leftover = subprocess.run(
        ["git", "config", "--get-regexp", r"^branch\.feat/logged\."],
        cwd=lab.repo,
        env=lab.env,
        check=False,
    )
    assert leftover.returncode == 1  # branch config section removed with the ref

    records = [r for r in lab.log() if r["branch"] == "feat/logged"]
    assert [r["phase"] for r in records] == ["intent", "done"]
    for r in records:
        assert r["tip"] == tip and r["worktree"] is None and r["approved"] is False
        assert "merged_pr" in {x["code"] for x in r["reasons"]}
    assert any(e["action"] == "deleted" for e in lab.report()["entries"])

    assert records[0]["branch_config"] == [
        ["branch.feat/logged.description", "line one\nline two"],
        ["branch.feat/logged.remote", "origin"],
        ["branch.feat/logged.merge", "refs/heads/feat/logged"],
    ]

    # README's restore procedure: the ref, then each logged config value.
    lab.git("branch", "feat/logged", records[0]["tip"])
    for key, value in records[0]["branch_config"]:
        lab.git("config", "--add", key, value)
    assert lab.tip("feat/logged") == tip
    assert lab.git("config", "branch.feat/logged.merge") == "refs/heads/feat/logged"
    assert lab.git("config", "branch.feat/logged.description") == "line one\nline two"


# ----------------------------------------------- approved worktree removal: clean only


def test_only_completely_clean_worktrees_can_be_removed(lab: Lab) -> None:
    clean = _merged_worktree(lab, "clean")
    (clean / ".venv").symlink_to(lab.tmp)  # the one disposable entry

    blocked: dict[str, tuple[Path, str]] = {}
    path = _merged_worktree(lab, "dirty")
    (path / "README").write_text("edited\n", encoding="utf-8")
    blocked["dirty"] = (path, "uncommitted_changes")
    path = _merged_worktree(lab, "untracked")
    (path / "notes.md").write_text("draft\n", encoding="utf-8")
    blocked["untracked"] = (path, "untracked_files")
    path = _merged_worktree(lab, "ignored")
    (path / "outputs").mkdir()
    (path / "outputs" / "metrics.json").write_text("{}", encoding="utf-8")
    blocked["ignored"] = (path, "ignored_files")
    path = _merged_worktree(lab, "pycache")  # caches are work too: a human decides
    (path / "pkg" / "__pycache__").mkdir(parents=True)
    (path / "pkg" / "__pycache__" / "x.pyc").write_bytes(b"0")
    blocked["pycache"] = (path, "ignored_files")
    path = _merged_worktree(lab, "other-link")  # only `.venv` is a known symlink
    (path / ".training_queue").symlink_to(lab.queue)
    blocked["other-link"] = (path, "untracked_files")

    entries = lab.scan()
    assert {r["kind"] for r in entries[wt_id(clean)]["reasons"]} <= {"approval", "info"}
    for name, (path, code) in blocked.items():
        e = entries[wt_id(path)]
        assert e["category"] == "approval_required", name
        assert code in codes(e), (name, e["reasons"])
        proc = lab.run("delete", wt_id(path))
        assert proc.returncode == 1 and f"blocked: {code}" in proc.stderr, name
        assert path.exists(), name
    assert (blocked["ignored"][0] / "outputs" / "metrics.json").exists()

    head = lab.git("rev-parse", "HEAD", cwd=clean)
    proc = lab.run("delete", f"{wt_id(clean)}@{'0' * 12}")
    assert proc.returncode == 1 and "approval was for" in proc.stderr
    proc = lab.run("delete", f"{wt_id(clean)}@{head[:12]}")
    assert proc.returncode == 0, proc.stderr
    assert not clean.exists() and lab.tmp.is_dir()  # symlink target untouched
    assert lab.has_branch("feat/clean")  # removing a worktree keeps its branch
    (record,) = [r for r in lab.log() if r["phase"] == "done"]
    assert record["worktree"] == str(clean) and record["tip"] == head
    # With the worktree gone, the merged branch becomes an auto_delete candidate.
    assert lab.scan()["branch:feat/clean"]["category"] == "auto_delete"


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_hidden_index_flags_block_removal(lab: Lab, flag: str) -> None:
    wt = _merged_worktree(lab, "hidden")
    lab.git("update-index", flag, "README", cwd=wt)
    (wt / "README").write_text(
        "local edit git status does not show\n", encoding="utf-8"
    )
    assert lab.git("status", "--porcelain", "--ignored", cwd=wt) == ""  # looks clean

    entry = lab.scan()[wt_id(wt)]
    assert entry["category"] == "approval_required"
    assert "hidden_index_flags" in codes(entry)
    assert entry["worktree_state"]["hidden_index_flags"] == ["README"]
    proc = lab.run("delete", wt_id(wt))
    assert proc.returncode == 1 and "blocked: hidden_index_flags" in proc.stderr
    assert (wt / "README").read_text(encoding="utf-8") == (
        "local edit git status does not show\n"
    )


def test_populated_submodule_blocks_removal(lab: Lab) -> None:
    sub = lab.tmp / "sub"
    sub.mkdir()
    lab.git("init", "-q", "-b", "main", cwd=sub)
    lab.git("commit", "-q", "--allow-empty", "-m", "s", cwd=sub)
    lab.git(
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(sub),
        "third_party/sub",
    )
    lab.git("commit", "-qm", "add submodule")
    wt = _merged_worktree(lab, "withsub")
    lab.git(
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "update",
        "-q",
        "--init",
        cwd=wt,
    )
    entry = lab.scan()[wt_id(wt)]
    assert entry["category"] == "approval_required"
    assert "submodule_populated" in codes(entry)
    proc = lab.run("delete", wt_id(wt))
    assert proc.returncode == 1 and "submodule_populated" in proc.stderr
    assert (wt / "third_party/sub/.git").exists()


def test_unpushed_worktree_head_blocks_removal(lab: Lab) -> None:
    side = lab.branch("feat/side")
    lab.git("branch", "-D", "feat/side")
    unpushed = lab.worktree("unpushed", detach=side)
    local_only = lab.worktree("local-main", detach="main")  # local main is not a remote

    entries = lab.scan()
    for path in (unpushed, local_only):
        assert "unpushed_commits" in codes(entries[wt_id(path)])
    proc = lab.run("delete", wt_id(unpushed))
    assert proc.returncode == 1 and "unpushed_commits" in proc.stderr
    assert unpushed.exists()

    lab.push_main()
    pushed = lab.scan()[wt_id(local_only)]
    assert "unpushed_commits" not in codes(pushed)
    assert "contained in refs/remotes/origin/main" in str(pushed["reasons"])


def test_approved_branch_delete_with_tip_guard(lab: Lab) -> None:
    tip = lab.branch("feat/unmerged")
    proc = lab.run("delete", f"branch:feat/unmerged@{'0' * 12}")
    assert proc.returncode == 1 and "approval was for" in proc.stderr
    assert lab.has_branch("feat/unmerged")

    proc = lab.run("delete", f"branch:feat/unmerged@{tip[:12]}")
    assert proc.returncode == 0, proc.stderr
    assert not lab.has_branch("feat/unmerged")
    (record,) = [r for r in lab.log() if r["phase"] == "done"]
    assert record["approved"] is True and record["tip"] == tip

    proc = lab.run("delete", "branch:does-not-exist")
    assert proc.returncode == 1 and "no such branch" in proc.stderr

    lab.branch("feat/checked-out")
    wt = lab.worktree("checked-out", "feat/checked-out")
    proc = lab.run("delete", "feat/checked-out")
    assert proc.returncode == 1 and "checked_out_in_worktree" in proc.stderr
    assert wt.exists() and lab.has_branch("feat/checked-out")


# ---------------------------------------------------------------------- protection


def test_protection_conditions(
    lab: Lab, background: list[subprocess.Popen[bytes]]
) -> None:
    by_cwd = _merged_worktree(lab, "by-cwd")
    background.append(subprocess.Popen(["sleep", "60"], cwd=by_cwd))

    by_fd = _merged_worktree(lab, "by-fd")
    ready = lab.tmp / "fd-ready"
    holder = (
        "import sys, time, pathlib\n"
        "f = open(sys.argv[1] + '/' + sys.argv[3] + '/README')\n"
        "pathlib.Path(sys.argv[2]).touch()\n"
        "time.sleep(60)\n"
    )
    # argv carries the worktree name only as a bare word: only the open fd matches.
    background.append(
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                holder,
                str(lab.wts),
                str(ready),
                lab.wt_name("by-fd"),
            ],
            cwd=lab.tmp,
        )
    )
    _wait_ready(ready)

    by_argv = _merged_worktree(lab, "by-argv")
    background.append(
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import time; time.sleep(60)",
                f"--out={by_argv}/run",
            ],
            cwd=lab.tmp,
        )
    )

    running = _merged_worktree(lab, "queue-running")
    (lab.queue / "running" / "1_1_train.job").write_text(
        f"#!/usr/bin/env bash\n# name: train-a\ncd {running}\npython train.py\n",
        encoding="utf-8",
    )
    pending = _merged_worktree(lab, "queue-pending")
    (lab.queue / "jobs" / "2_2_train.job").write_text(
        f"#!/usr/bin/env bash\n# name: train-b\ncd {lab.repo}\n"
        f"python x.py --out {pending}/outputs\n",
        encoding="utf-8",
    )
    # A path that only shares a prefix must not match (queue-pending vs queue-pending-x).
    lookalike = _merged_worktree(lab, "queue-pending-x")

    locked = _merged_worktree(lab, "locked")
    lab.git("worktree", "lock", "--reason", "campaign running", str(locked))
    pinned = _merged_worktree(lab, "pinned-model")
    here = _merged_worktree(lab, "here")
    lab.git("branch", "ai-env/base", "main")
    lab.git("branch", "keep/forever", "main")
    lab.push_main()

    entries = lab.scan(cwd=here)
    expected = {
        wt_id(by_cwd): "process_using_worktree",
        wt_id(by_fd): "process_using_worktree",
        wt_id(by_argv): "process_using_worktree",
        wt_id(running): "queue_job_references",
        wt_id(pending): "queue_job_references",
        wt_id(locked): "worktree_locked",
        wt_id(pinned): "protected_worktree",
        wt_id(here): "current_checkout",
        "branch:ai-env/base": "protected_branch",
        "branch:keep/forever": "protected_branch",
        "branch:main": "current_checkout",
    }
    for entry_id, code in expected.items():
        e = entries[entry_id]
        assert e["category"] == "protected", (entry_id, e["reasons"])
        assert code in codes(e), (entry_id, e["reasons"])
    assert "running/train-a" in str(entries[wt_id(running)]["reasons"])
    assert "jobs/train-b" in str(entries[wt_id(pending)]["reasons"])
    assert entries[wt_id(lookalike)]["category"] == "approval_required"

    proc = lab.run("apply-auto", cwd=here)
    assert proc.returncode == 0, proc.stderr
    assert lab.has_branch("ai-env/base") and lab.has_branch("keep/forever")
    # Protected worktrees are refused even with explicit approval.
    for path in (by_cwd, by_fd, by_argv, running, pending, locked, pinned, here):
        proc = lab.run("delete", wt_id(path), cwd=here)
        assert proc.returncode == 1 and "protected" in proc.stderr, path
        assert path.exists(), path
    proc = lab.run("delete", wt_id(lookalike), cwd=here)
    assert proc.returncode == 0, proc.stderr
    assert not lookalike.exists()


def test_process_argv_relative_to_its_cwd_protects_worktree(
    lab: Lab, background: list[subprocess.Popen[bytes]]
) -> None:
    by_fragment = _merged_in_repo(lab, "feat/proc-frag")
    by_cwd = _merged_in_repo(lab, "feat/proc-rel")
    lookalike = _merged_in_repo(lab, "feat/proc-rel-x")
    sleeper = "import time; time.sleep(60)"
    # Started from the main checkout; no fd stays open in the worktree.
    background.append(
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                sleeper,
                f".claude/worktrees/{by_fragment.name}/t.py",
            ],
            cwd=lab.repo,
        )
    )
    # Neither the absolute path nor the name as a path component: only resolving
    # the argument against the process's cwd finds the worktree.
    background.append(
        subprocess.Popen(
            [sys.executable, "-c", sleeper, f"--dir={by_cwd.name}"],
            cwd=lab.repo / ".claude/worktrees",
        )
    )
    entries = lab.scan()
    for path in (by_fragment, by_cwd):
        e = entries[wt_id(path)]
        assert e["category"] == "protected" and "process_using_worktree" in codes(e)
    assert "relative to cwd" in str(entries[wt_id(by_cwd)]["reasons"])
    assert entries[wt_id(lookalike)]["category"] == "approval_required"


def test_detached_main_and_missing_detached_worktree_do_not_crash(lab: Lab) -> None:
    head = lab.branch("feat/gone")
    lab.git("branch", "-D", "feat/gone")
    gone = lab.worktree("gone", detach=head)
    shutil.rmtree(gone)  # now prunable
    lab.git("checkout", "-q", "--detach")  # main checkout on a detached HEAD

    entries = lab.scan()
    assert entries["branch:main"]["category"] == "protected"
    missing = entries[wt_id(gone)]
    assert missing["category"] == "protected" and "worktree_missing" in codes(missing)
    assert missing["last_activity"] is not None
    proc = lab.run("apply-auto")
    assert proc.returncode == 0, proc.stderr
    assert str(gone) in lab.git("worktree", "list")  # left for a manual prune


# ----------------------------------------------- branches held by rebase / bisect


@pytest.mark.parametrize(
    "mode", ["merge-conflict", "merge-exec-stop", "apply-conflict"]
)
def test_branch_held_by_rebase_in_progress_is_protected(lab: Lab, mode: str) -> None:
    lab.branch("feat/rb")
    wt = lab.worktree("rb", "feat/rb")
    conflict = mode.endswith("conflict")
    tip = _commit(lab, wt, "README", "feat\n") if conflict else lab.tip("feat/rb")
    lab.pr("feat/rb", tip)
    _commit(lab, lab.repo, "README" if conflict else "other.txt", "main\n")
    assert "checked_out_in_worktree" in codes(lab.scan()["branch:feat/rb"])

    rebase = {
        "merge-conflict": ["rebase", "main"],
        "merge-exec-stop": ["rebase", "-x", "false", "main"],
        "apply-conflict": ["rebase", "--apply", "main"],
    }[mode]
    assert _git_status(lab, *rebase, cwd=wt) != 0  # stopped mid-rebase
    assert _git_status(lab, "symbolic-ref", "-q", "HEAD", cwd=wt) != 0  # detached

    entries = lab.scan()
    held = entries["branch:feat/rb"]
    # git worktree list no longer shows the branch, yet it is in use.
    assert "checked_out_in_worktree" not in codes(held)
    assert held["category"] == "protected" and "branch_in_use" in codes(held)
    assert f"rebase in progress in worktree {wt}" in str(held["reasons"])
    detached = entries[wt_id(wt)]
    assert detached["category"] == "protected"
    assert "operation_in_progress" in codes(detached)

    proc = lab.run("apply-auto")
    assert proc.returncode == 0, proc.stderr
    assert lab.has_branch("feat/rb") and wt.exists()
    proc = lab.run("delete", "feat/rb")
    assert proc.returncode == 1 and "branch_in_use" in proc.stderr


def test_branches_held_by_bisect_and_update_refs_are_protected(lab: Lab) -> None:
    bs_tip = lab.branch("feat/bs", commits=3)
    lab.pr("feat/bs", bs_tip)
    bs = lab.worktree("bs", "feat/bs")
    base_tip = lab.branch("feat/stack-base")
    lab.pr("feat/stack-base", base_tip)
    lab.branch("feat/stack-top", base="feat/stack-base")
    top = lab.worktree("top", "feat/stack-top")
    _commit(lab, lab.repo, "other.txt", "main\n")
    assert lab.scan()["branch:feat/stack-base"]["category"] == "auto_delete"

    lab.git("bisect", "start", "feat/bs", "main", cwd=bs)
    assert _git_status(lab, "symbolic-ref", "-q", "HEAD", cwd=bs) != 0
    # Stops after the first pick; stack-base is still pending in update-refs.
    assert _git_status(lab, "rebase", "--update-refs", "-x", "false", "main", cwd=top)

    entries = lab.scan()
    for entry_id, detail in (
        ("branch:feat/bs", f"bisect in progress in worktree {bs}"),
        (
            "branch:feat/stack-base",
            f"rebase --update-refs in progress in worktree {top}",
        ),
        ("branch:feat/stack-top", f"rebase in progress in worktree {top}"),
    ):
        e = entries[entry_id]
        assert e["category"] == "protected" and "branch_in_use" in codes(e), e
        assert detail in str(e["reasons"])
    assert "operation_in_progress" in codes(entries[wt_id(bs)])
    proc = lab.run("apply-auto")
    assert proc.returncode == 0, proc.stderr
    assert lab.has_branch("feat/stack-base")


# ------------------------------------------------------------------ symbolic refs


def test_symbolic_ref_branches_are_protected_and_never_dereferenced(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    lab.config["stale_days"] = 0  # alias -> main would otherwise be auto_delete
    main_tip = lab.tip("main")
    lab.git("symbolic-ref", "refs/heads/alias", "refs/heads/main")
    target_tip = lab.branch("feat/target")
    lab.pr("feat/target", target_tip)
    lab.git("symbolic-ref", "refs/heads/alias2", "refs/heads/feat/target")

    entries = lab.scan()
    for entry_id in ("branch:alias", "branch:alias2", "branch:feat/target"):
        e = entries[entry_id]
        assert e["category"] == "protected", (entry_id, e["reasons"])
        assert "symbolic_ref" in codes(e)
    proc = lab.run("apply-auto")
    assert proc.returncode == 0, proc.stderr
    proc = lab.run("delete", f"branch:alias@{main_tip[:12]}")
    assert proc.returncode == 1 and "symbolic_ref" in proc.stderr
    assert lab.tip("main") == main_tip and lab.has_branch("feat/target")

    # A ref that turns symbolic after classification is refused before deletion.
    lab.git("branch", "feat/racy", "main")
    cleanup = _import_cleanup(lab, monkeypatch)
    cfg = cleanup.load_config(lab.write_inputs())
    (entry,) = cleanup.build_inventory(
        lab.repo,
        cfg,
        invoking_cwd=lab.tmp,
        measure_disk=False,
        only_ids={"branch:feat/racy"},
        prs={},
    ).entries
    lab.git("symbolic-ref", "refs/heads/feat/racy", "refs/heads/main")
    with pytest.raises(cleanup.DriftError, match="symbolic ref"):
        cleanup.delete_entry(lab.repo, entry, lab.state, approved=True)
    assert lab.tip("main") == main_tip

    # Even if it turned symbolic right after that check, --no-deref removes only
    # the alias: hide the symbolic-ref probe and delete.
    real_git = cleanup.git

    def git_hiding_symref(repo: Path, *args: str, **kw: Any) -> Any:
        if args[0] == "symbolic-ref":
            return subprocess.CompletedProcess(args, 1, "", "")
        return real_git(repo, *args, **kw)

    monkeypatch.setattr(cleanup, "git", git_hiding_symref)
    cleanup.delete_entry(lab.repo, entry, lab.state, approved=True)
    assert not lab.has_branch("feat/racy")
    assert lab.tip("main") == main_tip


# ----------------------------------------- re-verification right before deletion


def _clean_pushed_worktree(lab: Lab, name: str) -> Path:
    lab.branch(f"feat/{name}")
    lab.merge(f"feat/{name}", squash=False)
    lab.push_main()
    return lab.worktree(name, f"feat/{name}")


@pytest.mark.parametrize("change", ["process", "queue", "commit", "untracked"])
def test_approved_delete_rechecks_each_entry_right_before_deleting(
    lab: Lab,
    background: list[subprocess.Popen[bytes]],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    one = _clean_pushed_worktree(lab, "one")
    two = _clean_pushed_worktree(lab, "two")
    argv = lab.argv("delete", wt_id(one), wt_id(two))
    cleanup = _import_cleanup(lab, monkeypatch)
    real_delete = cleanup.delete_entry

    def delete_then_change(repo: Path, entry: Any, state_dir: Path, **kw: Any) -> None:
        real_delete(repo, entry, state_dir, **kw)
        if entry.id != wt_id(one):
            return
        # Something starts using `two` while `one` is being removed.
        if change == "process":
            background.append(subprocess.Popen(["sleep", "60"], cwd=two))
        elif change == "queue":
            (lab.queue / "jobs" / "9_9_late.job").write_text(
                f"# name: late\ncd {two}\npython train.py\n", encoding="utf-8"
            )
        elif change == "commit":
            _commit(lab, two, "late.txt", "late\n")
        else:
            (two / "notes.md").write_text("new work\n", encoding="utf-8")

    monkeypatch.setattr(cleanup, "delete_entry", delete_then_change)
    assert cleanup.main(argv) == 1

    assert not one.exists() and two.exists()
    entries = {e["id"]: e for e in lab.report()["entries"]}
    assert entries[wt_id(one)]["action"] == "deleted"
    refused = entries[wt_id(two)]
    assert refused["action"] == "refused", refused
    expected = {
        "process": "+process_using_worktree",
        "queue": "+queue_job_references",
        "commit": "tip or branch changed",
        "untracked": "+untracked_files",
    }[change]
    assert expected in refused["action_error"]


def test_worktree_head_is_compared_before_removal(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    wt = _clean_pushed_worktree(lab, "moving")
    cleanup = _import_cleanup(lab, monkeypatch)
    cfg = cleanup.load_config(lab.write_inputs())
    (entry,) = cleanup.build_inventory(
        lab.repo,
        cfg,
        invoking_cwd=lab.tmp,
        measure_disk=False,
        only_ids={wt_id(wt)},
        prs={},
    ).entries
    _commit(lab, wt, "new.txt", "new\n")  # HEAD moves after classification
    with pytest.raises(cleanup.DriftError, match="worktree HEAD is"):
        cleanup.delete_entry(lab.repo, entry, lab.state, approved=True)
    assert wt.exists()
    assert [r["phase"] for r in lab.log()] == ["intent", "failed"]


# --------------------------------------------- queue jobs written by the real tool


def test_queue_jobs_written_by_training_queue_sh_protect_paths_with_spaces(
    lab: Lab,
) -> None:
    def merged(branch: str, dirname: str) -> Path:
        tip = lab.branch(branch)
        lab.pr(branch, tip)
        return lab.worktree(dirname, branch)

    by_cwd = merged("feat/q-cwd", "queued cwd")
    by_arg = merged("feat/q-arg", "queued arg")
    by_link = merged("feat/q-link", "queued link")
    lookalike = merged("feat/q-lookalike", "queued cwd-x")

    text = _enqueue(lab, "python train.py", by_cwd, "from-cwd")
    assert "queued\\ cwd" in text and str(by_cwd) not in text  # printf %q form
    escaped = str(by_arg).replace(" ", "\\ ")
    text = _enqueue(
        lab, f"python x.py --out {escaped}/outputs\necho done", lab.repo, "arg"
    )
    assert str(by_arg) not in text and "$'" in text  # only escaped spellings
    link = lab.tmp / "wts link"
    link.symlink_to(lab.wts)
    text = _enqueue(
        lab, "python train.py", by_link, "via-link", pwd=link / by_link.name
    )
    assert f"wts\\ link/{lab.uid}-queued\\ link" in text  # logical $PWD

    entries = lab.scan()
    for path, job in (
        (by_cwd, "jobs/from-cwd"),
        (by_arg, "jobs/arg"),
        (by_link, "jobs/via-link"),
    ):
        e = entries[wt_id(path)]
        assert e["category"] == "protected", (path, e["reasons"])
        assert "queue_job_references" in codes(e) and job in str(e["reasons"])
    assert "queue_job_references" not in codes(entries[wt_id(lookalike)])


@pytest.mark.parametrize(
    "body",
    [
        "cd '/unterminated\n",  # unterminated quote
        "python train.py\n",  # no cd: relative words have no known base directory
        "cd $HOME/x\npython train.py\n",  # cd target needs expansion
    ],
)
def test_unparsable_queue_job_stops_the_run(lab: Lab, body: str) -> None:
    (lab.queue / "running" / "1_1_bad.job").write_text(
        "#!/usr/bin/env bash\n# name: bad\n" + body, encoding="utf-8"
    )
    proc = lab.run("scan")
    assert proc.returncode == 1 and "cannot parse training-queue job" in proc.stderr


def test_queue_job_relative_references_resolve_against_job_cwd(lab: Lab) -> None:
    feature = _merged_in_repo(lab, "feat/feature")
    other = _merged_in_repo(lab, "feat/other")
    lookalike = _merged_in_repo(lab, "feat/feature-x")

    # Enqueued from the main checkout; the worktree appears only as a relative path.
    text = _enqueue(
        lab,
        f"python .claude/worktrees/{feature.name}/scripts/train.py",
        lab.repo,
        "rel",
    )
    assert str(feature) not in text
    # A cd inside the command moves the base directory.
    text = _enqueue(
        lab,
        f"cd .claude/worktrees && python {other.name}/train.py"
        f" --cfg={lookalike.name}-y/a.yaml",
        lab.repo,
        "rel-cd",
    )
    assert str(other) not in text

    entries = lab.scan()
    for path, job in ((feature, "jobs/rel"), (other, "jobs/rel-cd")):
        e = entries[wt_id(path)]
        assert e["category"] == "protected", (path, e["reasons"])
        assert "queue_job_references" in codes(e) and job in str(e["reasons"])
    assert entries[wt_id(lookalike)]["category"] == "approval_required"


@pytest.mark.parametrize(
    "command",
    [
        "bash -lc 'cd {link} && python train.py'",  # symlink hidden in quoted code
        "sh -c 'python .claude/worktrees/{name}/train.py'",
        "eval 'cd .claude/worktrees; cd {name}'",
        "python x.py --dir $(cd .claude/worktrees && cd {name} && pwd)",
        "source $HOME/env.sh && python train.py",
        "cd {outside} && python train.py",
    ],
)
def test_unparseable_queue_job_blocks_every_worktree_removal(
    lab: Lab, command: str
) -> None:
    target = _merged_in_repo(lab, "feat/target")
    bystander = _merged_in_repo(lab, "feat/bystander")
    link = lab.tmp / "run-link"
    link.symlink_to(target)  # reachable only through a symlink outside the repo
    outside = lab.tmp / "elsewhere"
    outside.mkdir()
    plain_tip = lab.branch("feat/plain")  # merged branch without a worktree
    lab.pr("feat/plain", plain_tip)
    command = command.format(link=link, outside=outside, name=target.name)
    _enqueue(lab, f"cd {lab.repo}; {command}", lab.repo, "opaque")

    entries = lab.scan()
    for path in (target, bystander):
        e = entries[wt_id(path)]
        assert e["category"] == "protected", e["reasons"]
        assert "queue_job_unparseable" in codes(e) and "jobs/opaque" in str(
            e["reasons"]
        )
    assert entries["branch:feat/plain"]["category"] == "auto_delete"
    assert any("blocks worktree removal" in w for w in lab.report()["warnings"])

    proc = lab.run("delete", wt_id(target))
    assert proc.returncode == 1 and "queue_job_unparseable" in proc.stderr
    proc = lab.run("apply-auto")
    assert proc.returncode == 0, proc.stderr
    assert target.exists() and bystander.exists()
    assert not lab.has_branch("feat/plain")  # branches without worktrees proceed

    for job in (lab.queue / "jobs").glob("*.job"):
        job.unlink()
    assert lab.scan()[wt_id(bystander)]["category"] == "approval_required"


@pytest.mark.parametrize("locale", ["C", "C.UTF-8"])
def test_shell_words_decodes_bash_printf_q(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, locale: str
) -> None:
    _import_cleanup(lab, monkeypatch)
    inventory = sys.modules["inventory"]
    samples = [
        "/a b/c",
        "it's",
        'dq"x',
        "tab\tx",
        "nl\nx",
        "back\\slash",
        "$HOME/`x`",
        "ünï código",
        "",
        "*glob?[x]",
        "a;b&c|d(e)<f>",
        "#notcomment",
        "\x01ctl\x7f",
    ]
    out = subprocess.run(
        ["bash", "-c", 'for a in "$@"; do printf "%q " "$a"; done', "_", *samples],
        env={**lab.env, "LC_ALL": locale},
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8", errors="surrogateescape")
    assert inventory.shell_words(out) == samples, out


# ------------------------------------------------------------------ failure modes


def test_lock_prevents_concurrent_mutating_runs(lab: Lab) -> None:
    lab.state.mkdir(parents=True)
    with open(lab.state / "cleanup.lock", "w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proc = lab.run("apply-auto")
    assert proc.returncode == 3 and "another cleanup run" in proc.stderr


@pytest.mark.parametrize("breakage", ["gh", "queue", "config"])
def test_failures_are_not_silent(lab: Lab, breakage: str) -> None:
    if breakage == "gh":
        lab.env["FAKE_GH_FAIL"] = "1"
    elif breakage == "queue":
        lab.config["queue_dir"] = str(lab.tmp / "no-such-queue")
    else:
        lab.config["unknown_key"] = 1
    proc = lab.run("scan")
    assert proc.returncode == 1
    assert proc.stderr.startswith("cleanup: ")
    assert not (lab.tmp / "report.json").exists()


def test_discard_changes_option_is_gone(lab: Lab) -> None:
    proc = lab.run("delete", "feat/x", "--discard-changes")
    assert proc.returncode == 2 and "unrecognized arguments" in proc.stderr


# ------------------------------------------------------------------ report schema


def test_report_json_schema(lab: Lab) -> None:
    classify = _load_classify()
    _merged_worktree(lab, "a")
    lab.branch("feat/b")
    lab.worktree("b", "feat/b")
    lab.scan()
    report = lab.report()

    assert report["schema_version"] == 2
    assert report["mode"] == "dry-run"
    assert set(report) == {
        "schema_version",
        "generated_at",
        "host",
        "repo",
        "mode",
        "config",
        "summary",
        "worktrees_by_size",
        "warnings",
        "entries",
    }
    summary = report["summary"]
    assert set(summary["categories"]) == {
        "auto_delete",
        "approval_required",
        "protected",
    }
    assert sum(summary["categories"].values()) == len(report["entries"])
    assert summary["branches"] == 3 and summary["worktrees"] == 2
    assert summary["disk_bytes_total"] == sum(
        summary["disk_bytes_by_category"].values()
    )
    entry_keys = {
        "id",
        "kind",
        "category",
        "branch",
        "tip",
        "worktree",
        "reasons",
        "prs",
        "last_activity",
        "activity_source",
        "idle_days",
        "worktree_state",
        "disk_bytes",
        "disk_error",
        "action",
        "action_error",
    }
    for e in report["entries"]:
        assert set(e) == entry_keys
        assert e["kind"] in ("branch", "worktree")
        assert e["id"].startswith(e["kind"] + ":")
        assert e["category"] in summary["categories"]
        assert e["reasons"], e
        for r in e["reasons"]:
            assert set(r) == {"code", "kind", "detail"}
            assert classify.REASON_KINDS[r["code"]] == r["kind"]
        assert len(e["tip"]) == 40
        if e["kind"] == "worktree":
            assert e["category"] != "auto_delete"
            assert set(e["worktree_state"]) == {
                "dirty_tracked",
                "untracked",
                "ignored",
                "disposable",
                "submodule_populated",
                "hidden_index_flags",
            }
        else:
            assert e["worktree"] is None and e["worktree_state"] is None
    sizes = [w["disk_bytes"] for w in report["worktrees_by_size"]]
    assert sizes == sorted(sizes, reverse=True) and len(sizes) == 2
    pr = next(e for e in report["entries"] if e["id"] == "branch:feat/a")["prs"][0]
    assert set(pr) == {"number", "state", "head_oid", "merged_at", "base_ref"}

    readme = (CLEANUP_DIR / "README.md").read_text(encoding="utf-8")
    for code in classify.REASON_KINDS:
        assert f"`{code}`" in readme, (
            f"reason code {code} is not documented in README.md"
        )
