"""End-to-end tests for .agents/ops/cleanup on throwaway git repositories.

Every test builds its own repository + worktrees under ``tmp_path`` and runs
cleanup.py as a subprocess with a fake ``gh`` that serves a canned PR list.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import stat
import subprocess
import sys
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
            "protected_worktree_globs": [str(self.wts / "pinned-*")],
            "default_branch_refs": ["refs/heads/main"],
            "stale_days": 14,
            "min_idle_hours": 0,
            "disposable_untracked": [".training_queue"],
            "disposable_ignored": ["__pycache__", ".venv"],
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
        path = self.wts / name
        if detach is not None:
            self.git("worktree", "add", "-q", "--detach", str(path), detach)
        else:
            assert branch is not None
            self.git("worktree", "add", "-q", str(path), branch)
        return path

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


# ----------------------------------------------------------------- merge detection


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


def test_unpushed_commits_after_merge_and_behind_head(lab: Lab) -> None:
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
    assert "commits_after_merge" in codes(ahead)
    assert "1 commit(s) after merged PR" in next(
        r["detail"] for r in ahead["reasons"] if r["code"] == "commits_after_merge"
    )
    assert entries["branch:feat/behind"]["category"] == "auto_delete"
    missing = entries["branch:feat/missing"]
    assert missing["category"] == "protected" and "undetermined" in codes(missing)


def test_ancestor_of_default_branch_needs_stale_threshold(lab: Lab) -> None:
    lab.branch("old/direct-merge")
    lab.merge("old/direct-merge", squash=False)
    entries = lab.scan()
    entry = entries["branch:old/direct-merge"]
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


# ------------------------------------------------------------------ worktree state


def _merged_worktree(lab: Lab, name: str) -> Path:
    tip = lab.branch(f"feat/{name}")
    lab.merge(f"feat/{name}", squash=True)
    lab.pr(f"feat/{name}", tip)
    return lab.worktree(name, f"feat/{name}")


def test_worktree_cleanliness(lab: Lab) -> None:
    clean = _merged_worktree(lab, "clean")
    (clean / ".venv").symlink_to(lab.tmp)  # symlinks are disposable
    (clean / ".training_queue").symlink_to(lab.queue)
    (clean / "pkg" / "__pycache__").mkdir(parents=True)
    (clean / "pkg" / "__pycache__" / "x.pyc").write_bytes(b"0")

    dirty = _merged_worktree(lab, "dirty")
    (dirty / "README").write_text("edited\n", encoding="utf-8")
    untracked = _merged_worktree(lab, "untracked")
    (untracked / "notes.md").write_text("draft\n", encoding="utf-8")
    ignored = _merged_worktree(lab, "ignored")
    (ignored / "outputs").mkdir()
    (ignored / "outputs" / "metrics.json").write_text("{}", encoding="utf-8")

    entries = lab.scan()
    e = entries["branch:feat/clean"]
    assert e["category"] == "auto_delete", e["reasons"]
    assert e["worktree_state"]["disposable_untracked"] == [".training_queue"]
    assert set(e["worktree_state"]["disposable_ignored"]) == {
        ".venv",
        "pkg/__pycache__/",
    }
    assert e["worktree"] == str(clean)
    assert isinstance(e["disk_bytes"], int) and e["disk_bytes"] > 0
    assert codes(entries["branch:feat/dirty"]) >= {"uncommitted_changes"}
    assert codes(entries["branch:feat/untracked"]) >= {"untracked_files"}
    assert codes(entries["branch:feat/ignored"]) >= {"ignored_files"}
    for name in ("dirty", "untracked", "ignored"):
        assert entries[f"branch:feat/{name}"]["category"] == "approval_required"

    proc = lab.run("apply-auto")
    assert proc.returncode == 0, proc.stderr
    assert not clean.exists() and not lab.has_branch("feat/clean")
    assert (
        lab.tmp.exists() and (lab.queue / "jobs").is_dir()
    )  # symlink targets untouched
    for path in (dirty, untracked, ignored):
        assert path.exists()
    assert (ignored / "outputs" / "metrics.json").exists()


def test_populated_submodule_requires_approval(lab: Lab) -> None:
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
    entry = lab.scan()["branch:feat/withsub"]
    assert entry["category"] == "approval_required"
    assert "submodule_populated" in codes(entry)


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
        "f = open(sys.argv[1] + '/by-fd/README')\n"
        "pathlib.Path(sys.argv[2]).touch()\n"
        "time.sleep(60)\n"
    )
    # argv names a parent dir so only the open fd can match the worktree path.
    background.append(
        subprocess.Popen(
            [sys.executable, "-c", holder, str(lab.wts), str(ready)], cwd=lab.tmp
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
        f"#!/usr/bin/env bash\n# name: train-b\ncd /elsewhere\npython x.py --out {pending}/outputs\n",
        encoding="utf-8",
    )
    # A path that only shares a prefix must not match (queue-pending vs queue-pending-x).
    lookalike = _merged_worktree(lab, "queue-pending-x")

    locked = _merged_worktree(lab, "locked")
    lab.git("worktree", "lock", "--reason", "campaign running", str(locked))
    pinned = _merged_worktree(lab, "pinned-model")

    base_tip = lab.branch("ai-env/base")
    lab.merge("ai-env/base", squash=True)
    lab.pr("ai-env/base", base_tip)
    keep_tip = lab.branch("keep/forever")
    lab.merge("keep/forever", squash=True)
    lab.pr("keep/forever", keep_tip)
    here = _merged_worktree(lab, "here")

    entries = lab.scan(cwd=here)
    expected = {
        "branch:feat/by-cwd": "process_using_worktree",
        "branch:feat/by-fd": "process_using_worktree",
        "branch:feat/by-argv": "process_using_worktree",
        "branch:feat/queue-running": "queue_job_references",
        "branch:feat/queue-pending": "queue_job_references",
        "branch:feat/locked": "worktree_locked",
        "branch:feat/pinned-model": "protected_worktree",
        "branch:ai-env/base": "protected_branch",
        "branch:keep/forever": "protected_branch",
        "branch:feat/here": "current_checkout",
        "branch:main": "main_worktree",
    }
    for entry_id, code in expected.items():
        assert entries[entry_id]["category"] == "protected", (
            entry_id,
            entries[entry_id]["reasons"],
        )
        assert code in codes(entries[entry_id]), (
            entry_id,
            entries[entry_id]["reasons"],
        )
    assert "running/train-a" in str(entries["branch:feat/queue-running"]["reasons"])
    assert "jobs/train-b" in str(entries["branch:feat/queue-pending"]["reasons"])
    assert entries["branch:feat/queue-pending-x"]["category"] == "auto_delete"

    proc = lab.run("apply-auto", cwd=here)
    assert proc.returncode == 0, proc.stderr
    for path in (by_cwd, by_fd, by_argv, running, pending, locked, pinned, here):
        assert path.exists(), path
    assert not lookalike.exists()
    assert (
        lab.has_branch("ai-env/base")
        and lab.has_branch("keep/forever")
        and lab.has_branch("main")
    )
    assert [r["branch"] for r in lab.log() if r["phase"] == "done"] == [
        "feat/queue-pending-x"
    ]

    # Protected entries are refused even with explicit approval.
    proc = lab.run("delete", "branch:feat/by-cwd", cwd=here)
    assert proc.returncode == 1 and "protected" in proc.stderr
    assert by_cwd.exists()


def test_detached_worktree(lab: Lab) -> None:
    lab.branch("feat/tmp")
    lab.merge("feat/tmp", squash=False)
    merged_head = lab.tip("feat/tmp")
    lab.git("branch", "-D", "feat/tmp")
    unmerged_head = lab.branch("feat/side")
    lab.git("branch", "-D", "feat/side")
    lab.config["stale_days"] = 0
    a = lab.worktree("detached-merged", detach=merged_head)
    b = lab.worktree("detached-unmerged", detach=unmerged_head)
    entries = lab.scan()
    assert entries[f"worktree:{a}"]["category"] == "auto_delete"
    assert "detached_head" in codes(entries[f"worktree:{b}"])
    assert entries[f"worktree:{b}"]["category"] == "approval_required"


# ------------------------------------------------------------- deletion and logging


def test_apply_auto_logs_tip_and_branch_is_restorable(lab: Lab) -> None:
    wt = _merged_worktree(lab, "logged")
    tip = lab.tip("feat/logged")
    lab.git("config", "branch.feat/logged.description", "x")
    proc = lab.run("apply-auto")
    assert proc.returncode == 0, proc.stderr
    assert not wt.exists() and not lab.has_branch("feat/logged")
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
        assert r["tip"] == tip and r["worktree"] == str(wt) and r["approved"] is False
        assert "merged_pr" in {x["code"] for x in r["reasons"]}
    assert lab.report()["entries"] and any(
        e["action"] == "deleted" for e in lab.report()["entries"]
    )

    # README's restore procedure.
    lab.git("branch", "feat/logged", records[0]["tip"])
    assert lab.tip("feat/logged") == tip


def test_approved_delete_with_tip_guard(lab: Lab) -> None:
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


def test_approved_delete_of_dirty_worktree_needs_discard_and_saves_patch(
    lab: Lab,
) -> None:
    lab.branch("feat/wip")
    wt = lab.worktree("wip", "feat/wip")
    (wt / "README").write_text("local edit\n", encoding="utf-8")

    proc = lab.run("delete", "feat/wip")
    assert proc.returncode == 1 and "--discard-changes" in proc.stderr
    assert wt.exists()

    proc = lab.run("delete", "feat/wip", "--discard-changes")
    assert proc.returncode == 0, proc.stderr
    assert not wt.exists() and not lab.has_branch("feat/wip")
    (record,) = [r for r in lab.log() if r["phase"] == "done"]
    patch = Path(record["discarded_patch"])
    assert "local edit" in patch.read_text(encoding="utf-8")

    lab.git("branch", "feat/wip", record["tip"])
    lab.git("worktree", "add", "-q", str(wt), "feat/wip")
    lab.git("apply", str(patch), cwd=wt)
    assert (wt / "README").read_text(encoding="utf-8") == "local edit\n"


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


# ------------------------------------------------------------------ report schema


def test_report_json_schema(lab: Lab) -> None:
    classify = _load_classify()
    _merged_worktree(lab, "a")
    lab.branch("feat/b")
    lab.worktree("b", "feat/b")
    lab.scan()
    report = lab.report()

    assert report["schema_version"] == 1
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
    assert summary["branches"] == 3 and summary["worktrees"] == 3
    assert summary["disk_bytes_total"] == sum(
        summary["disk_bytes_by_category"].values()
    )

    entry_keys = {
        "id",
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
        assert e["category"] in summary["categories"]
        assert e["reasons"], e
        for r in e["reasons"]:
            assert set(r) == {"code", "kind", "detail"}
            assert classify.REASON_KINDS[r["code"]] == r["kind"]
        assert len(e["tip"]) == 40
    sizes = [w["disk_bytes"] for w in report["worktrees_by_size"]]
    assert (
        sizes == sorted(sizes, reverse=True) and len(sizes) == 2
    )  # main checkout is not measured
    pr = report["entries"][[e["id"] for e in report["entries"]].index("branch:feat/a")][
        "prs"
    ][0]
    assert set(pr) == {"number", "state", "head_oid", "merged_at", "base_ref"}

    readme = (CLEANUP_DIR / "README.md").read_text(encoding="utf-8")
    for code in classify.REASON_KINDS:
        assert f"`{code}`" in readme, (
            f"reason code {code} is not documented in README.md"
        )


# ------------------------------------------------- in-process helpers (review fixes)

TRAINING_QUEUE = ROOT / ".agents/skills/training-queue/scripts/training_queue.sh"


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


# ----------------------------------------- re-verification right before deletion


@pytest.mark.parametrize("change", ["process", "queue", "commit"])
def test_approved_delete_rechecks_each_entry_right_before_deleting(
    lab: Lab,
    background: list[subprocess.Popen[bytes]],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    lab.branch("feat/one")
    one = lab.worktree("one", "feat/one")
    lab.branch("feat/two")
    two = lab.worktree("two", "feat/two")
    argv = lab.argv("delete", "feat/one", "feat/two")
    cleanup = _import_cleanup(lab, monkeypatch)
    real_delete = cleanup.delete_entry

    def delete_then_change(repo: Path, entry: Any, state_dir: Path, **kw: Any) -> None:
        real_delete(repo, entry, state_dir, **kw)
        if entry.id != "branch:feat/one":
            return
        # Something starts using feat/two while feat/one is being removed.
        if change == "process":
            background.append(subprocess.Popen(["sleep", "60"], cwd=two))
        elif change == "queue":
            (lab.queue / "jobs" / "9_9_late.job").write_text(
                f"# name: late\ncd {two}\npython train.py\n", encoding="utf-8"
            )
        else:
            _commit(lab, two, "late.txt", "late\n")

    monkeypatch.setattr(cleanup, "delete_entry", delete_then_change)
    assert cleanup.main(argv) == 1

    assert not one.exists() and not lab.has_branch("feat/one")
    assert two.exists() and lab.has_branch("feat/two")
    entries = {e["id"]: e for e in lab.report()["entries"]}
    assert entries["branch:feat/one"]["action"] == "deleted"
    refused = entries["branch:feat/two"]
    assert refused["action"] == "refused", refused
    expected = {
        "process": "+process_using_worktree",
        "queue": "+queue_job_references",
        "commit": "tip or worktree changed",
    }[change]
    assert expected in refused["action_error"]
    assert [r["branch"] for r in lab.log() if r["phase"] == "done"] == ["feat/one"]


def test_delete_entry_compares_tip_before_removing_the_worktree(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    lab.branch("feat/moving")
    wt = lab.worktree("moving", "feat/moving")
    cleanup = _import_cleanup(lab, monkeypatch)
    cfg = cleanup.load_config(lab.write_inputs())
    (entry,) = cleanup.build_inventory(
        lab.repo,
        cfg,
        invoking_cwd=lab.tmp,
        measure_disk=False,
        only_ids={"branch:feat/moving"},
        prs={},
    ).entries
    _commit(lab, wt, "new.txt", "new\n")  # tip moves after classification
    with pytest.raises(cleanup.DriftError, match="branch tip is"):
        cleanup.delete_entry(lab.repo, entry, lab.state, approved=True, discard=False)
    assert wt.exists() and lab.has_branch("feat/moving")
    assert [r["phase"] for r in lab.log()] == ["intent", "failed"]


# ----------------------------------------------- branches held by rebase / bisect


@pytest.mark.parametrize(
    "mode", ["merge-conflict", "merge-exec-stop", "apply-conflict"]
)
def test_branch_held_by_rebase_in_progress_is_protected(lab: Lab, mode: str) -> None:
    lab.branch("feat/rb")
    wt = lab.worktree("rb", "feat/rb")
    conflict = mode.endswith("conflict")
    tip = _commit(lab, wt, "README", "feat\n") if conflict else lab.tip("feat/rb")
    lab.pr("feat/rb", tip)  # merged on GitHub: an auto_delete candidate when idle
    _commit(lab, lab.repo, "README" if conflict else "other.txt", "main\n")
    assert lab.scan()["branch:feat/rb"]["category"] == "auto_delete"

    rebase = {
        "merge-conflict": ["rebase", "main"],
        "merge-exec-stop": ["rebase", "-x", "false", "main"],
        "apply-conflict": ["rebase", "--apply", "main"],
    }[mode]
    assert _git_status(lab, *rebase, cwd=wt) != 0  # stopped mid-rebase
    assert _git_status(lab, "symbolic-ref", "-q", "HEAD", cwd=wt) != 0  # detached

    entries = lab.scan()
    held = entries["branch:feat/rb"]
    assert held["worktree"] is None  # git worktree list no longer shows the branch
    assert held["category"] == "protected" and "branch_in_use" in codes(held)
    assert f"rebase in progress in worktree {wt}" in str(held["reasons"])
    detached = entries[f"worktree:{wt}"]
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
    before = lab.scan()
    assert before["branch:feat/bs"]["category"] == "auto_delete"
    assert before["branch:feat/stack-base"]["category"] == "auto_delete"

    lab.git("bisect", "start", "feat/bs", "main", cwd=bs)
    assert _git_status(lab, "symbolic-ref", "-q", "HEAD", cwd=bs) != 0
    # Stops after the first pick; stack-base is still pending in update-refs.
    assert _git_status(lab, "rebase", "--update-refs", "-x", "false", "main", cwd=top)
    assert "refs/heads/feat/stack-base" in (
        Path(lab.git("rev-parse", "--absolute-git-dir", cwd=top))
        / "rebase-merge"
        / "update-refs"
    ).read_text(encoding="utf-8")

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
    assert "operation_in_progress" in codes(entries[f"worktree:{bs}"])


# --------------------------------------------- queue jobs written by the real tool


def _enqueue(
    lab: Lab, command: str, cwd: Path, name: str, pwd: Path | None = None
) -> str:
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
        lab, f"python x.py --out {escaped}/outputs\necho done", lab.tmp, "arg"
    )
    assert str(by_arg) not in text and "$'" in text  # only escaped spellings
    link = lab.tmp / "wts link"
    link.symlink_to(lab.wts)
    text = _enqueue(
        lab, "python train.py", by_link, "via-link", pwd=link / "queued link"
    )
    assert "wts\\ link/queued\\ link" in text  # logical $PWD through the symlink

    entries = lab.scan()
    for entry_id, job in (
        ("branch:feat/q-cwd", "jobs/from-cwd"),
        ("branch:feat/q-arg", "jobs/arg"),
        ("branch:feat/q-link", "jobs/via-link"),
    ):
        e = entries[entry_id]
        assert e["category"] == "protected", (entry_id, e["reasons"])
        assert "queue_job_references" in codes(e) and job in str(e["reasons"])
    assert entries["branch:feat/q-lookalike"]["category"] == "auto_delete"

    proc = lab.run("apply-auto")
    assert proc.returncode == 0, proc.stderr
    assert by_cwd.exists() and by_arg.exists() and by_link.exists()
    assert not lookalike.exists()


def test_unparsable_queue_job_stops_the_run(lab: Lab) -> None:
    (lab.queue / "running" / "1_1_bad.job").write_text(
        "# name: bad\ncd '/unterminated\n", encoding="utf-8"
    )
    proc = lab.run("scan")
    assert proc.returncode == 1 and "cannot parse training-queue job" in proc.stderr


@pytest.mark.parametrize("locale", ["C", "C.UTF-8"])
def test_shell_words_decodes_bash_printf_q(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, locale: str
) -> None:
    cleanup = _import_cleanup(lab, monkeypatch)
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
    assert cleanup is not None


# ------------------------------------------------------------------- patch backups


def test_backups_for_ids_with_colliding_slugs_never_overwrite(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    lab.branch("feat/a")
    slash = lab.worktree("a-slash", "feat/a")
    lab.branch("feat_a")
    underscore = lab.worktree("a-underscore", "feat_a")
    (slash / "README").write_text("from feat/a\n", encoding="utf-8")
    (underscore / "README").write_text("from feat_a\n", encoding="utf-8")

    # One run, so both backups are written within the same second.
    proc = lab.run("delete", "feat/a", "feat_a", "--discard-changes")
    assert proc.returncode == 0, proc.stderr
    patches = {
        r["branch"]: Path(r["discarded_patch"])
        for r in lab.log()
        if r["phase"] == "done"
    }
    assert set(patches) == {"feat/a", "feat_a"}
    assert patches["feat/a"] != patches["feat_a"]
    assert "from feat/a" in patches["feat/a"].read_text(encoding="utf-8")
    assert "from feat_a" in patches["feat_a"].read_text(encoding="utf-8")

    cleanup = _import_cleanup(lab, monkeypatch)
    assert cleanup._slug("branch:feat/a") == cleanup._slug("branch:feat_a")
    # Even if two names ever coincided, the second write fails instead of overwriting.
    lab.branch("feat/c")
    wt = lab.worktree("c", "feat/c")
    (wt / "README").write_text("first\n", encoding="utf-8")
    monkeypatch.setattr(cleanup, "backup_name", lambda entry_id: "fixed.patch")
    first = cleanup.write_backup(lab.state, "branch:feat/c", wt)
    (wt / "README").write_text("second\n", encoding="utf-8")
    with pytest.raises(cleanup.CleanupError, match="refusing to overwrite"):
        cleanup.write_backup(lab.state, "branch:feat_c", wt)
    assert "first" in first.read_text(encoding="utf-8")
