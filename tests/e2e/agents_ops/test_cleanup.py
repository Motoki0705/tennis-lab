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
    def run(
        self, *args: str, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        (self.tmp / "prs.json").write_text(json.dumps(self.prs), encoding="utf-8")
        cfg = self.tmp / "config.toml"
        cfg.write_text(
            "".join(f"{k} = {json.dumps(v)}\n" for k, v in self.config.items()),
            encoding="utf-8",
        )
        command, *rest = args
        return subprocess.run(
            [
                sys.executable,
                str(CLEANUP),
                command,
                "--repo",
                str(self.repo),
                "--config",
                str(cfg),
                "--state-dir",
                str(self.state),
                "--report-json",
                str(self.tmp / "report.json"),
                *rest,
            ],
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
