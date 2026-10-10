"""Collect the facts the cleanup policy needs: git, gh PRs, /proc, training queue.

Every probe either returns a definite answer or raises :class:`CleanupError`
(or marks the entry ``undetermined``); nothing is silently assumed.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
import tomllib
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from classify import (
    Decision,
    Facts,
    MergedPr,
    MergeRelation,
    Policy,
    WorktreeState,
    classify,
    is_disposable,
    matches_any,
)


class CleanupError(RuntimeError):
    """A probe failed; the run must stop with a non-zero exit status."""


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class Config:
    protected_branches: tuple[str, ...]
    protected_branch_globs: tuple[str, ...]
    protected_worktree_globs: tuple[str, ...]
    default_branch_refs: tuple[str, ...]
    stale_days: float
    min_idle_hours: float
    disposable_untracked: tuple[str, ...]
    disposable_ignored: tuple[str, ...]
    queue_dir: str
    pr_limit: int

    @property
    def policy(self) -> Policy:
        return Policy(stale_days=self.stale_days, min_idle_hours=self.min_idle_hours)

    def as_json(self) -> dict[str, Any]:
        return {
            k: list(v) if isinstance(v, tuple) else v for k, v in self.__dict__.items()
        }


_STR_LISTS = (
    "protected_branches",
    "protected_branch_globs",
    "protected_worktree_globs",
    "default_branch_refs",
    "disposable_untracked",
    "disposable_ignored",
)
_NUMBERS = ("stale_days", "min_idle_hours")


def load_config(path: Path) -> Config:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CleanupError(f"cannot read config {path}: {exc}") from exc
    expected = set(_STR_LISTS) | set(_NUMBERS) | {"queue_dir", "pr_limit"}
    if set(raw) != expected:
        missing, unknown = expected - set(raw), set(raw) - expected
        raise CleanupError(
            f"config {path}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}"
        )
    values: dict[str, Any] = {}
    for key in _STR_LISTS:
        val = raw[key]
        if not isinstance(val, list) or not all(isinstance(v, str) for v in val):
            raise CleanupError(f"config {path}: {key} must be a list of strings")
        values[key] = tuple(val)
    for key in _NUMBERS:
        val = raw[key]
        if isinstance(val, bool) or not isinstance(val, int | float) or val < 0:
            raise CleanupError(f"config {path}: {key} must be a non-negative number")
        values[key] = float(val)
    if not isinstance(raw["queue_dir"], str) or not raw["queue_dir"]:
        raise CleanupError(f"config {path}: queue_dir must be a non-empty string")
    values["queue_dir"] = raw["queue_dir"]
    if (
        isinstance(raw["pr_limit"], bool)
        or not isinstance(raw["pr_limit"], int)
        or raw["pr_limit"] <= 0
    ):
        raise CleanupError(f"config {path}: pr_limit must be a positive integer")
    values["pr_limit"] = raw["pr_limit"]
    if not values["default_branch_refs"]:
        raise CleanupError(f"config {path}: default_branch_refs must not be empty")
    return Config(**values)


# ------------------------------------------------------------------------ helpers


def run(
    cmd: Sequence[str], cwd: Path, ok_codes: Iterable[int] = (0,)
) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            list(cmd), cwd=cwd, text=True, capture_output=True, check=False
        )
    except OSError as exc:
        raise CleanupError(f"cannot run {cmd[0]}: {exc}") from exc
    if proc.returncode not in set(ok_codes):
        raise CleanupError(
            f"{' '.join(cmd)} (cwd={cwd}) exited {proc.returncode}: {proc.stderr.strip()}"
        )
    return proc


def git(
    repo: Path, *args: str, ok_codes: Iterable[int] = (0,)
) -> subprocess.CompletedProcess[str]:
    # --no-optional-locks: never refresh other sessions' index files while probing.
    return run(["git", "--no-optional-locks", *args], cwd=repo, ok_codes=ok_codes)


def main_checkout(cwd: Path) -> Path:
    """The main worktree of the repository containing ``cwd``."""
    out = git(cwd, "worktree", "list", "--porcelain", "-z").stdout
    first = out.split("\0", 1)[0]
    if not first.startswith("worktree "):
        raise CleanupError(f"unexpected `git worktree list` output in {cwd}")
    return Path(first[len("worktree ") :])


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return (
        git(
            repo, "merge-base", "--is-ancestor", ancestor, descendant, ok_codes=(0, 1)
        ).returncode
        == 0
    )


def has_commit(repo: Path, oid: str) -> bool:
    return (
        git(
            repo, "cat-file", "-e", f"{oid}^{{commit}}", ok_codes=(0, 1, 128)
        ).returncode
        == 0
    )


def path_is_within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


# ---------------------------------------------------------------------- git facts


@dataclass(frozen=True)
class WorktreeInfo:
    path: str
    head: str | None
    branch: str | None
    locked: str | None
    prunable: bool
    is_main: bool


def list_worktrees(repo: Path) -> list[WorktreeInfo]:
    out = git(repo, "worktree", "list", "--porcelain", "-z").stdout
    result: list[WorktreeInfo] = []
    for record in out.split("\0\0"):
        fields = [f for f in record.split("\0") if f]
        if not fields:
            continue
        attrs: dict[str, str] = {}
        for f in fields:
            key, _, value = f.partition(" ")
            attrs[key] = value
        if "worktree" not in attrs:
            raise CleanupError(f"unparsable worktree record: {fields}")
        if "bare" in attrs:
            raise CleanupError("bare repositories are not supported")
        branch = attrs.get("branch")
        if branch is not None:
            if not branch.startswith("refs/heads/"):
                raise CleanupError(f"unexpected worktree branch ref {branch}")
            branch = branch[len("refs/heads/") :]
        result.append(
            WorktreeInfo(
                path=attrs["worktree"],
                head=attrs.get("HEAD"),
                branch=branch,
                locked=attrs.get("locked"),
                prunable="prunable" in attrs or not Path(attrs["worktree"]).is_dir(),
                is_main=not result,
            )
        )
    return result


@dataclass(frozen=True)
class BranchInfo:
    name: str
    tip: str
    commit_time: float


def list_branches(repo: Path) -> list[BranchInfo]:
    fmt = "%(refname:lstrip=2)%00%(objectname)%00%(committerdate:unix)"
    out = git(repo, "for-each-ref", f"--format={fmt}", "refs/heads").stdout
    branches = []
    for line in out.splitlines():
        name, tip, ts = line.split("\0")
        branches.append(BranchInfo(name=name, tip=tip, commit_time=float(ts)))
    return branches


def merged_into(repo: Path, refs: Sequence[str]) -> dict[str, str]:
    """Branch name -> first default ref (from ``refs``) that contains its tip."""
    found: dict[str, str] = {}
    existing = [
        r
        for r in refs
        if git(
            repo, "rev-parse", "--verify", "-q", f"{r}^{{commit}}", ok_codes=(0, 1)
        ).returncode
        == 0
    ]
    if not existing:
        raise CleanupError(f"none of default_branch_refs exist: {list(refs)}")
    for ref in existing:
        out = git(
            repo,
            "for-each-ref",
            f"--merged={ref}",
            "--format=%(refname:lstrip=2)",
            "refs/heads",
        ).stdout
        for name in out.splitlines():
            found.setdefault(name, ref)
    return found


def commit_contained(repo: Path, oid: str, refs: Sequence[str]) -> str | None:
    for ref in refs:
        exists = (
            git(
                repo,
                "rev-parse",
                "--verify",
                "-q",
                f"{ref}^{{commit}}",
                ok_codes=(0, 1),
            ).returncode
            == 0
        )
        if exists and is_ancestor(repo, oid, ref):
            return ref
    return None


def worktree_gitdir(path: str) -> Path:
    return Path(git(Path(path), "rev-parse", "--absolute-git-dir").stdout.strip())


def common_dir(repo: Path) -> Path:
    return Path(
        git(
            repo, "rev-parse", "--path-format=absolute", "--git-common-dir"
        ).stdout.strip()
    )


def worktree_state(path: str, cfg: Config) -> WorktreeState:
    wt = Path(path)
    out = git(
        wt,
        "status",
        "--porcelain=v1",
        "-z",
        "--ignored=matching",
        "--untracked-files=normal",
        "--ignore-submodules=none",
    ).stdout
    fields = out.split("\0")
    dirty: list[str] = []
    untracked: list[str] = []
    ignored: list[str] = []
    disposable: dict[str, list[str]] = {"??": [], "!!": []}
    i = 0
    while i < len(fields):
        entry = fields[i]
        i += 1
        if not entry:
            continue
        xy, rel = entry[:2], entry[3:]
        if xy in ("??", "!!"):
            patterns = (
                cfg.disposable_untracked if xy == "??" else cfg.disposable_ignored
            )
            # Removing a worktree unlinks symlinks without touching their targets.
            if (wt / rel.rstrip("/")).is_symlink() or is_disposable(rel, patterns):
                disposable[xy].append(rel)
            else:
                (untracked if xy == "??" else ignored).append(rel)
            continue
        if xy[0] in "RC":
            i += 1  # skip the rename/copy source path
        dirty.append(rel)
    populated: list[str] = []
    stage = git(wt, "ls-files", "-s", "-z").stdout
    for item in stage.split("\0"):
        if item.startswith("160000 "):
            sub = item.split("\t", 1)[1]
            if (wt / sub / ".git").exists():
                populated.append(sub)
    return WorktreeState(
        dirty_tracked=tuple(dirty),
        untracked=tuple(untracked),
        ignored=tuple(ignored),
        disposable_untracked=tuple(disposable["??"]),
        disposable_ignored=tuple(disposable["!!"]),
        submodule_populated=tuple(populated),
    )


# ------------------------------------------------------------------------ gh PRs


@dataclass(frozen=True)
class PrInfo:
    number: int
    state: str
    head_ref: str
    head_oid: str
    merged_at: str | None
    base_ref: str

    def as_json(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "state": self.state,
            "head_oid": self.head_oid,
            "merged_at": self.merged_at,
            "base_ref": self.base_ref,
        }


def load_prs(repo: Path, limit: int) -> dict[str, list[PrInfo]]:
    import json

    fields = (
        "number,state,headRefName,headRefOid,mergedAt,baseRefName,isCrossRepository"
    )
    proc = run(
        ["gh", "pr", "list", "--state", "all", "--limit", str(limit), "--json", fields],
        cwd=repo,
    )
    try:
        raw = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise CleanupError(f"gh pr list returned invalid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise CleanupError("gh pr list did not return a JSON list")
    if len(raw) >= limit:
        raise CleanupError(
            f"gh pr list hit pr_limit={limit}; raise it so no PR is missed"
        )
    by_branch: dict[str, list[PrInfo]] = {}
    for item in raw:
        try:
            if item["isCrossRepository"]:
                continue  # a fork's branch name says nothing about our local branches
            if item["state"] not in ("OPEN", "CLOSED", "MERGED"):
                raise CleanupError(
                    f"PR #{item['number']} has unknown state {item['state']}"
                )
            pr = PrInfo(
                number=int(item["number"]),
                state=item["state"],
                head_ref=item["headRefName"],
                head_oid=item["headRefOid"],
                merged_at=item["mergedAt"],
                base_ref=item["baseRefName"],
            )
        except (KeyError, TypeError) as exc:
            raise CleanupError(f"gh pr list entry is missing fields: {item!r}") from exc
        by_branch.setdefault(pr.head_ref, []).append(pr)
    return by_branch


def merge_relation(repo: Path, tip: str, head: str) -> tuple[MergeRelation, int | None]:
    if tip == head:
        return "equal", None
    if not has_commit(repo, head):
        return "head_missing", None
    if is_ancestor(repo, tip, head):
        return "tip_behind_head", None
    if is_ancestor(repo, head, tip):
        count = int(git(repo, "rev-list", "--count", f"{head}..{tip}").stdout.strip())
        return "tip_ahead", count
    return "diverged", None


# ----------------------------------------------------------- processes and queue


@dataclass
class ProcessScan:
    paths: list[tuple[str, str]] = field(
        default_factory=list
    )  # (path, holder description)
    args: list[tuple[str, str]] = field(
        default_factory=list
    )  # (argv element, holder description)
    unreadable: list[str] = field(default_factory=list)

    def refs_for(self, worktree: str) -> list[str]:
        hits: list[str] = []
        pattern = re.compile(re.escape(worktree.rstrip("/")) + r"(?![\w.\-])")
        for target, desc in self.paths:
            if path_is_within(target, worktree):
                hits.append(desc)
        for arg, desc in self.args:
            if pattern.search(arg):
                hits.append(f"{desc} argv")
        return sorted(set(hits))


def scan_processes(proc_root: Path = Path("/proc")) -> ProcessScan:
    """Record every cwd/root/open-fd path and argv element of visible processes."""
    scan = ProcessScan()
    me = os.getpid()
    my_uid = os.getuid()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            argv = [
                a.decode(errors="replace")
                for a in (entry / "cmdline").read_bytes().split(b"\0")
                if a
            ]
            owner = entry.stat().st_uid
        except (FileNotFoundError, ProcessLookupError):
            continue  # exited while scanning
        except PermissionError:
            scan.unreadable.append(f"pid {entry.name} (cmdline unreadable)")
            continue
        desc = f"pid {entry.name} [{' '.join(argv)[:80]}]"
        for arg in argv:
            scan.args.append((arg, desc))
        links: list[Path] = [entry / "cwd", entry / "root"]
        hidden: list[str] = []
        try:
            links += sorted((entry / "fd").iterdir())
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            hidden.append("fd/")
        for link in links:
            try:
                target = os.readlink(link)
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError:
                hidden.append(link.name)
                continue
            if target.startswith("/"):
                scan.paths.append(
                    (target.removesuffix(" (deleted)"), f"{desc} {link.name}")
                )
        if hidden and owner == my_uid:
            scan.unreadable.append(
                f"{desc} (unreadable: {' '.join(hidden[:4])}{' ...' if len(hidden) > 4 else ''})"
            )
    return scan


@dataclass(frozen=True)
class QueueJob:
    label: str
    text: str


def load_queue_jobs(queue_dir: Path) -> list[QueueJob]:
    """Pending (``jobs/``) and running (``running/``) training-queue job files."""
    jobs: list[QueueJob] = []
    for state in ("jobs", "running"):
        directory = queue_dir / state
        if not directory.is_dir():
            raise CleanupError(f"training queue directory missing: {directory}")
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                continue  # job moved to done/ while scanning
            match = re.search(r"^# name: (.*)$", text, flags=re.MULTILINE)
            name = match.group(1).strip() if match else path.name
            jobs.append(QueueJob(label=f"{state}/{name}", text=text))
    return jobs


def queue_refs_for(jobs: Sequence[QueueJob], worktree: str) -> list[str]:
    pattern = re.compile(re.escape(worktree.rstrip("/")) + r"(?![\w.\-])")
    return [job.label for job in jobs if pattern.search(job.text)]


# ------------------------------------------------------------------- activity/disk


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


_REFLOG_TIME = re.compile(r"> (\d+) [+-]\d{4}(?:\t|$)")


def _reflog_time(path: Path) -> float | None:
    """Timestamp of the newest reflog entry (file mtimes change on `git gc`)."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return None
    lines = [line for line in lines if line.strip()]
    if not lines:
        return None
    match = _REFLOG_TIME.search(lines[-1])
    if match is None:
        raise CleanupError(f"cannot parse the last reflog entry of {path}")
    return float(match.group(1))


def last_activity(
    commit_time: float | None,
    branch: str | None,
    common: Path,
    wt_gitdir: Path | None,
    wt_path: str | None,
) -> tuple[float, str]:
    candidates: list[tuple[float, str]] = []
    if commit_time is not None:
        candidates.append((commit_time, "tip commit date"))
    if branch is not None:
        t = _reflog_time(common / "logs" / "refs" / "heads" / branch)
        if t is not None:
            candidates.append((t, "branch reflog"))
    if wt_gitdir is not None:
        t = _reflog_time(wt_gitdir / "logs" / "HEAD")
        if t is not None:
            candidates.append((t, "worktree HEAD reflog"))
        for name in ("HEAD", "index"):
            t = _mtime(wt_gitdir / name)
            if t is not None:
                candidates.append((t, f"worktree {name} mtime"))
    if wt_path is not None:
        t = _mtime(Path(wt_path))
        if t is not None:
            candidates.append((t, "worktree directory"))
    if not candidates:
        raise CleanupError(
            f"no activity timestamp for branch={branch} worktree={wt_path}"
        )
    return max(candidates)


@dataclass(frozen=True)
class DiskUsage:
    bytes: int | None
    error: str | None


def disk_usage(path: str) -> DiskUsage:
    proc = subprocess.run(
        ["du", "-s", "-x", "-B1", path], text=True, capture_output=True, check=False
    )
    size: int | None = None
    if proc.stdout.strip():
        size = int(proc.stdout.split()[0])
    error = None
    if proc.returncode != 0:
        error = (proc.stderr.strip().splitlines() or [f"du exited {proc.returncode}"])[
            -1
        ]
    return DiskUsage(bytes=size, error=error)


# ------------------------------------------------------------------------ entries


@dataclass
class Entry:
    id: str
    facts: Facts
    decision: Decision
    prs: list[PrInfo]
    last_activity: float
    activity_source: str
    disk: DiskUsage | None = None
    action: str = "none"
    action_error: str | None = None


@dataclass
class Inventory:
    repo: Path
    entries: list[Entry]
    warnings: list[str]
    prs: dict[str, list[PrInfo]]


def _worktree_for_cwd(
    worktrees: Sequence[WorktreeInfo], cwd: Path
) -> WorktreeInfo | None:
    real = str(cwd.resolve())
    best: WorktreeInfo | None = None
    for wt in worktrees:
        if path_is_within(real, wt.path) and (
            best is None or len(wt.path) > len(best.path)
        ):
            best = wt
    return best


def build_inventory(
    repo: Path,
    cfg: Config,
    *,
    invoking_cwd: Path,
    measure_disk: bool,
    now: float | None = None,
    only_ids: set[str] | None = None,
    prs: dict[str, list[PrInfo]] | None = None,
) -> Inventory:
    """Collect facts and classify every local branch and linked worktree."""
    now = time.time() if now is None else now
    warnings: list[str] = []
    worktrees = list_worktrees(repo)
    branches = list_branches(repo)
    common = common_dir(repo)
    if prs is None:
        prs = load_prs(repo, cfg.pr_limit)
    in_default = merged_into(repo, cfg.default_branch_refs)
    queue_jobs = load_queue_jobs(
        (repo / cfg.queue_dir)
        if not Path(cfg.queue_dir).is_absolute()
        else Path(cfg.queue_dir)
    )
    procs = scan_processes()
    for item in procs.unreadable:
        warnings.append(f"process scan could not inspect {item}")

    by_branch: dict[str, list[WorktreeInfo]] = {}
    for w in worktrees:
        if w.branch is not None:
            by_branch.setdefault(w.branch, []).append(w)
    invoking = _worktree_for_cwd(worktrees, invoking_cwd)
    main = worktrees[0]

    targets: list[tuple[str, BranchInfo | None, WorktreeInfo | None]] = []
    for b in branches:
        checked_out = by_branch.get(b.name)
        targets.append((f"branch:{b.name}", b, checked_out[0] if checked_out else None))
    for w in worktrees:
        if w.branch is None:
            targets.append((f"worktree:{w.path}", None, w))
    if only_ids is not None:
        targets = [t for t in targets if t[0] in only_ids]

    live_paths = [
        t[2].path
        for t in targets
        if t[2] is not None and not t[2].is_main and not t[2].prunable
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        state_jobs = {p: pool.submit(worktree_state, p, cfg) for p in live_paths}
        gitdir_jobs = {p: pool.submit(worktree_gitdir, p) for p in live_paths}
        disk_jobs = (
            {p: pool.submit(disk_usage, p) for p in live_paths} if measure_disk else {}
        )
        states = {p: f.result() for p, f in state_jobs.items()}
        gitdirs = {p: f.result() for p, f in gitdir_jobs.items()}
        disks = {p: f.result() for p, f in disk_jobs.items()}

    entries: list[Entry] = []
    for entry_id, br, wt in targets:
        tip = br.tip if br is not None else (wt.head if wt is not None else None)
        if tip is None:
            raise CleanupError(f"{entry_id}: no commit to inspect")
        name = br.name if br is not None else None
        undetermined: list[str] = []
        if name is not None and len(by_branch.get(name, [])) > 1:
            undetermined.append(
                f"branch checked out in several worktrees: {[w.path for w in by_branch[name]]}"
            )

        current = None
        if wt is not None and wt.path == main.path:
            current = None  # covered by main_worktree
        elif name is not None and main.branch == name:
            current = "checked out in the main checkout"
        if (
            wt is not None
            and invoking is not None
            and wt.path == invoking.path
            and not wt.is_main
        ):
            current = "worktree containing the invoking working directory"

        branch_prs = prs.get(name, []) if name is not None else []
        merged: list[MergedPr] = []
        for pr in branch_prs:
            if pr.state == "MERGED":
                relation, count = merge_relation(repo, tip, pr.head_oid)
                merged.append(MergedPr(pr.number, pr.head_oid, relation, count))
        ancestor = (
            in_default.get(name)
            if name is not None
            else commit_contained(repo, tip, cfg.default_branch_refs)
        )

        live = wt is not None and not wt.is_main and not wt.prunable
        activity, source = last_activity(
            br.commit_time if br is not None else None,
            name,
            common,
            gitdirs.get(wt.path) if live and wt is not None else None,
            wt.path if live and wt is not None else None,
        )
        facts = Facts(
            branch=name,
            tip=tip,
            worktree=wt.path if wt is not None else None,
            is_main_worktree=wt is not None and wt.is_main,
            worktree_locked=wt.locked if wt is not None else None,
            worktree_missing=wt is not None and wt.prunable,
            worktree_state=states.get(wt.path) if wt is not None else None,
            protected_branch=name is not None
            and (
                name in cfg.protected_branches
                or matches_any(name, cfg.protected_branch_globs)
            ),
            protected_worktree=wt is not None
            and matches_any(wt.path, cfg.protected_worktree_globs),
            current_checkout=current,
            process_refs=tuple(procs.refs_for(wt.path))
            if live and wt is not None
            else (),
            queue_refs=tuple(queue_refs_for(queue_jobs, wt.path))
            if live and wt is not None
            else (),
            open_prs=tuple(p.number for p in branch_prs if p.state == "OPEN"),
            closed_prs=tuple(p.number for p in branch_prs if p.state == "CLOSED"),
            merged_prs=tuple(merged),
            ancestor_of=ancestor,
            idle_hours=max(0.0, (now - activity) / 3600.0),
            undetermined=tuple(undetermined),
        )
        entries.append(
            Entry(
                id=entry_id,
                facts=facts,
                decision=classify(facts, cfg.policy),
                prs=branch_prs,
                last_activity=activity,
                activity_source=source,
                disk=disks.get(wt.path) if wt is not None else None,
            )
        )
    return Inventory(repo=repo, entries=entries, warnings=warnings, prs=prs)


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec="seconds")
