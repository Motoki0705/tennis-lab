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
    BranchFacts,
    Decision,
    MergedPr,
    MergeRelation,
    Policy,
    WorktreeFacts,
    WorktreeState,
    classify_branch,
    classify_worktree,
    matches_any,
)

# The only untracked/ignored entry a removable worktree may contain: the
# top-level `.venv` symlink to the shared environment (unlinking it never
# touches its target).
DISPOSABLE_SYMLINK = ".venv"


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
    symref: str  # full target ref when refs/heads/<name> is a symbolic ref, else ""


def list_branches(repo: Path) -> list[BranchInfo]:
    fmt = "%(refname:lstrip=2)%00%(objectname)%00%(committerdate:unix)%00%(symref)"
    out = git(repo, "for-each-ref", f"--format={fmt}", "refs/heads").stdout
    branches = []
    for line in out.splitlines():
        name, tip, ts, symref = line.split("\0")
        branches.append(
            BranchInfo(name=name, tip=tip, commit_time=float(ts), symref=symref)
        )
    return branches


def symref_reasons(branches: Sequence[BranchInfo]) -> dict[str, list[str]]:
    """Branch -> why it is involved in a symbolic ref (as the alias or its target).

    Ref deletion follows symbolic refs, and deleting a target leaves a dangling
    alias, so both sides are protected.
    """
    found: dict[str, list[str]] = {}
    for b in branches:
        if not b.symref:
            continue
        found.setdefault(b.name, []).append(
            f"refs/heads/{b.name} is a symbolic ref to {b.symref}"
        )
        if b.symref.startswith("refs/heads/"):
            target = b.symref[len("refs/heads/") :]
            found.setdefault(target, []).append(
                f"target of symbolic ref refs/heads/{b.name}"
            )
    return found


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


def common_dir(repo: Path) -> Path:
    return Path(
        git(
            repo, "rev-parse", "--path-format=absolute", "--git-common-dir"
        ).stdout.strip()
    )


def worktree_admin_dirs(
    common: Path, worktrees: Sequence[WorktreeInfo]
) -> dict[str, Path]:
    """Worktree path -> its git administrative dir (``.git`` or ``.git/worktrees/<id>``).

    Read from ``<common>/worktrees/*/gitdir`` (as git does) so it also works for
    worktrees whose directory is gone.
    """
    linked: dict[str, Path] = {}
    base = common / "worktrees"
    if base.is_dir():
        for admin in sorted(base.iterdir()):
            try:
                raw = (admin / "gitdir").read_text(encoding="utf-8").strip()
            except (FileNotFoundError, NotADirectoryError):
                continue
            gitfile = Path(raw) if Path(raw).is_absolute() else admin / raw
            wt_path = (
                str(gitfile)[: -len("/.git")]
                if str(gitfile).endswith("/.git")
                else str(gitfile)
            )
            linked[os.path.realpath(wt_path)] = admin
    result: dict[str, Path] = {}
    for w in worktrees:
        if w.is_main:
            result[w.path] = common
            continue
        admin_dir = linked.get(os.path.realpath(w.path))
        if admin_dir is None:
            raise CleanupError(
                f"no administrative directory under {base} for worktree {w.path}"
            )
        result[w.path] = admin_dir
    return result


@dataclass(frozen=True)
class InProgress:
    operations: tuple[
        str, ...
    ]  # rebase, am, bisect, merge, cherry-pick, revert, sequencer
    held: tuple[tuple[str, str], ...]  # (local branch, operation holding it)


def _read_state_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def in_progress(admin: Path, branch_names: set[str]) -> InProgress:
    """Operations in progress in one worktree and the branches they hold.

    Mirrors git's own "branch is in use" check (``branch.c``
    ``prepare_checked_out_branches``): a rebase holds the branch named in
    ``rebase-{merge,apply}/head-name`` plus ``rebase-merge/update-refs``, and a
    bisect holds the branch named in ``BISECT_START``. HEAD is detached while
    these run, so ``git worktree list`` does not show those branches.
    """
    ops: list[str] = []
    held: list[tuple[str, str]] = []
    for sub in ("rebase-merge", "rebase-apply"):
        state_dir = admin / sub
        if not state_dir.is_dir():
            continue
        if sub == "rebase-apply" and (state_dir / "applying").exists():
            ops.append("am")
            continue
        ops.append("rebase")
        head = _read_state_file(state_dir / "head-name")
        if head is None:
            if state_dir.is_dir():
                raise CleanupError(
                    f"{state_dir} exists without head-name; cannot tell which branch the rebase holds"
                )
            continue  # finished while scanning
        if head.startswith("refs/heads/"):
            held.append((head[len("refs/heads/") :], "rebase"))
        elif head != "detached HEAD":
            raise CleanupError(
                f"unexpected {state_dir / 'head-name'} content: {head!r}"
            )
        for line in (_read_state_file(state_dir / "update-refs") or "").splitlines():
            if line.startswith("refs/heads/"):
                held.append((line[len("refs/heads/") :], "rebase --update-refs"))
    if (admin / "BISECT_LOG").exists():
        ops.append("bisect")
        start = _read_state_file(admin / "BISECT_START") or ""
        name = start.removeprefix("refs/heads/")
        if name in branch_names:  # otherwise BISECT_START holds a detached commit id
            held.append((name, "bisect"))
    for marker, op in (
        ("MERGE_HEAD", "merge"),
        ("CHERRY_PICK_HEAD", "cherry-pick"),
        ("REVERT_HEAD", "revert"),
    ):
        if (admin / marker).exists():
            ops.append(op)
    if (admin / "sequencer").is_dir():
        ops.append("sequencer")
    return InProgress(operations=tuple(ops), held=tuple(held))


def worktree_state(path: str) -> WorktreeState:
    """Everything ``git status --ignored`` reports, plus populated submodules and
    index entries whose edits ``git status`` hides.

    Nothing is treated as disposable except the top-level ``.venv`` symlink.
    Submodules are not inspected recursively: a populated one already blocks.
    """
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
    disposable: list[str] = []
    i = 0
    while i < len(fields):
        entry = fields[i]
        i += 1
        if not entry:
            continue
        xy, rel = entry[:2], entry[3:]
        if xy in ("??", "!!"):
            if rel == DISPOSABLE_SYMLINK and (wt / rel).is_symlink():
                disposable.append(rel)
            else:
                (untracked if xy == "??" else ignored).append(rel)
            continue
        if xy[0] in "RC":
            i += 1  # skip the rename/copy source path
        dirty.append(rel)
    hidden: list[str] = []
    for item in git(wt, "ls-files", "-v", "-z").stdout.split("\0"):
        if not item:
            continue
        tag, rel = item[0], item[2:]
        # lowercase tag = assume-unchanged; S/s = skip-worktree (see git ls-files -v)
        if tag.islower() or tag == "S":
            hidden.append(rel)
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
        disposable=tuple(disposable),
        submodule_populated=tuple(populated),
        hidden_index_flags=tuple(hidden),
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


def pushed_evidence(repo: Path, head: str, prs: Sequence[PrInfo]) -> str | None:
    """Why ``head`` exists outside this machine, or None (unpushed commits).

    Evidence: a remote-tracking ref contains it, or it is (an ancestor of) the
    head of one of the branch's PRs. Local refs such as refs/heads/main do not
    count.
    """
    if not has_commit(repo, head):
        return None
    remote = git(
        repo,
        "for-each-ref",
        "--count=1",
        f"--contains={head}",
        "--format=%(refname)",
        "refs/remotes",
    ).stdout.strip()
    if remote:
        return f"contained in {remote}"
    for pr in prs:
        if pr.head_oid == head:
            return f"PR #{pr.number} head"
        if has_commit(repo, pr.head_oid) and is_ancestor(repo, head, pr.head_oid):
            return f"ancestor of PR #{pr.number} head"
    return None


# ----------------------------------------------------------- processes and queue


def component_pattern(name: str) -> re.Pattern[str]:
    """``name`` used as a path component: ``name/...`` or ``.../name``."""
    n = re.escape(name)
    return re.compile(rf"(?<![\w.\-]){n}/|/{n}(?![\w.\-])")


def token_pattern(name: str) -> re.Pattern[str]:
    """``name`` as a whole token (not part of a longer name)."""
    n = re.escape(name)
    return re.compile(rf"(?<![\w.\-]){n}(?![\w.\-])")


def _relative_candidates(word: str) -> list[str]:
    """Non-empty relative paths in a word: the word itself or a KEY=/--opt= value."""
    _, sep, value = word.partition("=")
    return [
        c for c in ((word, value) if sep else (word,)) if c and not c.startswith("/")
    ]


@dataclass
class ProcessScan:
    paths: list[tuple[str, str]] = field(
        default_factory=list
    )  # (path, holder description)
    argvs: list[tuple[tuple[str, ...], str | None, str]] = field(
        default_factory=list
    )  # (argv, cwd if readable, holder description)
    unreadable: list[str] = field(default_factory=list)

    def refs_for(self, worktree: str) -> list[str]:
        """Holders whose cwd/root/fd lies in ``worktree`` or whose argv names it.

        argv counts when it contains the absolute path, a relative path that
        resolves into the worktree from the process's cwd, or (conservatively,
        since argv may be relative to a directory we cannot see) the worktree's
        directory name as a path component, e.g. ``.claude/worktrees/<name>/x``.
        """
        root = worktree.rstrip("/")
        hits: list[str] = []
        pattern = re.compile(re.escape(root) + r"(?![\w.\-])")
        component = component_pattern(os.path.basename(root))
        for target, desc in self.paths:
            if path_is_within(target, root):
                hits.append(desc)
        for argv, cwd, desc in self.argvs:
            for arg in argv:
                if pattern.search(arg) or component.search(arg):
                    hits.append(f"{desc} argv")
                    break
                if cwd is not None and any(
                    path_is_within(os.path.normpath(os.path.join(cwd, c)), root)
                    for c in _relative_candidates(arg)
                ):
                    hits.append(f"{desc} argv (relative to cwd {cwd})")
                    break
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
        hidden: list[str] = []
        try:
            cwd: str | None = os.readlink(entry / "cwd").removesuffix(" (deleted)")
        except (FileNotFoundError, ProcessLookupError):
            cwd = None
        except PermissionError:
            cwd = None
            hidden.append("cwd")
        scan.argvs.append((tuple(argv), cwd, desc))
        if cwd is not None and cwd.startswith("/"):
            scan.paths.append((cwd, f"{desc} cwd"))
        links: list[Path] = [entry / "root"]
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


_ANSI_C_ESCAPES = {
    "a": b"\a",
    "b": b"\b",
    "e": b"\x1b",
    "E": b"\x1b",
    "f": b"\f",
    "n": b"\n",
    "r": b"\r",
    "t": b"\t",
    "v": b"\v",
    "\\": b"\\",
    "'": b"'",
    '"': b'"',
    "?": b"?",
}
_WORD_BREAKS = frozenset(" \t\n;&|<>()")


def _ansi_c(text: str, i: int) -> tuple[str, int]:
    """Decode a bash ``$'...'`` body starting at ``text[i]``; return (value, index after ')."""
    out = bytearray()
    n = len(text)
    while i < n:
        c = text[i]
        if c == "'":
            return out.decode("utf-8", errors="surrogateescape"), i + 1
        if c != "\\":
            out += c.encode("utf-8", errors="surrogateescape")
            i += 1
            continue
        if i + 1 >= n:
            break
        e = text[i + 1]
        i += 2
        if e in _ANSI_C_ESCAPES:
            out += _ANSI_C_ESCAPES[e]
        elif e in "01234567":
            m = re.match(r"[0-7]{0,2}", text[i:])
            assert m is not None
            out.append(int(e + m.group(0), 8) & 0xFF)
            i += m.end()
        elif e in "xuU":
            width = {"x": 2, "u": 4, "U": 8}[e]
            m = re.match(rf"[0-9A-Fa-f]{{1,{width}}}", text[i:])
            if m is None:
                out += ("\\" + e).encode()
                continue
            value = int(m.group(0), 16)
            out += bytes([value]) if e == "x" else chr(value).encode("utf-8")
            i += m.end()
        elif e == "c" and i < n:
            out.append(ord(text[i]) & 0x1F)
            i += 1
        else:
            out += ("\\" + e).encode("utf-8", errors="surrogateescape")
    raise ValueError("unterminated $'...' quote")


def shell_words(text: str) -> list[str]:
    """Split shell source into words with quoting removed, without executing anything.

    Covers everything bash's ``printf %q`` emits (backslash escapes, ``'...'``,
    ``$'...'``) plus ``"..."``, line continuations and comments. Expansions
    (``$VAR``, ``$(...)``, globs) stay literal; operators split words. Raises
    ``ValueError`` on an unterminated quote.
    """
    words: list[str] = []
    cur: list[str] | None = None
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in _WORD_BREAKS:
            if cur is not None:
                words.append("".join(cur))
                cur = None
            i += 1
            continue
        if c == "#" and cur is None:
            end = text.find("\n", i)
            i = n if end < 0 else end
            continue
        if cur is None:
            cur = []
        if c == "\\":
            if i + 1 < n and text[i + 1] == "\n":
                i += 2  # line continuation
            elif i + 1 < n:
                cur.append(text[i + 1])
                i += 2
            else:
                cur.append(c)
                i += 1
        elif c == "'":
            end = text.find("'", i + 1)
            if end < 0:
                raise ValueError("unterminated '...' quote")
            cur.append(text[i + 1 : end])
            i = end + 1
        elif c == "$" and text.startswith("$'", i):
            value, i = _ansi_c(text, i + 2)
            cur.append(value)
        elif c == '"' or (c == "$" and text.startswith('$"', i)):
            i += 1 if c == '"' else 2
            while True:
                if i >= n:
                    raise ValueError('unterminated "..." quote')
                d = text[i]
                if d == '"':
                    i += 1
                    break
                if d == "\\" and i + 1 < n and text[i + 1] in '$`"\\\n':
                    if text[i + 1] != "\n":
                        cur.append(text[i + 1])
                    i += 2
                    continue
                cur.append(d)
                i += 1
        else:
            cur.append(c)
            i += 1
    if cur is not None:
        words.append("".join(cur))
    return words


_EXPANSION = re.compile(r"[$`]|^~")


def job_paths(words: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Every word of a job script (and each ``KEY=``/``--opt=`` value) as absolute
    paths, plus every directory the script ``cd``s into.

    Relative words are resolved against every directory the script ``cd``s or
    ``pushd``es into (a subshell or a conditional ``cd`` cannot be told apart
    statically, so all of them count). Words with expansions are skipped.
    Raises ``ValueError`` when a relative word or ``cd`` target appears before
    any absolute ``cd`` (the working directory is unknown) or when a ``cd``
    target cannot be resolved without running the script.
    """
    cwds: list[str] = []
    out: list[str] = []

    def resolve(word: str) -> list[str]:
        if word.startswith("/"):
            return [os.path.normpath(word)]
        if not cwds:
            raise ValueError(
                f"relative word {word!r} appears before the job's first absolute cd"
            )
        return [os.path.normpath(os.path.join(c, word)) for c in cwds]

    i = 0
    while i < len(words):
        word = words[i]
        if word in ("cd", "pushd") and i + 1 < len(words):
            target = words[i + 1]
            if not target or target.startswith("-") or _EXPANSION.search(target):
                raise ValueError(f"cannot resolve `{word} {target}` statically")
            for d in resolve(target):
                if d not in cwds:
                    cwds.append(d)
                out.append(d)
            i += 2
            continue
        _, sep, value = word.partition("=")
        for cand in (word, value) if sep else (word,):
            if cand and not _EXPANSION.search(cand):
                out.extend(resolve(cand))
        i += 1
    return tuple(dict.fromkeys(out)), tuple(cwds)


_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh"})


def nested_code(text: str, words: Sequence[str]) -> str | None:
    """The construct that makes a job run shell code held in a string, if any.

    Paths inside such code (``bash -lc 'cd /tmp/link && ...'``) cannot be resolved
    statically, so the caller treats the job as unparseable.
    """
    if "$(" in text:
        return "$(...)"
    if "`" in text:
        return "backticks"
    for i, word in enumerate(words):
        if word == "eval":
            return "eval"
        if (
            word in ("source", ".")
            and i + 1 < len(words)
            and _EXPANSION.search(words[i + 1])
        ):
            return f"{word} {words[i + 1]}"
        if os.path.basename(word) in _SHELLS:
            for opt in words[i + 1 :]:
                if not opt.startswith("-"):
                    break
                if "c" in opt.lstrip("-") and not opt.startswith("--"):
                    return f"{word} {opt}"
    return None


@dataclass(frozen=True)
class QueueJob:
    label: str
    text: str
    words: tuple[str, ...]  # the job script's shell words, quoting removed
    paths: tuple[str, ...]  # words resolved to absolute paths (see job_paths)
    cd_targets: tuple[str, ...]  # directories the script cd/pushd-es into
    nested: str | None  # construct running code we cannot resolve (sh -c, eval, ...)


def load_queue_jobs(queue_dir: Path) -> list[QueueJob]:
    """Pending (``jobs/``) and running (``running/``) training-queue job files.

    ``training_queue.sh add`` writes ``cd <job cwd>`` and paths with ``printf %q``
    (e.g. ``cd /a\\ b``), so jobs are matched on decoded shell words, with
    relative words resolved against the job's ``cd`` targets, not on raw text.
    """
    jobs: list[QueueJob] = []
    for state in ("jobs", "running"):
        directory = queue_dir / state
        if not directory.is_dir():
            raise CleanupError(f"training queue directory missing: {directory}")
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="surrogateescape")
            except FileNotFoundError:
                continue  # job moved to done/ while scanning
            try:
                words = tuple(shell_words(text))
                paths, cd_targets = job_paths(words)
            except ValueError as exc:
                raise CleanupError(
                    f"cannot parse training-queue job {path}: {exc}"
                ) from exc
            match = re.search(r"^# name: (.*)$", text, flags=re.MULTILINE)
            name = match.group(1).strip() if match else path.name
            jobs.append(
                QueueJob(
                    label=f"{state}/{name}",
                    text=text,
                    words=words,
                    paths=paths,
                    cd_targets=cd_targets,
                    nested=nested_code(text, words),
                )
            )
    return jobs


def unparseable_jobs(jobs: Sequence[QueueJob], roots: Sequence[str]) -> list[str]:
    """Jobs whose worktree references cannot be resolved with certainty.

    A job is unparseable when it runs nested shell code, or ``cd``s outside the
    main checkout and every worktree (``roots``), where a symlink or script could
    lead back into any worktree. While one exists, no worktree is removed.
    """
    real_roots = [os.path.realpath(r) for r in roots]
    found: list[str] = []
    for job in jobs:
        if job.nested is not None:
            found.append(f"{job.label}: nested shell code ({job.nested})")
            continue
        outside = [
            d
            for d in job.cd_targets
            if not any(path_is_within(os.path.realpath(d), r) for r in real_roots)
        ]
        if outside:
            found.append(f"{job.label}: cd outside the repository ({outside[0]})")
    return found


def queue_refs_for(jobs: Sequence[QueueJob], worktree: str) -> list[str]:
    """Jobs that may use ``worktree``.

    Besides exact resolution, any mention of the worktree's directory name as a
    path component counts, and in jobs with nested shell code (``bash -c``,
    ``eval``, ``$(...)``) any mention of the name as a word counts.
    """
    root = worktree.rstrip("/")
    real_root = os.path.realpath(root)
    pattern = re.compile(re.escape(root) + r"(?![\w.\-])")
    name = os.path.basename(root)
    component, token = component_pattern(name), token_pattern(name)
    refs: list[str] = []
    for job in jobs:
        mention = token if job.nested else component
        hit = any(
            pattern.search(t) or mention.search(t) for t in (job.text, *job.words)
        )
        if not hit:
            # Relative words (resolved against the job's cd) and paths spelled
            # through symlinks (e.g. a logical $PWD) count too.
            hit = any(
                path_is_within(p, root)
                or path_is_within(os.path.realpath(p), real_root)
                for p in job.paths
            )
        if hit:
            refs.append(job.label)
    return refs


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


def commit_time(repo: Path, oid: str) -> float | None:
    proc = git(repo, "show", "-s", "--format=%ct", oid, ok_codes=(0, 128))
    return float(proc.stdout.strip()) if proc.returncode == 0 else None


def last_activity(
    commit_time: float | None,
    branch: str | None,
    common: Path,
    admin_dir: Path | None,
    wt_path: str | None,
) -> tuple[float, str] | None:
    """Newest activity timestamp and its source, or None when there is none."""
    candidates: list[tuple[float, str]] = []
    if commit_time is not None:
        candidates.append((commit_time, "tip commit date"))
    if branch is not None:
        t = _reflog_time(common / "logs" / "refs" / "heads" / branch)
        if t is not None:
            candidates.append((t, "branch reflog"))
    if admin_dir is not None:
        t = _reflog_time(admin_dir / "logs" / "HEAD")
        if t is not None:
            candidates.append((t, "worktree HEAD reflog"))
        for name in ("HEAD", "index"):
            t = _mtime(admin_dir / name)
            if t is not None:
                candidates.append((t, f"worktree {name} mtime"))
    if wt_path is not None:
        t = _mtime(Path(wt_path))
        if t is not None:
            candidates.append((t, "worktree directory"))
    return max(candidates) if candidates else None


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
    facts: BranchFacts | WorktreeFacts
    decision: Decision
    prs: list[PrInfo]
    last_activity: float | None
    activity_source: str
    disk: DiskUsage | None = None
    action: str = "none"
    action_error: str | None = None

    @property
    def branch(self) -> str | None:
        return self.facts.branch

    @property
    def tip(self) -> str:
        f = self.facts
        return f.tip if isinstance(f, BranchFacts) else f.head

    @property
    def worktree(self) -> str | None:
        f = self.facts
        return f.worktree if isinstance(f, WorktreeFacts) else None

    @property
    def state(self) -> WorktreeState | None:
        f = self.facts
        return f.state if isinstance(f, WorktreeFacts) else None


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


def _activity(
    found: tuple[float, str] | None, now: float, undetermined: list[str]
) -> tuple[float | None, str, float]:
    if found is None:
        undetermined.append("no activity timestamp (commit, reflog or mtime)")
        return None, "none", 0.0
    activity, source = found
    return activity, source, max(0.0, (now - activity) / 3600.0)


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
    """Collect facts and classify every local branch and every linked worktree."""
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
    blockers = unparseable_jobs(queue_jobs, [w.path for w in worktrees])
    for item in blockers:
        warnings.append(
            f"unparseable training-queue job blocks worktree removal: {item}"
        )
    procs = scan_processes()
    for item in procs.unreadable:
        warnings.append(f"process scan could not inspect {item}")

    by_branch: dict[str, list[WorktreeInfo]] = {}
    for w in worktrees:
        if w.branch is not None:
            by_branch.setdefault(w.branch, []).append(w)
    branch_names = {b.name for b in branches}
    symrefs = symref_reasons(branches)
    admin_dirs = worktree_admin_dirs(common, worktrees)
    ops_by_worktree = {
        w.path: in_progress(admin_dirs[w.path], branch_names) for w in worktrees
    }
    held_by: dict[str, list[str]] = {}
    for w in worktrees:
        for held_branch, op in ops_by_worktree[w.path].held:
            held_by.setdefault(held_branch, []).append(
                f"{op} in progress in worktree {w.path}"
            )
    invoking = _worktree_for_cwd(worktrees, invoking_cwd)
    main = worktrees[0]

    def wanted(entry_id: str) -> bool:
        return only_ids is None or entry_id in only_ids

    target_branches = [b for b in branches if wanted(f"branch:{b.name}")]
    target_worktrees = [w for w in worktrees[1:] if wanted(f"worktree:{w.path}")]
    live_paths = [w.path for w in target_worktrees if not w.prunable]
    with ThreadPoolExecutor(max_workers=8) as pool:
        state_jobs = {p: pool.submit(worktree_state, p) for p in live_paths}
        disk_jobs = (
            {p: pool.submit(disk_usage, p) for p in live_paths} if measure_disk else {}
        )
        states = {p: f.result() for p, f in state_jobs.items()}
        disks = {p: f.result() for p, f in disk_jobs.items()}

    entries: list[Entry] = []
    for b in target_branches:
        undetermined: list[str] = []
        branch_prs = prs.get(b.name, [])
        merged: list[MergedPr] = []
        for pr in branch_prs:
            if pr.state == "MERGED":
                relation, count = merge_relation(repo, b.tip, pr.head_oid)
                merged.append(MergedPr(pr.number, pr.head_oid, relation, count))
        activity, source, idle = _activity(
            last_activity(b.commit_time, b.name, common, None, None), now, undetermined
        )
        bfacts = BranchFacts(
            branch=b.name,
            tip=b.tip,
            protected_branch=b.name in cfg.protected_branches
            or matches_any(b.name, cfg.protected_branch_globs),
            current_checkout="checked out in the main checkout"
            if main.branch == b.name
            else None,
            checked_out_in=tuple(
                w.path for w in by_branch.get(b.name, []) if not w.is_main
            ),
            symbolic_ref=tuple(symrefs.get(b.name, [])),
            branch_in_use=tuple(held_by.get(b.name, [])),
            open_prs=tuple(p.number for p in branch_prs if p.state == "OPEN"),
            closed_prs=tuple(p.number for p in branch_prs if p.state == "CLOSED"),
            merged_prs=tuple(merged),
            ancestor_of=in_default.get(b.name),
            idle_hours=idle,
            undetermined=tuple(undetermined),
        )
        entries.append(
            Entry(
                id=f"branch:{b.name}",
                facts=bfacts,
                decision=classify_branch(bfacts, cfg.policy),
                prs=branch_prs,
                last_activity=activity,
                activity_source=source,
            )
        )
    for w in target_worktrees:
        if w.head is None:
            raise CleanupError(f"worktree {w.path}: no HEAD commit to inspect")
        undetermined = []
        alive = not w.prunable
        branch_prs = prs.get(w.branch, []) if w.branch is not None else []
        activity, source, idle = _activity(
            last_activity(
                commit_time(repo, w.head),
                w.branch,
                common,
                admin_dirs[w.path],
                w.path if alive else None,
            ),
            now,
            undetermined,
        )
        wfacts = WorktreeFacts(
            worktree=w.path,
            head=w.head,
            branch=w.branch,
            locked=w.locked,
            missing=w.prunable,
            state=states.get(w.path),
            protected_worktree=matches_any(w.path, cfg.protected_worktree_globs),
            current_checkout="worktree containing the invoking working directory"
            if invoking is not None and invoking.path == w.path
            else None,
            operations_in_progress=ops_by_worktree[w.path].operations,
            process_refs=tuple(procs.refs_for(w.path)) if alive else (),
            queue_refs=tuple(queue_refs_for(queue_jobs, w.path)) if alive else (),
            queue_unparseable=tuple(blockers) if alive else (),
            pushed=pushed_evidence(repo, w.head, branch_prs),
            idle_hours=idle,
            undetermined=tuple(undetermined),
        )
        entries.append(
            Entry(
                id=f"worktree:{w.path}",
                facts=wfacts,
                decision=classify_worktree(wfacts, cfg.policy),
                prs=branch_prs,
                last_activity=activity,
                activity_source=source,
                disk=disks.get(w.path),
            )
        )
    return Inventory(repo=repo, entries=entries, warnings=warnings, prs=prs)


def iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec="seconds")
