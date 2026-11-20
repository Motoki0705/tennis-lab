"""Git object and index access for isolated shared-memory publication."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from .layout import DEDICATED_PATH, is_shared_path
from .tree import Edits, FileKind, resolve_links

_LOCAL_GIT_ENV = {
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
    "GIT_DIR",
    "GIT_GRAFT_FILE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_NAMESPACE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_PREFIX",
    "GIT_REPLACE_REF_BASE",
    "GIT_SHALLOW_FILE",
    "GIT_WORK_TREE",
}
_KINDS: dict[bytes, FileKind] = {
    b"040000": "directory",
    b"100644": "file",
    b"100755": "file",
    b"120000": "symlink",
    b"160000": "other",  # submodule contents are a separate repository
}
_STATUSES = {
    b"A": (b":000000", b"100644", "added"),
    b"M": (b":100644", b"100644", "modified"),
    b"D": (b":100644", b"000000", "removed"),
}


def is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and value != "0" * 40
        and re.fullmatch(r"[0-9a-f]{40}", value) is not None
    )


def _sha(raw: bytes) -> str:
    value = raw.decode("ascii").strip()
    if not is_sha(value):
        raise ValueError("Git did not return a full commit or object SHA")
    return value


class GitError(ValueError):
    def __init__(self, args: tuple[str, ...], status: int) -> None:
        self.status = status
        super().__init__(f"git {args[0]} failed (exit {status})")


class Git:
    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd.resolve()
        self.env = {k: v for k, v in os.environ.items() if k not in _LOCAL_GIT_ENV}

    def run(self, *args: str, input: bytes | None = None) -> bytes:
        try:
            result = subprocess.run(
                [
                    "git",
                    "--no-replace-objects",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "-c",
                    "core.sparseCheckout=false",
                    "-c",
                    "index.sparse=false",
                    *args,
                ],
                cwd=self.cwd,
                env=self.env,
                input=input,
                capture_output=True,
                check=False,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            raise ValueError(f"git {args[0]} timed out") from None
        if result.returncode:
            raise GitError(args, result.returncode)
        return result.stdout

    def remote_main(self, remote: str) -> str:
        ref = "refs/heads/main"
        result = self.run("ls-remote", "--refs", "--exit-code", "--", remote, ref)
        sha, separator, found = result.partition(b"\t")
        if separator != b"\t" or found != ref.encode() + b"\n":
            raise ValueError("remote did not return exactly one main ref")
        return _sha(sha)

    def fetch_main(self, remote: str) -> str:
        sha = self.remote_main(remote)
        self.run(
            "fetch",
            "--no-tags",
            "--no-prune",
            "--no-prune-tags",
            "--no-recurse-submodules",
            "--no-write-fetch-head",
            "--refmap=",
            "--",
            remote,
            sha,
        )
        if self.run("cat-file", "-t", sha).strip() != b"commit":
            raise ValueError("remote main did not identify a commit")
        return sha

    def is_ancestor(self, previous: str, current: str) -> bool:
        try:
            self.run("merge-base", "--is-ancestor", previous, current)
        except GitError as exc:
            if exc.status != 1:
                raise
            return False
        return True

    def build_commit(self, base: str, edits: Edits) -> str:
        """Build new trees bottom-up without changing any checkout, index or ref."""
        updates: dict[str, dict[bytes, bytes | None]] = {}
        for path, content in sorted(edits.items()):
            directory, _, name = path.rpartition("/")
            metadata = None
            if content is not None:
                sha = _sha(self.run("hash-object", "-w", "--stdin", input=content))
                metadata = f"100644 blob {sha}".encode()
            updates.setdefault(directory, {})[name.encode()] = metadata
            while directory:
                directory = directory.rpartition("/")[0]
                updates.setdefault(directory, {})

        directories = sorted(
            updates, key=lambda path: len(PurePosixPath(path).parts), reverse=True
        )
        for directory in directories:
            if directory:
                tree = self._write_tree(f"{base}:{directory}", updates[directory])
                parent, _, name = directory.rpartition("/")
                updates[parent][name.encode()] = f"040000 tree {tree}".encode()
        tree = self._write_tree(base, updates[""])
        return _sha(
            self.run("commit-tree", tree, "-p", base, "-m", "共有memoryを更新する")
        )

    def _write_tree(self, source: str, edits: dict[bytes, bytes | None]) -> str:
        records = {}
        for row in self.run("ls-tree", "-z", source).split(b"\0")[:-1]:
            old_metadata, name = row.split(b"\t", 1)
            records[name] = old_metadata
        for name, metadata in edits.items():
            if metadata is None:
                del records[name]
            else:
                records[name] = metadata
        raw = b"".join(
            metadata + b"\t" + name + b"\0" for name, metadata in records.items()
        )
        return _sha(self.run("mktree", "-z", input=raw))

    def push(self, remote: str, branch: str, sha: str, *, expected: str) -> None:
        ref = f"refs/heads/{branch}"
        self.run(
            "push",
            "--no-follow-tags",
            "--recurse-submodules=no",
            f"--force-with-lease={ref}:{expected}",
            "--",
            remote,
            f"{sha}:{ref}",
        )

    def github_repo(self, remote: str) -> str:
        fetch = self.run("remote", "get-url", "--all", "--", remote).splitlines()
        push = self.run(
            "remote", "get-url", "--push", "--all", "--", remote
        ).splitlines()
        if len(fetch) != 1 or len(push) != 1:
            raise ValueError("remote must have one fetch URL and one push URL")
        source, target = (_github_repo(url.decode("utf-8")) for url in [*fetch, *push])
        if source.casefold() != target.casefold():
            raise ValueError(
                "remote fetch and push URLs identify different repositories"
            )
        return source


def _github_repo(url: str) -> str:
    if match := re.fullmatch(r"git@github\.com:(.+)", url, re.IGNORECASE):
        path = match[1]
    else:
        try:
            parsed = urlsplit(url)
            valid_transport = (
                parsed.scheme == "https"
                and parsed.username is None
                and parsed.port in (None, 443)
            ) or (
                parsed.scheme == "ssh"
                and parsed.username == "git"
                and parsed.port in (None, 22)
            )
            if (
                not valid_transport
                or parsed.hostname != "github.com"
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError()
        except ValueError:
            raise ValueError("expected a github.com HTTPS or Git SSH remote") from None
        path = parsed.path.removeprefix("/")
    repo = path.removesuffix(".git")
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+", repo
    ) or repo.split("/")[-1] in {".", ".."}:
        raise ValueError("GitHub remote path must identify owner/repo")
    return repo


class GitTree:
    """Read only paths and bytes present in a pinned Git tree, including symlinks."""

    def __init__(self, git: Git, revision: str) -> None:
        self.git = git
        self.nodes: dict[str, tuple[FileKind, str]] = {".": ("directory", "")}
        self.directories: dict[str, list[str]] = {".": []}
        self.blobs: dict[str, bytes] = {}
        for row in git.run("ls-tree", "-rtz", "--full-tree", revision).split(b"\0")[
            :-1
        ]:
            metadata, name = row.split(b"\t", 1)
            mode, _, oid = metadata.split()
            path = os.fsdecode(name)
            if path.startswith(f"{DEDICATED_PATH}/"):
                continue
            self.nodes[path] = (_KINDS[mode], _sha(oid))
            parent, _, filename = path.rpartition("/")
            self.directories.setdefault(parent or ".", []).append(filename)

    def _kind(self, path: str) -> FileKind | None:
        node = self.nodes.get(path)
        return node[0] if node is not None else None

    def _blob(self, path: str) -> bytes:
        _, oid = self.nodes[path]
        if oid not in self.blobs:
            self.blobs[oid] = self.git.run("cat-file", "blob", oid)
        return self.blobs[oid]

    def _read_link(self, path: str) -> str:
        target = os.fsdecode(self._blob(path))
        if not target or "\0" in target:
            raise ValueError("invalid symlink target")
        return target

    def resolve(self, path: str) -> str:
        return resolve_links(path, self._kind, self._read_link)

    def kind(self, path: str, *, follow_links: bool = True) -> FileKind | None:
        try:
            path = resolve_links(
                path, self._kind, self._read_link, follow_links=follow_links
            )
        except (FileNotFoundError, NotADirectoryError):
            return None
        return self._kind(path)

    def children(self, path: str) -> tuple[str, ...]:
        return tuple(sorted(self.directories.get(self.resolve(path), ())))

    def read_bytes(self, path: str) -> bytes:
        return self._blob(self.resolve(path))


def diff_changes(raw: bytes) -> dict[str, str]:
    """Accept only A/M/D records for allowed paths and regular 100644 blobs."""
    records = raw.split(b"\0")
    if records[-1] or len(records) % 2 != 1:
        raise ValueError("malformed committed diff")
    changes = {}
    for metadata, name in zip(records[:-1:2], records[1::2], strict=True):
        fields = metadata.split()
        status = _STATUSES.get(fields[-1]) if len(fields) == 5 else None
        path = os.fsdecode(name)
        if status is None or tuple(fields[:2]) != status[:2]:
            raise ValueError("unsafe file mode or status in committed diff")
        if not is_shared_path(path) or path in changes:
            raise ValueError("unexpected path in committed diff")
        changes[path] = status[2]
    if not changes:
        raise ValueError("committed diff is empty (no-op)")
    return changes
