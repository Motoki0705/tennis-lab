"""Private local state and git evidence for the headless runner."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
import subprocess
from pathlib import Path
from typing import Any


class RunError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def redact(data: bytes, secrets: list[bytes]) -> bytes:
    for secret in sorted(
        {secret for secret in secrets if secret}, key=len, reverse=True
    ):
        data = data.replace(secret, b"<redacted-api-key>")
    return data


def private_write(path: Path, data: bytes, secrets: list[bytes] | None = None) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(redact(data, secrets or []))


def private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise RunError(
            "UNSAFE_DIRECTORY",
            "Private state directories must be owned by you, nonsymlink directories with mode 0700.",
        )
    return path


def git(worktree: Path, *arguments: str) -> bytes:
    environment = {
        name: os.environ[name]
        for name in ("PATH", "HOME", "LANG")
        if name in os.environ
    }
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        result = subprocess.run(
            ["git", "-C", str(worktree), *arguments],
            env=environment,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RunError(
            "GIT_ERROR", "Cannot inspect the selected worktree with git."
        ) from error
    if result.returncode:
        raise RunError("GIT_ERROR", "Cannot inspect the selected worktree with git.")
    return result.stdout


def validate_worktree(path: Path) -> tuple[Path, Path]:
    try:
        worktree = path.resolve(strict=True)
        top = Path(
            os.fsdecode(git(worktree, "rev-parse", "--show-toplevel")).rstrip("\n")
        )
        common = Path(
            os.fsdecode(
                git(worktree, "rev-parse", "--path-format=absolute", "--git-common-dir")
            ).strip()
        )
        gitdir = Path(
            os.fsdecode(git(worktree, "rev-parse", "--absolute-git-dir")).strip()
        )
        # Git lists the main worktree first; unlike common.parent this also
        # handles a main worktree whose git directory was relocated.
        first = git(worktree, "worktree", "list", "--porcelain", "-z").split(b"\0", 1)[
            0
        ]
        main = Path(os.fsdecode(first.removeprefix(b"worktree "))).resolve()
        if (
            worktree != top.resolve()
            or common.resolve() == gitdir.resolve()
            or worktree == main
            or not (worktree / ".git").is_file()
            or not worktree.is_relative_to(main / ".claude/worktrees")
        ):
            raise ValueError("not a dedicated linked root")
        return worktree, main
    except (OSError, ValueError) as error:
        raise RunError(
            "INVALID_WORKTREE",
            "Use an existing linked worktree root under the main repository's .claude/worktrees/; the main tree is forbidden.",
        ) from error


def external_path(path: Path, main: Path) -> Path:
    if path.resolve().is_relative_to(main):
        raise RunError(
            "REPOSITORY_SECRET_PATH",
            "Keep the key file and persistent DSH_HOME outside the repository, including its worktrees.",
        )
    return path.absolute()


def read_key(path: Path) -> str:
    hint = (
        "Create the external key file with `install -m 600 /dev/null "
        + shlex.quote(str(path))
        + "`, then use an editor to write only the API key. Existing files must be owned by you and mode 0600 (or 0400). No environment/.env/credentials fallback is used."
    )
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & ~0o600
                or info.st_size > 16384
            ):
                raise ValueError("unsafe key file")
            value = stream.read(16385).decode("utf-8").strip()
        if not value or any(
            character.isspace() or character == "\0" for character in value
        ):
            raise ValueError("empty or malformed key file")
        return value
    except (OSError, ValueError, UnicodeError) as error:
        # Never include the underlying exception: it may contain secret bytes.
        raise RunError("INVALID_KEY_FILE", hint) from error


def file_digest(path: Path) -> str | None:
    if path.is_symlink():
        return hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest()
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(
    worktree: Path, records: Path, phase: str, secrets: list[bytes] | None = None
) -> dict[str, Any]:
    status = git(
        worktree,
        "status",
        "--porcelain=v1",
        "--no-renames",
        "--untracked-files=all",
        "-z",
    )
    tracked: list[dict[str, Any]] = []
    untracked: list[dict[str, Any]] = []
    for entry in status.split(b"\0"):
        if not entry:
            continue
        code, name = entry[:2].decode("ascii"), os.fsdecode(entry[3:])
        item = {"path": name, "status": code, "sha256": file_digest(worktree / name)}
        (untracked if code == "??" else tracked).append(item)
    result = {
        "head": git(worktree, "rev-parse", "HEAD").decode().strip(),
        "tracked": tracked,
        "untracked": untracked,
    }
    private_write(records / f"git-{phase}.status", status, secrets)
    for suffix, extra in (("diff", []), ("index.diff", ["--cached"])):
        data = git(
            worktree,
            "diff",
            *extra,
            "--binary",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "HEAD",
            "--",
        )
        private_write(records / f"git-{phase}.{suffix}", data, secrets)
        result[suffix + "_sha256"] = hashlib.sha256(data).hexdigest()
    private_write(records / f"git-{phase}.json", json_bytes(result), secrets)
    return result


def diff_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    def files(state: dict[str, Any]) -> dict[str, Any]:
        return {item["path"]: item for item in state["tracked"] + state["untracked"]}

    old, new = files(before), files(after)
    return {
        "changed": before != after,
        "head_changed": before["head"] != after["head"],
        "changed_paths": sorted(
            name for name in old.keys() | new.keys() if old.get(name) != new.get(name)
        ),
        "tracked_before": [item["path"] for item in before["tracked"]],
        "tracked_after": [item["path"] for item in after["tracked"]],
        "untracked_before": [item["path"] for item in before["untracked"]],
        "untracked_after": [item["path"] for item in after["untracked"]],
    }
