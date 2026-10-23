"""Private receiver discovery, conversation fencing and assignment binding."""

from __future__ import annotations

import json
import os
import shutil
import stat
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from herdr_labels.coordination import Guard, locked, owner_token
from herdr_labels.service import registry, target
from lib import codex_queue
from lib.identity import session_identity
from parent.state import Store as ParentStore

from .access import parent_state
from .errors import BoardError
from .names import for_parent
from .storage import Store, directory


def root(env: Mapping[str, str]) -> Path:
    path = directory(env) / "receivers"
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory(env).chmod(0o700)
    path.chmod(0o700)
    return path


def write(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    ParentStore.write_json(path, record)


def read(path: Path) -> dict[str, Any] | None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    with os.fdopen(fd) as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.getuid()
        ):
            raise BoardError("invalid private receiver state permissions")
        record = json.load(stream)
    if not isinstance(record, dict) or record.get("version") != 1:
        raise BoardError("invalid private receiver state")
    return record


def process(pid: int) -> dict[str, Any]:
    fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "start": fields[19],
        "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def hook_process(env: Mapping[str, str]) -> dict[str, Any]:
    """Use the foreground ancestor guard shared with herdr_labels.

    Outside Herdr, exactly one native Claude ancestor is required. Environment
    markers alone cannot distinguish an unmarked headless nested Claude.
    """
    identity = session_identity(env)
    if not identity.startswith("claude:"):
        raise BoardError("only a root Claude process may run this hook")
    if env.get("HERDR_ENV") == "1":
        destination = target(env, identity)
        if destination is None:
            raise BoardError("Claude hook has no confirmed foreground agent")
        client, pane = destination
        guard = Guard(env, identity, client, pane)
        guard.require_native_ancestor()
        return process(guard.stamp[0]["pid"])
    ancestors = []
    pid = os.getppid()
    while pid > 1:
        record = (Path("/proc") / str(pid) / "stat").read_text()
        if record.split("(", 1)[1].rsplit(")", 1)[0] == "claude":
            ancestors.append(pid)
        pid = int(record.rsplit(")", 1)[1].split()[1])
    if len(ancestors) != 1:
        raise BoardError("Claude hook requires one root native agent ancestor")
    return process(ancestors[0])


def _binding(
    env: Mapping[str, str], identity: str, parents: Mapping[str, Any]
) -> dict[str, Any] | None:
    owned = [
        (key, row)
        for key, row in parents.items()
        if row["session"] == identity and row["status"] == "active"
    ]
    if len(owned) > 1:
        raise BoardError("ambiguous board receiver assignment")
    if not owned:
        return None
    key, row = owned[0]
    repo, _, number = key.rpartition("#")
    store = Store(directory(env))
    if not store.path.exists():
        return None
    with store.read() as connection:
        assignment = for_parent(connection, repo, int(number))
        if assignment is None or assignment.retired_at is not None:
            return None
        return {
            "repo": repo,
            "parent": int(number),
            "assignment_id": assignment.id,
            "generation": row["generation"],
        }


def capture_claude(env: Mapping[str, str], source: str, native: dict[str, Any]) -> None:
    identity = session_identity(env)
    # Only the user's settings are evidence of the agreed prerequisite. Project
    # accept does not enable this route; never modify settings from this code.
    config_dir = (
        Path(env["CLAUDE_CONFIG_DIR"])
        if env.get("CLAUDE_CONFIG_DIR")
        else Path(env.get("HOME", str(Path.home()))) / ".claude"
    )
    settings = config_dir / "settings.json"
    try:
        config = json.loads(settings.read_text())
        accepting = (
            isinstance(config, dict) and config.get("crossSessionInbound") == "accept"
        )
    except (OSError, ValueError):
        accepting = False
    epoch = uuid.uuid4().hex
    socket_path = env.get("CLAUDE_CODE_MESSAGING_SOCKET", "")
    record = {
        "version": 1,
        "identity": identity,
        "epoch": epoch,
        "process": native,
        "socket": socket_path,
        "token": env.get("CLAUDE_CODE_MESSAGING_TOKEN", ""),
        "accepting": accepting,
        "source": source,
    }
    with parent_state(env) as parents, locked(root(env) / "routes.lock"):
        if process(native["pid"]) != native:
            raise BoardError("Claude receiver process changed")
        # Even /clear with an unchanged session ID/socket must require a fresh
        # explicit parent registration, like the display lifecycle contract.
        record["binding"] = (
            None if source == "clear" else _binding(env, identity, parents)
        )
        slot = owner_token(json.dumps(native, sort_keys=True))
        record["slot"] = slot
        current = {"version": 1, "identity": identity, "epoch": epoch}
        write(root(env) / "processes" / f"{slot}.json", current)
        if socket_path:
            write(root(env) / "sockets" / f"{owner_token(socket_path)}.json", current)
        write(root(env) / f"{owner_token(identity)}.json", record)


def _codex(env: Mapping[str, str], identity: str) -> dict[str, Any] | None:
    if not all(env.get(key) for key in codex_queue.ENV_KEYS):
        return None
    thread = identity.partition(":")[2]
    if str(uuid.UUID(thread)) != thread:
        raise BoardError("Codex root identity is not a canonical thread ID")
    home = Path(env["CODEX_HOME"])
    if not home.is_absolute() or not Path(env["CODEX_SQLITE_HOME"]).is_absolute():
        raise BoardError("Codex receiver environment must use absolute locations")
    # Exact root-thread lookup, never latest-file selection. Multiple rollout
    # generations require explicit inspection, not guessing by mtime.
    paths = list((home / "sessions").rglob(f"*{thread}.jsonl"))
    if len(paths) != 1:
        raise BoardError("Codex receiver rollout is missing or ambiguous")
    with paths[0].open("rb") as stream:
        metadata = json.loads(stream.readline())["payload"]
    task = {
        "thread_id": thread,
        "cwd": metadata["cwd"],
        "rollout": str(paths[0].resolve()),
        "environment": {key: env[key] for key in codex_queue.ENV_KEYS},
        "codex": shutil.which("codex", path=env["PATH"]),
    }
    if not task["codex"] or not Path(task["cwd"]).is_absolute():
        raise BoardError("Codex receiver runtime is unavailable")
    _, boundary = codex_queue.history(task, None)
    if not boundary["line_complete"]:
        raise BoardError("Codex receiver rollout is incomplete")
    return {"version": 1, "identity": identity, **task}


def bind_receiver(env: Mapping[str, str]) -> dict[str, str]:
    """Run after parent register/accept, with its registry writer lock released."""
    identity = session_identity(env)
    engine = identity.partition(":")[0]
    with parent_state(env) as parents, locked(root(env) / "routes.lock"):
        binding = _binding(env, identity, parents)
        if binding is None:
            raise BoardError("receiver has no active board assignment")
        path = root(env) / f"{owner_token(identity)}.json"
        if engine == "codex":
            # Registration that discovers a broken/ambiguous runtime must not
            # leave an older same-generation route usable by later posts.
            write(path, {"version": 1, "identity": identity, "binding": None})
            record = _codex(env, identity)
        else:
            record = read(path)
        if record is None:
            return {"status": "not_configured", "reason": "route_unavailable"}
        if engine == "claude":
            _check_claude(env, record)
        record["binding"] = binding
        write(path, record)
    return {"status": "bound"}


def _check_claude(env: Mapping[str, str], record: dict[str, Any]) -> None:
    expected = {"version": 1, "identity": record["identity"], "epoch": record["epoch"]}
    if (
        process(record["process"]["pid"]) != record["process"]
        or read(root(env) / "processes" / f"{record['slot']}.json") != expected
        or (
            record["socket"]
            and read(root(env) / "sockets" / f"{owner_token(record['socket'])}.json")
            != expected
        )
    ):
        raise BoardError("Claude receiver conversation was replaced")


def resolve(env: Mapping[str, str], notification: Any) -> dict[str, Any]:
    """Caller holds routes.lock and the dispatcher already fences the registry."""
    row = registry(env).get(f"{notification.repo}#{notification.parent}")
    if not row or row["status"] != "active":
        raise BoardError("receiver assignment is unavailable")
    identity = row["session"]
    if identity.partition(":")[0] != notification.engine:
        raise BoardError("receiver engine changed")
    record = read(root(env) / f"{owner_token(identity)}.json")
    binding = {
        "repo": notification.repo,
        "parent": notification.parent,
        "assignment_id": notification.assignment_id,
        "generation": row["generation"],
    }
    if (
        not record
        or record.get("identity") != identity
        or record.get("binding") != binding
    ):
        raise BoardError("receiver route is unbound or stale")
    if notification.engine == "claude":
        _check_claude(env, record)
    return record
