"""Provenance for a local Antigravity CLI conversation on Linux."""

from __future__ import annotations

import http.client
import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

CONVERSATION_ENV = "ANTIGRAVITY_CONVERSATION_ID"
TRAJECTORY_ENV = "ANTIGRAVITY_TRAJECTORY_ID"
ADDRESS_ENV = "ANTIGRAVITY_LS_ADDRESS"
TOKEN_ENV = "ANTIGRAVITY_CSRF_TOKEN"
_UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
_VALUE_OPTIONS = {
    "add-dir",
    "agent",
    "conversation",
    "effort",
    "i",
    "log-file",
    "mode",
    "model",
    "project",
    "prompt-interactive",
    "settings",
}
_SWITCHES = {
    "c",
    "continue",
    "dangerously-skip-permissions",
    "new-project",
    "remote-control",
    "sandbox",
}


def interactive_options(args: Sequence[str]) -> Iterator[tuple[str, str | None]]:
    """Recognize the local TUI's options, including Go's single-dash form."""
    values = iter(args)
    for arg in values:
        name, equals, value = arg.removeprefix("-").removeprefix("-").partition("=")
        if not arg.startswith("-"):
            raise ValueError("agy parent operations require the interactive TUI")
        if name in _VALUE_OPTIONS:
            parameter = value if equals else next(values, None)
            if not parameter:
                raise ValueError("agy interactive option requires a value")
            yield name, parameter
        elif name in _SWITCHES:
            yield name, value if equals else None
        else:
            raise ValueError("agy parent operations require a local interactive TUI")


def _stat(pid: int) -> tuple[str, list[str]]:
    raw = Path(f"/proc/{pid}/stat").read_text()
    label, fields = raw.rsplit(")", 1)
    return label.split("(", 1)[1], fields.split()


def _fd_targets(pid: int) -> Iterator[tuple[str, str]]:
    for fd in Path(f"/proc/{pid}/fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue  # Unrelated descriptors may close during the scan.
        yield fd.name, target


@dataclass(frozen=True)
class AgyTui:
    pid: int
    start_ticks: str

    @classmethod
    def inspect(cls, pid: int) -> AgyTui:
        name, fields = _stat(pid)
        if (
            name != "agy"
            or fields[0] == "Z"
            or fields[4] == "0"  # No controlling terminal.
            or fields[2] != fields[5]  # Not the foreground process group.
        ):
            raise ValueError("agy parent operations require the foreground host TUI")
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        tuple(interactive_options([os.fsdecode(arg) for arg in argv[1:]]))
        return cls(pid, fields[19])

    def conversation(self) -> str | None:
        """Read the conversation actually locked by this TUI."""
        proc = Path(f"/proc/{self.pid}")
        if self.inspect(self.pid) != self:
            raise ValueError("agy TUI process changed")
        homes = [
            os.fsdecode(item[5:])
            for item in (proc / "environ").read_bytes().split(b"\0")
            if item.startswith(b"HOME=")
        ]
        if len(homes) != 1 or not Path(homes[0]).is_absolute():
            raise ValueError("agy TUI home is unavailable")
        presence = Path(homes[0]).resolve() / ".gemini/antigravity-cli/presence"
        held: set[str] = set()
        for fd, target in _fd_targets(self.pid):
            path = Path(target)
            if (
                path.parent != presence
                or path.suffix != ".lock"
                or not _UUID.fullmatch(path.stem)
            ):
                continue
            try:
                info = (proc / "fdinfo" / fd).read_text()
                if os.readlink(proc / "fd" / fd) != target:
                    continue
            except FileNotFoundError:
                continue
            if any(
                line.split()[2:6] == ["FLOCK", "ADVISORY", "WRITE", str(self.pid)]
                for line in info.splitlines()
                if line.startswith("lock:")
            ):
                held.add(path.stem)
        if self.inspect(self.pid) != self or len(held) > 1:
            raise ValueError("agy TUI conversation lock changed or is ambiguous")
        return next(iter(held), None)

    def require_conversation(self, conversation: str) -> None:
        if self.conversation() != conversation:
            raise ValueError("agy conversation differs from the foreground TUI lock")


def caller_tui() -> AgyTui:
    """Select exactly one native agent in the actual host ancestry."""
    found = None
    pid = os.getppid()
    try:
        while pid > 1:
            name, fields = _stat(pid)
            if name in {"agy", "claude", "codex", "antigravity", "antigravity-cli"}:
                if found is not None or name != "agy":
                    raise ValueError("nested agy parent operations are unsupported")
                found = AgyTui.inspect(pid)
            pid = int(fields[1])
    except OSError:
        raise ValueError("agy host process ancestry is unavailable") from None
    if found is None:
        raise ValueError("agy host process ancestry is unavailable (terminal sandbox?)")
    return found


def _listener(tui: AgyTui, port: int) -> tuple[str, str]:
    """Route the read-only request only to the verified agy process."""
    sockets = {target for _, target in _fd_targets(tui.pid)}
    for family, local, host in (
        ("tcp", "0100007F", "127.0.0.1"),
        ("tcp6", "00000000000000000000000001000000", "::1"),
    ):
        table = Path(f"/proc/{tui.pid}/net/{family}")
        try:
            rows = table.read_text().splitlines()[1:]
        except FileNotFoundError:
            continue
        for row in rows:
            fields = row.split()
            if (
                fields[1] == f"{local}:{port:04X}"
                and fields[3] == "0A"  # TCP_LISTEN
                and f"socket:[{fields[9]}]" in sockets
            ):
                return host, fields[9]
    raise ValueError("agy metadata endpoint is not owned by the foreground TUI")


def _require_root(env: Mapping[str, str], tui: AgyTui, conversation: str) -> None:
    trajectory = env.get(TRAJECTORY_ENV, "")
    address = re.fullmatch(
        r"(?:localhost|127\.0\.0\.1|\[::1\]):([0-9]{1,5})",
        env.get(ADDRESS_ENV, ""),
    )
    token = env.get(TOKEN_ENV)
    if not address or not _UUID.fullmatch(trajectory) or not token:
        raise ValueError("agy root metadata requires a host run_command environment")
    port = int(address[1])
    if not 0 < port <= 65535:
        raise ValueError("invalid agy metadata endpoint")
    listener = _listener(tui, port)
    connection = http.client.HTTPConnection(listener[0], port, timeout=2)
    try:
        connection.connect()
        if _listener(tui, port) != listener:
            raise ValueError("agy metadata listener changed during connection")
        connection.request(
            "POST",
            "/exa.language_server_pb.LanguageServerService/GetConversationMetadata",
            body=json.dumps({"conversationId": conversation}),
            headers={
                "Content-Type": "application/json",
                "x-codeium-csrf-token": token,
            },
        )
        response = connection.getresponse()
        limit = 1024 * 1024
        data = response.read(limit + 1) if response.status == 200 else b""
        if len(data) > limit:
            raise ValueError("oversized agy metadata response")
        payload = json.loads(data)
        metadata = payload["metadata"]
        root = metadata["rootConversationId"]
        observed_trajectory = payload["trajectoryId"]
    except (http.client.HTTPException, OSError, ValueError, TypeError, KeyError):
        raise ValueError("could not verify agy conversation metadata") from None
    finally:
        connection.close()
    if root != conversation or metadata.get("parentConversationId"):
        raise ValueError("agy parent operations require a root conversation")
    if observed_trajectory != trajectory:
        raise ValueError("agy trajectory differs from the caller")


def root_session(env: Mapping[str, str], conversation: str) -> str:
    if not _UUID.fullmatch(conversation):
        raise ValueError("invalid agy conversation ID")
    try:
        tui = caller_tui()
        tui.require_conversation(conversation)
        _require_root(env, tui, conversation)
        tui.require_conversation(conversation)
    except OSError:
        raise ValueError(
            "could not verify the agy host TUI or conversation lock"
        ) from None
    return f"agy:{conversation}"
