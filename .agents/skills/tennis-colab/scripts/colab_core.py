"""Local state and explicit Colab/SSH transport; never allocate on a read."""

from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

STATE_ROOT = Path.home() / ".local/state/tennis-lab/colab"
COLAB_BIN = Path.home() / ".local/share/tennis-lab/colab-tools/bin/colab"
NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,47}\Z")
SMALL_FILE_LIMIT = 16 * 1024 * 1024


class ColabError(RuntimeError):
    """A failed or ambiguous remote operation requiring inspection."""


def now() -> str:
    return datetime.now(UTC).isoformat()


def validate_name(value: str) -> str:
    if not NAME.fullmatch(value):
        raise ColabError(
            "Use a 1–48 character lower-case session/job name: letters, digits, _ or -"
        )
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ColabError(f"Expected a JSON object: {path}")
    return value


def github_url(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.port
    ):
        raise ColabError("Repository URL must be a credential-free GitHub HTTPS URL")
    if not re.fullmatch(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", parsed.path):
        raise ColabError("Invalid GitHub repository path")
    if any(part in {".", ".."} for part in parsed.path.split("/")):
        raise ColabError("Invalid GitHub repository components")
    return value


def checked(
    argv: Sequence[str], *, timeout: int = 180, data: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            argv, input=data, capture_output=True, check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired as error:
        raise ColabError(
            f"Command timed out after {timeout}s; inspect state before retrying"
        ) from error
    if result.returncode:
        message = (result.stderr or result.stdout).decode(errors="replace")[-4000:]
        raise ColabError(f"Command failed ({result.returncode}): {message}")
    return result


class Session:
    def __init__(
        self, name: str, *, state_root: Path = STATE_ROOT, colab_bin: Path = COLAB_BIN
    ) -> None:
        self.name = validate_name(name)
        self.state_root = state_root.expanduser().resolve()
        self.directory = self.state_root / name
        self.state_file = self.directory / "state.json"
        self.registry = self.directory / "cli-sessions.json"
        self.identity = self.directory / "identity"
        self.colab_bin = colab_bin.expanduser().resolve()
        self.remote_root = f"/content/.tennis-colab/{name}"
        self.remote_helper = self.remote_root + "/remote.py"
        self.socket = self.state_root / (
            "ssh-" + hashlib.sha256(name.encode()).hexdigest()[:12]
        )
        if len(str(self.socket).encode()) > 100:
            raise ColabError("State root is too long for an SSH control socket")

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        with (self.directory / "lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def state(self) -> dict[str, Any]:
        state = read_json(self.state_file)
        if state.get("schema_version") != 1 or state.get("session") != self.name:
            raise ColabError("Unrecognized session metadata")
        return state

    def save(self, **fields: Any) -> dict[str, Any]:
        state = (
            self.state()
            if self.state_file.exists()
            else {"schema_version": 1, "session": self.name}
        )
        state.update(fields, updated_at=now())
        atomic_json(self.state_file, state)
        return state

    def cli_argv(self, *arguments: str) -> list[str]:
        if not self.colab_bin.is_file():
            raise ColabError(
                f"Colab CLI is missing: {self.colab_bin}; run the skill's install_cli.sh"
            )
        return [
            str(self.colab_bin),
            "--auth",
            "oauth2",
            "--config",
            str(self.registry),
            *arguments,
        ]

    def cli(
        self, *arguments: str, timeout: int = 180
    ) -> subprocess.CompletedProcess[bytes]:
        result = checked(self.cli_argv(*arguments), timeout=timeout, data=b"")
        if self.registry.exists():
            self.registry.chmod(0o600)
        return result

    def require_registered(self) -> None:
        """The official SSH proxy auto-creates absent names; never call it then.

        Read only the owned registry, retaining all tokens within the process.
        The official CLI remains the sole writer of that registry.
        """
        if not self.state_file.exists() or self.state().get("phase") in {
            "stopped",
            "allocating",
        }:
            raise ColabError(
                "No connectable owned session; use explicit start or inspect allocation state"
            )
        if not self.registry.is_file():
            raise ColabError(
                "Colab registry is absent; refusing an implicit allocation"
            )
        names = read_json(self.registry)
        if self.name not in names or not isinstance(names[self.name], dict):
            raise ColabError(
                "Session is absent from Colab registry; refusing an implicit allocation"
            )

    def ssh_argv(self, remote_argv: Sequence[str]) -> list[str]:
        self.require_registered()
        proxy = shlex.join(
            self.cli_argv(
                "ssh", "--proxy-mode", "-s", self.name, "--identity", str(self.identity)
            )
        )
        return [
            "ssh",
            "-F",
            "/dev/null",
            "-T",
            "-i",
            str(self.identity),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={self.directory / 'known_hosts'}",
            "-o",
            f"HostKeyAlias=tennis-colab-{self.name}",
            "-o",
            "ConnectTimeout=45",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=300",
            "-o",
            f"ControlPath={self.socket}",
            "-o",
            f"ProxyCommand={proxy}",
            "root@colab-runtime",
            shlex.join(remote_argv),
        ]

    def ssh(
        self,
        remote_argv: Sequence[str],
        *,
        data: bytes | None = None,
        timeout: int = 180,
    ) -> bytes:
        return checked(
            self.ssh_argv(remote_argv), data=data or b"", timeout=timeout
        ).stdout

    def kernel_argv(self, source: Path, *, timeout_seconds: int) -> list[str]:
        """Execute real work in the existing Notebook kernel, without allocation."""
        self.require_registered()
        if timeout_seconds <= 0:
            raise ColabError("Kernel execution timeout must be positive")
        return self.cli_argv(
            "exec", "--session", self.name, "--timeout", str(timeout_seconds),
            "--file", str(source),
        )

    def rpc(self, action: str, *, timeout: int = 180, **fields: Any) -> dict[str, Any]:
        data = json.dumps({"action": action, **fields}).encode()
        raw = self.ssh(
            ["python3", self.remote_helper, "rpc"], data=data, timeout=timeout
        )
        try:
            response = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ColabError(
                "Remote helper did not return a complete JSON receipt"
            ) from error
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise ColabError(f"Remote operation failed: {response}")
        return response

    def upload(self, local: Path, remote: str, *, mode: int = 0o600) -> dict[str, Any]:
        content = local.read_bytes()
        if len(content) > SMALL_FILE_LIMIT:
            raise ColabError("Use the Drive skill for files larger than 16 MiB")
        # stdin carries file bytes, so credentials never enter command history.
        code = (
            "import sys, pathlib, hashlib, json, os; "
            f"p=pathlib.Path({remote!r}); p.parent.mkdir(parents=True,exist_ok=True); "
            "data=sys.stdin.buffer.read(); tmp=p.with_name(p.name+'.uploading'); "
            f"tmp.write_bytes(data); tmp.chmod({mode}); os.replace(tmp,p); "
            "print(json.dumps({'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}))"
        )
        result = json.loads(self.ssh(["python3", "-c", code], data=content))
        if result.get("sha256") != hashlib.sha256(content).hexdigest():
            raise ColabError("Uploaded file checksum differs")
        return {"path": remote, **result}

    def download(self, remote: str, local: Path) -> dict[str, Any]:
        if local.exists():
            raise ColabError(f"Download destination exists: {local}")
        response = self.rpc("read_file", path=remote, max_bytes=SMALL_FILE_LIMIT)
        content = base64.b64decode(response["data"], validate=True)
        if hashlib.sha256(content).hexdigest() != response["sha256"]:
            raise ColabError("Downloaded file checksum differs")
        local.parent.mkdir(parents=True, exist_ok=True)
        with local.open("xb") as stream:
            stream.write(content)
        return {"path": str(local), "bytes": len(content), "sha256": response["sha256"]}

    def close_transport(self) -> None:
        if self.socket.exists():
            result = subprocess.run(
                [
                    "ssh",
                    "-F",
                    "/dev/null",
                    "-S",
                    str(self.socket),
                    "-O",
                    "exit",
                    "root@colab-runtime",
                ],
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            if result.returncode and self.socket.exists():
                raise ColabError("SSH control transport did not close")
