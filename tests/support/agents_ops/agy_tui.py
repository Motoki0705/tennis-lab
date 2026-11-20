"""Disposable agy stand-in with a real foreground TTY, flock and HTTP listener."""

from __future__ import annotations

import ctypes
import fcntl
import json
import os
import pty
import selectors
import subprocess
import sys
import termios
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

ROOT = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"
CHILD = "33333333-3333-4333-8333-333333333333"
TRAJECTORY = "44444444-4444-4444-8444-444444444444"
TOKEN = "agy-fixture-csrf-token"
ARGV_ENV = "TEST_AGY_CMDLINES"


@dataclass
class Tui:
    process: subprocess.Popen[str]
    port: int
    args: tuple[str, ...]

    @property
    def pid(self) -> int:
        return self.process.pid

    def _reply(self) -> Any:
        assert self.process.stdout is not None
        with selectors.DefaultSelector() as ready:
            ready.register(self.process.stdout, selectors.EVENT_READ)
            assert ready.select(timeout=30), "agy fixture stopped responding"
        return json.loads(self.process.stdout.readline())

    def _request(self, **request: Any) -> Any:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return self._reply()

    def lock(self, conversation: str | None) -> bool:
        return self._request(action="lock", conversation=conversation) is True

    def metadata(
        self,
        conversation: str,
        *,
        root: str = ROOT,
        trajectory: str = TRAJECTORY,
        parent: str | None = None,
    ) -> None:
        value = {"rootConversationId": root}
        if parent:
            value["parentConversationId"] = parent
        assert self._request(
            action="metadata",
            conversation=conversation,
            payload={"metadata": value, "trajectoryId": trajectory},
        )

    def run(
        self,
        entry: Path | str,
        *args: str,
        env: Mapping[str, str],
        cwd: Path,
        conversation: str = ROOT,
        trajectory: str = TRAJECTORY,
        metadata_port: int | None = None,
        stdin: dict[str, Any] | None = None,
        nested: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        from tests.support.agents_ops.hook_process import hook_command

        port = self.port if metadata_port is None else metadata_port
        caller = {
            **{
                key: val
                for key, val in env.items()
                if not key.startswith("ANTIGRAVITY_")
            },
            "ANTIGRAVITY_CONVERSATION_ID": conversation,
            "ANTIGRAVITY_TRAJECTORY_ID": trajectory,
            "ANTIGRAVITY_LS_ADDRESS": f"localhost:{port}",
            "ANTIGRAVITY_CSRF_TOKEN": TOKEN,
            ARGV_ENV: json.dumps({str(self.pid): ("agy", *self.args)}),
        }
        command = [*hook_command(entry), *args]
        if nested:
            command = [sys.executable, __file__, "nested", *command]
        return subprocess.CompletedProcess(
            command,
            **self._request(
                action="run",
                command=command,
                cwd=str(cwd),
                env=caller,
                stdin=None if stdin is None else json.dumps(stdin),
            ),
        )


@contextmanager
def foreground(home: Path, *args: str, tty: bool = True) -> Iterator[Tui]:
    home.mkdir(parents=True, exist_ok=True)
    master, slave = pty.openpty()
    try:
        with subprocess.Popen(
            [sys.executable, __file__, "serve", str(slave) if tty else ""],
            start_new_session=True,
            pass_fds=(slave,),
            env={**os.environ, "HOME": str(home)},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        ) as process:
            tui = Tui(process, 0, args)
            try:
                tui.port = int(tui._reply())
                yield tui
            finally:
                assert process.stdin is not None
                process.stdin.close()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    finally:
        os.close(slave)
        os.close(master)


def _name_process() -> None:
    if ctypes.CDLL(None).prctl(15, b"agy", 0, 0, 0) != 0:
        raise OSError("PR_SET_NAME failed")


def _serve() -> None:
    _name_process()
    if sys.argv[2]:
        fcntl.ioctl(int(sys.argv[2]), termios.TIOCSCTTY, 0)
    presence = Path.home() / ".gemini/antigravity-cli/presence"
    presence.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, dict[str, Any]] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: Any) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            value = metadata.get(body["conversationId"])
            if (
                self.path
                != "/exa.language_server_pb.LanguageServerService/GetConversationMetadata"
                or self.headers.get("x-codeium-csrf-token") != TOKEN
                or value is None
            ):
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(json.dumps(server.server_port), flush=True)
        lock = None
        for line in sys.stdin:
            request = json.loads(line)
            if request["action"] == "lock":
                if lock is not None:
                    lock.close()
                    lock = None
                if conversation := request["conversation"]:
                    lock = (presence / f"{conversation}.lock").open("a")
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        # Keep a descriptor to exercise open-but-not-held detection.
                        print("false", flush=True)
                        continue
                reply: Any = True
            elif request["action"] == "metadata":
                metadata[request["conversation"]] = request["payload"]
                reply = True
            else:
                env = request["env"]
                if "AGENTS_TEST_GITHUB_HTTP" in os.environ:
                    env["AGENTS_TEST_GITHUB_HTTP"] = os.environ[
                        "AGENTS_TEST_GITHUB_HTTP"
                    ]
                    env["PYTHONPATH"] = os.pathsep.join(
                        (os.environ["PYTHONPATH"], env.get("PYTHONPATH", ""))
                    )
                result = subprocess.run(
                    request["command"],
                    cwd=request["cwd"],
                    env=env,
                    input=request["stdin"],
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=25,
                )
                reply = {
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            print(json.dumps(reply), flush=True)


def _nested() -> int:
    _name_process()
    cmdlines = json.loads(os.environ[ARGV_ENV])
    cmdlines[str(os.getpid())] = ["agy"]
    env = {**os.environ, ARGV_ENV: json.dumps(cmdlines)}
    return subprocess.run(sys.argv[2:], env=env, check=False).returncode


if __name__ == "__main__":
    if sys.argv[1] == "nested":
        raise SystemExit(_nested())
    _serve()
