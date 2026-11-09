"""Short-lived local runtime: start/serve/status/stop/export, never systemd."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import select
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from lib.opsproc import process_identity

from .paths import SafeDirectory
from .schema import ContractError, ErrorResponse, Identifier, Model, parse
from .store import RevisionStore


class _Session(Model):
    pid: int
    start_ticks: str
    boot_id: str
    port: int
    token: str
    document_id: Identifier | None = None

    @property
    def identity(self) -> dict[str, str | int]:
        return {
            "pid": self.pid,
            "start_ticks": self.start_ticks,
            "boot_id": self.boot_id,
        }

    @property
    def url(self) -> str:
        document = f"document_id={self.document_id}&" if self.document_id else ""
        return f"http://localhost:{self.port}/#{document}token={self.token}"


def _load(specification: str) -> Any:
    module, separator, name = specification.partition(":")
    if not separator or not module or not name or "." in name:
        raise ContractError("invalid_request")
    function = getattr(importlib.import_module(module), name)
    if not callable(function):
        raise ContractError("invalid_request")
    return function


def _session(root: Path) -> _Session | None:
    try:
        with SafeDirectory(root) as filesystem:
            result: _Session = parse(_Session, filesystem.read("private/session.json"))
            return result
    except FileNotFoundError:
        return None


def _live(session: _Session | None) -> bool:
    return session is not None and process_identity(session.pid) == session.identity


def _serve(arguments: argparse.Namespace) -> None:
    import secrets

    import uvicorn

    from .gemini import DEFAULT_AGY, Gemini
    from .jobs import JobController
    from .managed import ManagedResources, with_visual_validation
    from .service import create_app

    controller: JobController | None = None
    ready = arguments.ready_fd
    try:
        store = RevisionStore(arguments.root)
        if arguments.document:
            store.current(arguments.document)
        resources = ManagedResources()
        validator = with_visual_validation(_load(arguments.validator))
        compiler_factory = _load(arguments.compiler) if arguments.compiler else None
        prepare = _load(arguments.prepare) if arguments.prepare else resources.prepare
        ui = _load(arguments.ui)() if arguments.ui else None
        gemini = Gemini(
            arguments.root / "private/gemini", agy=arguments.agy or DEFAULT_AGY
        )

        editor_factory = resources.editor_factory(
            gemini, compiler_factory=compiler_factory
        )
        controller = JobController(store, editor_factory, validator, prepare=prepare)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # There is deliberately no configurable LAN/wildcard bind.
            sock.bind(("127.0.0.1", arguments.port))
            sock.listen(socket.SOMAXCONN)
            identity = process_identity(os.getpid())
            assert identity is not None
            session = parse(
                _Session,
                json.dumps(
                    {
                        **identity,
                        "port": sock.getsockname()[1],
                        "token": secrets.token_urlsafe(32),
                        "document_id": arguments.document,
                    }
                ),
            )
            app = create_app(
                controller, port=session.port, token=session.token, ui_files=ui
            )

            class Server(uvicorn.Server):
                async def startup(
                    self, sockets: list[socket.socket] | None = None
                ) -> None:
                    nonlocal ready
                    await super().startup(sockets=sockets)
                    if not self.started:
                        raise ContractError("storage_error")
                    with SafeDirectory(arguments.root) as filesystem:
                        filesystem.atomic_write(
                            "private/session.json", session.model_dump_json().encode()
                        )
                    announcement = json.dumps(
                        {
                            "running": True,
                            "pid": session.pid,
                            "port": session.port,
                            "url": session.url,
                        }
                    )
                    if ready is not None:
                        os.write(ready, announcement.encode() + b"\n")
                        os.close(ready)
                        ready = None
                    else:
                        print(announcement, flush=True)

            Server(uvicorn.Config(app, log_config=None, access_log=False)).run(
                sockets=[sock]
            )
    except Exception:
        logging.getLogger(__name__).exception("Private startup failure")
        if ready is not None:
            os.write(ready, b'{"error":"startup_failed"}\n')
        raise
    finally:
        if ready is not None:
            os.close(ready)
        if controller is not None:
            controller.close()


def _start(arguments: argparse.Namespace) -> None:
    RevisionStore(arguments.root)
    with SafeDirectory(arguments.root) as filesystem:
        if os.fstat(filesystem.fd).st_mode & 0o077:
            raise ContractError("unsafe_path")
        with filesystem.open_file(
            "private/service.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND
        ) as log:
            read_fd, write_fd = os.pipe()
            command = [
                sys.executable,
                "-m",
                "explain_html",
                "serve",
                "--root",
                str(arguments.root),
                "--port",
                str(arguments.port),
                "--validator",
                arguments.validator,
                "--ready-fd",
                str(write_fd),
            ]
            for name in ("agy", "prepare", "ui", "document", "compiler"):
                value = getattr(arguments, name)
                if value:
                    command.extend([f"--{name}", str(value)])
            try:
                child = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    pass_fds=(write_fd,),
                    start_new_session=True,
                )
            finally:
                os.close(write_fd)
            try:
                if not select.select([read_fd], [], [], 10)[0]:
                    raise ContractError("timeout")
                value = json.loads(os.read(read_fd, 65536))
                if "error" in value:
                    raise ContractError("storage_error")
                session = _session(arguments.root)
                if not _live(session) or session is None or session.pid != child.pid:
                    raise ContractError("storage_error")
                print(json.dumps(value), flush=True)
            except BaseException:
                child.terminate()
                child.wait(timeout=10)
                raise
            finally:
                os.close(read_fd)


def _stop(root: Path) -> None:
    session = _session(root)
    if session is None or not _live(session):
        print(json.dumps({"running": False}))
        return
    try:
        descriptor = os.pidfd_open(session.pid)
    except ProcessLookupError:
        print(json.dumps({"running": False}))
        return
    try:
        if process_identity(session.pid) != session.identity:
            raise ContractError("interrupted")
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        if not select.select([descriptor], [], [], 15)[0]:
            raise ContractError("timeout")
    finally:
        os.close(descriptor)
    print(json.dumps({"running": False, "stopped": True}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "serve", "status", "stop", "export"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name in {"start", "serve"}:
            command.add_argument(
                "--document",
                help="published document ID for the fixed UI launch fragment",
            )
            command.add_argument("--port", type=int, default=0)
            command.add_argument(
                "--compiler",
                help="trusted compiler factory override; default uses managed visual compilation",
            )
            command.add_argument(
                "--validator",
                required=True,
                help="trusted module:callable performing sanitization and rendering",
            )
            command.add_argument(
                "--prepare",
                help="trusted module:callable preparing assets and evidence",
            )
            command.add_argument(
                "--ui", help="trusted module:callable returning fixed UI files"
            )
            command.add_argument(
                "--agy",
                type=Path,
                help="pinned agy executable; otherwise use the adapter default",
            )
            command.add_argument("--ready-fd", type=int, help=argparse.SUPPRESS)
        elif name == "export":
            command.add_argument("--document", required=True)
            command.add_argument(
                "--revision", help="published revision; defaults to current"
            )
    arguments = parser.parse_args()
    arguments.root = arguments.root.absolute()
    try:
        if arguments.command in {"serve", "start"}:
            # Do not send raw adapter errors or prompts to a terminal or HTTP.
            with SafeDirectory(arguments.root, create=True) as filesystem:
                filesystem.mkdir("private")
                with filesystem.open_file(
                    "private/service.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND
                ) as descriptor:
                    handler = logging.StreamHandler(os.fdopen(os.dup(descriptor), "a"))
                    logging.getLogger().addHandler(handler)
            if arguments.command == "serve":
                _serve(arguments)
            else:
                _start(arguments)
        elif arguments.command == "stop":
            _stop(arguments.root)
        elif arguments.command == "status":
            session = _session(arguments.root)
            print(
                json.dumps(
                    {
                        "running": _live(session),
                        "url": session.url if _live(session) and session else None,
                    }
                )
            )
        else:
            store = RevisionStore(arguments.root)
            revision_id = (
                arguments.revision or store.current(arguments.document).revision_id
            )
            snapshot = store.snapshot(arguments.document, revision_id)
            print(
                json.dumps(
                    {
                        **snapshot.model_dump(mode="json"),
                        "directory": str(
                            arguments.root / "exports" / snapshot.snapshot_id
                        ),
                    }
                )
            )
        return 0
    except Exception as exc:
        error = (
            exc if isinstance(exc, ContractError) else ContractError("storage_error")
        )
        print(ErrorResponse(error=error.error).model_dump_json())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
