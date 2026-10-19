"""Drain private redacted streams and stop a complete dsh process group."""

from __future__ import annotations

import os
import selectors
import subprocess
import time
from pathlib import Path
from typing import BinaryIO

from tools.dsh.process_tree import ProcessTree


class Interrupt:
    def __init__(self) -> None:
        self.signal: int | None = None

    def receive(self, number: int, _frame: object) -> None:
        self.signal = self.signal or number


class Redactor:
    def __init__(self, stream: BinaryIO, secrets: list[bytes]) -> None:
        self.stream = stream
        self.secrets = sorted(
            {secret for secret in secrets if secret}, key=len, reverse=True
        )
        self.keep = max((len(secret) for secret in self.secrets), default=1) - 1
        self.pending = b""

    def write(self, data: bytes, *, final: bool = False) -> None:
        self.pending += data
        # Keeping a full possible prefix prevents a key split across pipe reads
        # from reaching disk. Replacement happens before any bytes are written.
        for secret in self.secrets:
            self.pending = self.pending.replace(secret, b"<redacted-api-key>")
        count = len(self.pending) if final else max(0, len(self.pending) - self.keep)
        self.stream.write(self.pending[:count])
        self.stream.flush()
        self.pending = self.pending[count:]


def execute(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    task: bytes,
    timeout: float,
    interrupt: Interrupt,
    stdout: BinaryIO,
    stderr: BinaryIO,
    secrets: list[bytes],
) -> tuple[int, str | None]:
    output = Redactor(stdout, secrets)
    diagnostic = Redactor(stderr, secrets)
    tree = ProcessTree()
    try:
        child = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except BaseException:
        tree.close()
        raise
    assert (
        child.stdin is not None
        and child.stdout is not None
        and child.stderr is not None
    )
    deadline = time.monotonic() + timeout
    stopped = False
    outcome = None
    offset = 0
    try:
        with selectors.DefaultSelector() as selector:
            for pipe, writer in ((child.stdout, output), (child.stderr, diagnostic)):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, writer)
            os.set_blocking(child.stdin.fileno(), False)
            selector.register(child.stdin, selectors.EVENT_WRITE, None)
            while selector.get_map():
                if not stopped:
                    if interrupt.signal is not None:
                        outcome = "interrupted"
                    elif time.monotonic() >= deadline:
                        outcome = "timeout"
                    if outcome is not None or child.poll() is not None:
                        tree.stop(child)
                        stopped = True
                for key, _ in selector.select(timeout=0.05):
                    if key.data is None:
                        try:
                            offset += os.write(key.fd, task[offset : offset + 65536])
                        except BrokenPipeError:
                            offset = len(task)
                        except BlockingIOError:
                            continue
                        if offset == len(task):
                            selector.unregister(key.fileobj)
                            child.stdin.close()
                    else:
                        try:
                            data = os.read(key.fd, 65536)
                        except BlockingIOError:
                            continue
                        if data:
                            key.data.write(data)
                        else:
                            selector.unregister(key.fileobj)
                if not selector.get_map() and child.poll() is None:
                    # A process can close all pipes and keep running.
                    while (
                        child.poll() is None
                        and interrupt.signal is None
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.05)
                    if child.poll() is None:
                        outcome = (
                            "interrupted" if interrupt.signal is not None else "timeout"
                        )
    finally:
        if not stopped:
            tree.stop(child)
        for pipe in (child.stdin, child.stdout, child.stderr):
            pipe.close()
        output.write(b"", final=True)
        diagnostic.write(b"", final=True)
        tree.close()
    return child.returncode, outcome
