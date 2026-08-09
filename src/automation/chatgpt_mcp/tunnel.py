"""Outbound Cloudflare Quick Tunnel lifecycle for a private WSL listener."""

from __future__ import annotations

import os
import queue
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import TextIO

_QUICK_TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


class TunnelError(RuntimeError):
    """Raised when the outbound HTTPS tunnel cannot become ready."""


class QuickTunnel:
    """Launch cloudflared, capture its assigned HTTPS origin, and monitor it."""

    def __init__(
        self,
        *,
        cloudflared_path: Path,
        local_port: int,
        log_path: Path,
    ) -> None:
        self.cloudflared_path = cloudflared_path
        self.local_port = local_port
        self.log_path = log_path
        self.process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str] = queue.Queue()
        self._stopping = threading.Event()
        self._log_lock = threading.Lock()

    @staticmethod
    def extract_public_url(line: str) -> str | None:
        match = _QUICK_TUNNEL_URL.search(line)
        return match.group(0) if match else None

    def _read_stream(self, stream: TextIO) -> None:
        for line in iter(stream.readline, ""):
            self._lines.put(line)
            with self._log_lock, self.log_path.open("a", encoding="utf-8") as log:
                log.write(line)
        stream.close()

    def start(self, *, timeout_seconds: int = 45) -> str:
        if not self.cloudflared_path.is_file():
            raise TunnelError(f"cloudflared was not found: {self.cloudflared_path}")
        self.log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")
        os.chmod(self.log_path, 0o600)
        self.process = subprocess.Popen(
            [
                str(self.cloudflared_path),
                "tunnel",
                "--url",
                f"http://127.0.0.1:{self.local_port}",
                "--no-autoupdate",
                "--loglevel",
                "info",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        for stream in (self.process.stdout, self.process.stderr):
            threading.Thread(
                target=self._read_stream,
                args=(stream,),
                daemon=True,
            ).start()

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise TunnelError(
                    f"cloudflared exited before readiness with {self.process.returncode}"
                )
            try:
                line = self._lines.get(timeout=0.5)
            except queue.Empty:
                continue
            public_url = self.extract_public_url(line)
            if public_url is not None:
                threading.Thread(target=self._monitor, daemon=True).start()
                return public_url
        self.stop()
        raise TunnelError("cloudflared did not publish a Quick Tunnel URL in time")

    def _monitor(self) -> None:
        assert self.process is not None
        self.process.wait()
        if not self._stopping.is_set():
            os.kill(os.getpid(), signal.SIGTERM)

    def stop(self) -> None:
        self._stopping.set()
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)

    def __enter__(self) -> QuickTunnel:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.stop()
