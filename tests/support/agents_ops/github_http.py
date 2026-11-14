"""HTTP bridge for existing executable fake-gh scenarios (never production).

The public CLI uses the real pooled HTTP transport. Only the server's backend
executes the existing fake scenario so its race/failure controls stay reusable.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:
        pass

    def handle_request(self) -> None:
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        executable = shutil.which("gh")
        if executable is None or not Path(executable).resolve().is_relative_to(
            Path(tempfile.gettempdir()).resolve()
        ):
            self.send_error(
                503, "fake gh must be installed in the temporary test directory"
            )
            return
        endpoint = self.path.removeprefix("/")
        method = self.command
        payload = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if endpoint == "graphql":
            args = ["gh", "api", "graphql", "--input", "-"]
        else:
            args = ["gh", "api"]
            if method == "GET":
                args.append("--include")
            args += [
                "--method",
                method,
                "--header",
                "Accept: application/vnd.github+json",
                "--header",
                "X-GitHub-Api-Version: 2022-11-28",
                endpoint,
            ]
            if method != "GET":
                args += ["--input", "-"]
        result = subprocess.run(
            args,
            input=payload.decode() if payload else None,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        output = result.stdout.replace("\r\n", "\n")
        status = re.match(r"HTTP/\S+ (\d{3})", output)
        if status:
            code = int(status[1])
            _, separator, body = output.partition("\n\n")
            if not separator:
                self.connection.sendall(output.encode())
                self.close_connection = True
                return
        elif result.returncode:
            # A scenario failure is an explicit server error. Real statusless
            # disconnects are separately tested with socket fault injection.
            code, body = 503, json.dumps({"message": result.stderr.strip()})
        else:
            code, body = 200, output
        encoded = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    do_GET = do_POST = do_PATCH = do_PUT = do_DELETE = handle_request


class Bridge:
    def __init__(self) -> None:
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

    @property
    def url(self) -> str:
        with self.lock:
            if self.server is None:
                self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                self.thread = threading.Thread(
                    target=self.server.serve_forever,
                    kwargs={"poll_interval": 0.01},
                    daemon=True,
                )
                self.thread.start()
            return f"http://127.0.0.1:{self.server.server_port}/"

    def close(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            assert self.thread is not None
            self.thread.join(timeout=5)


def install(module: Any) -> Bridge:
    bridge = Bridge()
    # Importing a client is common in non-GitHub commands too. Start the server
    # only for actual I/O, so process-lifecycle tests don't acquire extra threads.
    original = getattr(
        module.HTTPClient.request, "_http_bridge_original", module.HTTPClient.request
    )

    def request(self: Any, *args: Any, **kwargs: Any) -> Any:
        module._API_URL = bridge.url
        return original(self, *args, **kwargs)

    wrapper: Any = request
    wrapper._http_bridge_original = original
    module.HTTPClient.request = wrapper
    module._token = lambda: "test-github-http-token"
    # localhost fixtures must not use a proxy configured on the runner.
    os.environ["NO_PROXY"] = ",".join(
        filter(None, [os.environ.get("NO_PROXY"), "127.0.0.1"])
    )
    return bridge
