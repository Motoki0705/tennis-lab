"""Process-local connection-startup repair for the pinned host Colab CLI."""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.metadata
import math
from pathlib import Path
from typing import Any

SOURCE_HASHES = {
    "jupyter_kernel_client.wsclient": "97744a71e4c1290449ee85a922a5a538f86a2b108934966a76c3b028fa5a4a6d",
    "colab_cli.runtime": "e95adf8dc98c9a08dce1fff035839863cbe80c9571d36634c3bc3d0cfcf06bcb",
    "colab_cli.commands.execution": "ad82ab0798ca28b598c807d0490043d80751a25a61f0260ee5291575f2fee759",
    "colab_cli.cli": "46918b27d3e8d0f7fa99c4e9861e600e36ed144fdb4f4a6743cb4884ce861cf7",
}
VERSIONS = {"google-colab-cli": "0.7.4", "jupyter-kernel-client": "0.9.0"}


def verify_client() -> None:
    """Refuse different installed code rather than adapting it implicitly."""
    for package, expected in VERSIONS.items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"Unqualified Notebook client version: {package}")
    for module_name, expected in SOURCE_HASHES.items():
        module = importlib.import_module(module_name)
        module_file = module.__file__
        if (
            module_file is None
            or hashlib.sha256(Path(module_file).read_bytes()).hexdigest() != expected
        ):
            raise RuntimeError(f"Unqualified Notebook client source: {module_name}")


def install_initial_wait(client_class: Any, *, startup_seconds: float = 60.0) -> None:
    """Wait for this connection once; never retry or restart a kernel/dispatch."""
    if not math.isfinite(startup_seconds) or startup_seconds <= 0:
        raise ValueError("Connection startup bound must be finite and positive")
    original = client_class.start_channels
    if getattr(original, "court_initial_wait", False):
        raise RuntimeError("Notebook startup repair already installed")

    @functools.wraps(original)
    def start_channels(self: Any, *args: Any, **kwargs: Any) -> Any:
        normal_timeout = self.timeout
        try:
            self.timeout = startup_seconds
            result = original(self, *args, **kwargs)
        finally:
            self.timeout = normal_timeout
        if not self.connection_ready.is_set():
            # Stop only this client's channels/socket; never shut down the kernel.
            self.stop_channels()
            raise TimeoutError(
                "Notebook WebSocket did not become ready within the startup bound"
            )
        return result

    start_channels.court_initial_wait = True  # type: ignore[attr-defined]
    client_class.start_channels = start_channels


def main() -> None:
    verify_client()
    websocket_client = importlib.import_module("jupyter_kernel_client.wsclient")
    install_initial_wait(websocket_client.KernelWebSocketClient)
    importlib.import_module("colab_cli.cli").main()


if __name__ == "__main__":
    main()
