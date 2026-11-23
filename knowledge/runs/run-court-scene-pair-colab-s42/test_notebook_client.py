"""Fault contracts for the host-only Notebook connection startup repair."""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "court_notebook_client", Path(__file__).with_name("notebook_client.py")
)
assert _spec and _spec.loader
client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(client)


def channel_class(*, error=None):
    class Channel:
        def __init__(self):
            self.timeout = 0.005
            self.connection_ready = threading.Event()
            self.calls = []
            self.closed = False

        def start_channels(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            if error:
                raise error
            self.connection_ready.wait(self.timeout)
            return "original-result"

        def stop_channels(self):
            self.closed = True

    return Channel


def test_delayed_connection_is_waited_once_and_normal_timeout_restored():
    cls = channel_class()
    original = cls()
    original.start_channels()
    assert not original.connection_ready.is_set()  # Original premature return.
    client.install_initial_wait(cls, startup_seconds=0.3)
    repaired = cls()
    thread = threading.Thread(
        target=lambda: (threading.Event().wait(0.03), repaired.connection_ready.set())
    )
    thread.start()
    try:
        assert repaired.start_channels(shell=False) == "original-result"
        assert repaired.connection_ready.is_set()
        assert repaired.calls == [((), {"shell": False})]
        assert repaired.timeout == 0.005 and not repaired.closed
    finally:
        thread.join(1)


def test_never_ready_fails_and_closes_only_client():
    cls = channel_class()
    client.install_initial_wait(cls, startup_seconds=0.01)
    obj = cls()
    with pytest.raises(TimeoutError, match="startup bound"):
        obj.start_channels()
    assert len(obj.calls) == 1 and obj.closed and obj.timeout == 0.005


def test_original_error_is_preserved_without_retry():
    error = ConnectionError("fixture handshake rejected")
    cls = channel_class(error=error)
    client.install_initial_wait(cls)
    obj = cls()
    with pytest.raises(ConnectionError) as found:
        obj.start_channels()
    assert found.value is error and len(obj.calls) == 1 and obj.timeout == 0.005


@pytest.mark.parametrize("bound", [0, -1, float("inf"), float("nan")])
def test_invalid_bound_fails(bound):
    with pytest.raises(ValueError):
        client.install_initial_wait(channel_class(), startup_seconds=bound)


def test_repeated_install_fails():
    cls = channel_class()
    client.install_initial_wait(cls)
    with pytest.raises(RuntimeError, match="already installed"):
        client.install_initial_wait(cls)


def test_unqualified_version_and_tampered_source_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(
        client.importlib.metadata, "version", lambda name: "unqualified"
    )
    with pytest.raises(RuntimeError, match="version"):
        client.verify_client()
    monkeypatch.setattr(
        client.importlib.metadata, "version", lambda name: client.VERSIONS[name]
    )
    source = tmp_path / "tampered.py"
    source.write_text("different installed source")
    monkeypatch.setattr(
        client.importlib,
        "import_module",
        lambda name: SimpleNamespace(__file__=str(source)),
    )
    with pytest.raises(RuntimeError, match="source"):
        client.verify_client()
