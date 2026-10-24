"""Append-only observations, independent per process, including interrupted runs."""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, TextIO

import pytest

_IMPORTED_AT = time.time()
_STREAM: TextIO | None = None
_WRITE_SECONDS = 0.0
_ROLE = os.environ.get("PYTEST_XDIST_WORKER", "controller")


def _torch_threads() -> dict[str, int] | None:
    loaded = sys.modules.get("torch")
    if loaded is None:
        return None
    return {
        "intraop": loaded.get_num_threads(),
        "interop": loaded.get_num_interop_threads(),
    }


def _emit(event: str, **data: Any) -> None:
    global _WRITE_SECONDS
    if _STREAM is None:
        return
    started = time.perf_counter()
    _STREAM.write(json.dumps({"event": event, "at": time.time(), **data}) + "\n")
    _STREAM.flush()
    _WRITE_SECONDS += time.perf_counter() - started


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--ci-metrics-dir", help="New directory for CI measurement events")


def pytest_configure(config: pytest.Config) -> None:
    global _STREAM
    destination = config.getoption("--ci-metrics-dir")
    if destination is None:
        return
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    _STREAM = (root / f"events-{_ROLE}-{os.getpid()}.jsonl").open("x", encoding="utf-8")
    _emit("configured", role=_ROLE, pid=os.getpid(), imported_at=_IMPORTED_AT)


def pytest_sessionstart(session: pytest.Session) -> None:
    _emit("session_start")


def pytest_collection_finish(session: pytest.Session) -> None:
    _emit("collection", nodeids=[item.nodeid for item in session.items])


def pytest_deselected(items: list[pytest.Item]) -> None:
    _emit("deselected", nodeids=[item.nodeid for item in items])


def pytest_collectreport(report: pytest.CollectReport) -> None:
    if report.failed or report.skipped:
        _emit(
            "collection_result",
            nodeid=report.nodeid,
            outcome=report.outcome,
            detail=str(report.longrepr),
        )


@pytest.hookimpl(optionalhook=True)  # type: ignore[untyped-decorator]
def pytest_xdist_node_collection_finished(node: Any, ids: list[str]) -> None:
    _emit("worker_collection", worker=node.gateway.id, nodeids=ids)


@pytest.hookimpl(tryfirst=True)  # type: ignore[untyped-decorator]
def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    # xdist delivers reports to the controller; do not count worker copies twice.
    if _ROLE != "controller":
        # xdist serializes custom report fields after this hook. Observe existing
        # per-test thread changes without importing torch or changing its state.
        report.ci_torch_threads = _torch_threads()
        return
    _emit(
        "phase",
        nodeid=report.nodeid,
        phase=report.when,
        outcome=report.outcome,
        duration_s=report.duration,
        start=report.start,
        stop=report.stop,
        worker=getattr(report, "worker_id", "controller"),
        wasxfail=getattr(report, "wasxfail", None),
        torch_threads=getattr(report, "ci_torch_threads", _torch_threads()),
    )


@pytest.hookimpl(wrapper=True)  # type: ignore[untyped-decorator]
def pytest_fixture_setup(
    fixturedef: pytest.FixtureDef[Any],
    request: pytest.FixtureRequest,
) -> Generator[None, Any, Any]:
    started = time.perf_counter()
    outcome = "passed"
    try:
        return (yield)
    except BaseException:
        outcome = "error"
        raise
    finally:
        _emit(
            "fixture",
            name=fixturedef.argname,
            scope=fixturedef.scope,
            source=f"{fixturedef.func.__module__}.{fixturedef.func.__qualname__}",
            nodeid=request.node.nodeid,
            outcome=outcome,
            setup_s=time.perf_counter() - started,
        )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _emit(
        "session_finish",
        exitstatus=int(exitstatus),
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        event_write_s=_WRITE_SECONDS,
    )


def pytest_unconfigure(config: pytest.Config) -> None:
    global _STREAM
    if _STREAM is not None:
        _STREAM.close()
        _STREAM = None
