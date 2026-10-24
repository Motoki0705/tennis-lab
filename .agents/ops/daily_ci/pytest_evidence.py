"""Explicitly loaded pytest plugin: record master reports, never parse log text."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from .evidence import read_json


class Reports:
    def __init__(self, manifest: Path) -> None:
        self.directory = manifest.parent
        self.data: dict[str, Any] = {
            "schema_version": 1,
            "identity": read_json(manifest)["identity"],
            "collected": [],
            "reports": {},
            "collection_errors": [],
            "collection_consistent": True,
            "finished": False,
            "exit_code": -1,
        }
        self.events = (self.directory / "pytest.events.jsonl").open(
            "w", encoding="utf-8"
        )
        self.event("identity", self.data["identity"])

    def event(self, kind: str, value: Any) -> None:
        self.events.write(
            json.dumps({"kind": kind, "value": value}, ensure_ascii=False) + "\n"
        )
        self.events.flush()

    def collection(self, ids: list[str]) -> None:
        if self.data["collected"] and self.data["collected"] != ids:
            self.data["collection_consistent"] = False
        self.data["collected"] = ids
        self.event("collected", ids)

    def pytest_collection_finish(self, session: Any) -> None:
        if session.items:
            self.collection([item.nodeid for item in session.items])

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node: Any, ids: list[str]) -> None:
        self.collection(ids)

    def pytest_collectreport(self, report: Any) -> None:
        if report.failed:
            self.data["collection_errors"].append(report.nodeid)
            self.event("collection_error", report.nodeid)

    def pytest_runtest_logreport(self, report: Any) -> None:
        phase = {
            "when": report.when,
            "outcome": report.outcome,
            "wasxfail": hasattr(report, "wasxfail"),
        }
        self.data["reports"].setdefault(report.nodeid, []).append(phase)
        self.event("report", {"nodeid": report.nodeid, **phase})

    def pytest_sessionfinish(self, session: Any, exitstatus: int) -> None:
        self.data.update(finished=True, exit_code=int(exitstatus))
        self.event("finish", {"exit_code": int(exitstatus)})
        self.events.close()
        path = self.directory / "pytest.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


def pytest_configure(config: Any) -> None:
    # xdist forwards the worker's TestReport objects to these master hooks.
    if not hasattr(config, "workerinput"):
        config.pluginmanager.register(Reports(Path(os.environ["DAILY_CI_MANIFEST"])))
