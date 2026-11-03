"""Fail explicit environmental acceptance when any requested test is skipped.

Load with ``-p scripts.ci_metrics.require_execution``. Intended for serial manual
asset/GPU acceptance, including module-level skips and empty selections.
"""

from __future__ import annotations

import pytest

_skipped: set[str] = set()


def pytest_configure() -> None:
    _skipped.clear()


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.skipped:
        _skipped.add(report.nodeid)


def pytest_collectreport(report: pytest.CollectReport) -> None:
    if report.skipped:
        _skipped.add(report.nodeid)


def pytest_sessionfinish(session: pytest.Session) -> None:
    if _skipped or not session.testscollected:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_sep(
                "=", "Requested acceptance was not fully executed", red=True
            )
            for nodeid in sorted(_skipped):
                reporter.write_line(f"SKIPPED: {nodeid}", red=True)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
