"""Contract for the notifier entry point used by systemd."""

from __future__ import annotations

import importlib
import shlex
from pathlib import Path

import pytest

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"


def test_toast_identifies_failed_unit_and_its_journal_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.syspath_prepend(str(OPS))
    notifier = importlib.import_module("lib.notify_unit_failure")
    unit = r"ops-worker@left\x2dside.service"
    toasts: list[tuple[str, str]] = []

    def send(title: str, message: str) -> None:
        toasts.append((title, message))

    monkeypatch.setattr(notifier, "send_windows_toast", send)
    assert notifier.main(["--", unit]) == 0
    ((title, body),) = toasts
    summary, command = body.splitlines()
    assert "tennis-lab" in title
    assert summary == f"{unit} failed."
    assert shlex.split(command) == ["journalctl", "--user", "-u", unit]
    output = capsys.readouterr()
    assert f"Windows toast submitted for {unit}" in output.out
    assert output.err == ""


def test_toast_failure_reports_unit_and_reason_to_journal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.syspath_prepend(str(OPS))
    notifier = importlib.import_module("lib.notify_unit_failure")
    unit = "tennis-lab-storage-tier.service"

    def reject(title: str, message: str) -> None:
        raise notifier.ToastError("Windows notifications are disabled")

    monkeypatch.setattr(notifier, "send_windows_toast", reject)
    assert notifier.main(["--", unit]) == 1
    output = capsys.readouterr()
    assert f"Windows toast failed for {unit}" in output.err
    assert "Windows notifications are disabled" in output.err
    assert output.out == ""
