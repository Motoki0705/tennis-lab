"""Process-group discovery while /proc entries disappear during a scan."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops/lib"))
opsproc: Any = importlib.import_module("opsproc")


@pytest.mark.parametrize("vanished", [FileNotFoundError, ProcessLookupError])
def test_exiting_process_does_not_abort_group_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vanished: type[OSError]
) -> None:
    for pid, state, group in (
        (101, "S", 555),
        (102, "S", 555),
        (103, "Z", 555),
        (104, "S", 777),
    ):
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "stat").write_text(
            f"{pid} (comm with ) and spaces) {state} 1 {group} 0\n"
        )
    (tmp_path / "self").mkdir()
    read_text = Path.read_text

    def read(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == tmp_path / "102/stat":
            raise vanished("process exited")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert opsproc._live_group_members(555, tmp_path) == [101]


def test_unexpected_proc_read_failure_is_not_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "101").mkdir()

    def denied(path: Path, *args: Any, **kwargs: Any) -> str:
        raise PermissionError("unexpected permissions")

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(PermissionError, match="unexpected permissions"):
        opsproc._live_group_members(555, tmp_path)


def test_process_group_reads_caller_owned_stdin_file(tmp_path: Path) -> None:
    payload = "日本語の入力\n" * 20000
    source = tmp_path / "stdin.txt"
    source.write_text(payload)
    with source.open() as stdin:
        output = opsproc.run_process_group(
            [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"],
            cwd=tmp_path,
            timeout=10,
            stdin_file=stdin,
        )
    assert output == payload
