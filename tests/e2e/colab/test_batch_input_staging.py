"""Batch immutable sibling files without weakening per-input provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.colab.workflow import remote_runner


def request() -> dict[str, Any]:
    return {
        "drive": {"mode": "rclone", "root": "tennis_lab", "remote": "gdrive"},
        "job": {
            "inputs": [
                {
                    "source": "data/a.bin",
                    "destination": "data/a.bin",
                    "writable": False,
                },
                {
                    "source": "data/b.bin",
                    "destination": "ckpt/b.bin",
                    "writable": False,
                },
            ]
        },
    }


def test_batch_copy_uses_one_inventory_and_copy_with_exact_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    payloads = {"a.bin": b"first", "b.bin": b"second"}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if "lsjson" in argv:
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=json.dumps(
                    [
                        {"Path": name, "IsDir": False, "Size": len(value)}
                        for name, value in payloads.items()
                    ]
                ),
            )
        assert argv[3] == "copy"
        names = Path(argv[argv.index("--files-from-raw") + 1]).read_text().splitlines()
        assert names == list(payloads)
        destination = Path(argv[5])
        destination.mkdir()
        for name in names:
            (destination / name).write_bytes(payloads[name])
        return subprocess.CompletedProcess(argv, 0, stdout="")

    monkeypatch.setattr(remote_runner, "_run", fake_run)
    result = remote_runner._stage_inputs(
        request(), tmp_path, tmp_path / "rclone.conf", reset_existing=False
    )
    assert len(calls) == 2
    assert (tmp_path / "ckpt/b.bin").read_bytes() == b"second"
    assert [item["source"] for item in result] == ["data/a.bin", "data/b.bin"]
    assert result[0]["entries"][0]["sha256"] == hashlib.sha256(b"first").hexdigest()
    assert not list(tmp_path.glob(".colab-inputs-*"))


@pytest.mark.parametrize(
    "inventory,message",
    [
        (
            [
                {"Path": "a.bin", "IsDir": False, "Size": 1},
                {"Path": "a.bin", "IsDir": False, "Size": 1},
            ],
            "duplicate",
        ),
        ([{"Path": "a.bin", "IsDir": False, "Size": 1}], "Missing"),
    ],
)
def test_batch_rejects_ambiguous_or_missing_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inventory: list[dict[str, object]],
    message: str,
) -> None:
    monkeypatch.setattr(
        remote_runner,
        "_run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps(inventory)
        ),
    )
    with pytest.raises(remote_runner.RemoteWorkflowError, match=message):
        remote_runner._stage_inputs(
            request(), tmp_path, tmp_path / "config", reset_existing=False
        )


def test_writable_inputs_keep_individual_copy_semantics(tmp_path: Path) -> None:
    value = request()
    value["job"]["inputs"][0]["writable"] = True
    assert (
        remote_runner._batch_stage_rclone_files(
            value, tmp_path, tmp_path / "config", reset_existing=False
        )
        is False
    )
