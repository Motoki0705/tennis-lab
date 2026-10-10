"""A new connection never overwrites credentials or prints client secrets."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def test_oauth_import_is_private_exclusive_and_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/skills/tennis-drive/scripts")
    )
    module = importlib.import_module("configure_oauth")
    source = tmp_path / "client.json"
    source.write_text(json.dumps({"installed": {
        "client_id": "test.apps.googleusercontent.com", "client_secret": "private-value"
    }}))
    destination = tmp_path / "private/config"
    receipt = module.prepare(source, destination, "gdrive")
    assert "private-value" not in json.dumps(receipt)
    assert receipt["status"] == "needs_browser_auth"
    assert destination.stat().st_mode & 0o777 == 0o600
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        module.prepare(source, destination, "gdrive")
    assert destination.read_bytes() == before
    source.write_text(json.dumps({"web": {"client_secret": "private-value"}}))
    with pytest.raises(ValueError, match="Desktop"):
        module.prepare(source, tmp_path / "other", "gdrive")
