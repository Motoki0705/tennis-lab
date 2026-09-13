import json
import sys
from pathlib import Path

import pytest

from src.synthetic_data_generation.scripts import compact_court_storage


def test_main_compacts_with_explicit_options_and_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    report_path = tmp_path / "reports" / "compaction.json"
    expected: dict[str, object] = {"bitwise_verified": True, "samples": 12}

    def fake_compact(
        actual_source: Path,
        actual_destination: Path,
        *,
        resume: bool = False,
        workers: int = 4,
    ) -> dict[str, object]:
        assert actual_source == source
        assert actual_destination == destination
        assert resume is True
        assert workers == 2
        return expected

    monkeypatch.setattr(compact_court_storage, "compact", fake_compact)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compact_court_storage",
            "--source",
            str(source),
            "--destination",
            str(destination),
            "--report",
            str(report_path),
            "--resume",
            "--workers",
            "2",
        ],
    )

    compact_court_storage.main()

    assert json.loads(report_path.read_text()) == expected
    assert json.loads(capsys.readouterr().out) == expected
