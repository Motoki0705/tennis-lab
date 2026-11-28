from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import pytest

from src.tennis_scene.analytics.__main__ import main
from src.tennis_scene.analytics.db.connection import database_url
from src.tennis_scene.analytics.export import write_csv, write_csv_export, write_json


def test_json_and_csv_preserve_evidence_and_distinguish_null_from_empty_text() -> None:
    values = {
        "event_id": "e,02",
        "included": True,
        "detail": {"frames": [7, 11], "label": "打球"},
        "created_at": datetime(2026, 11, 29, 9, 4, tzinfo=timezone(timedelta(hours=9))),
        "note": "映像\n確認済み",
        "absent": None,
        "empty": "",
        "literal": r"\N",
    }
    json_out, csv_out = StringIO(), StringIO()
    write_json(json_out, [values])
    write_csv(csv_out, tuple(values), [values])
    row = json.loads(json_out.getvalue())[0]
    csv_row = next(csv.DictReader(StringIO(csv_out.getvalue())))
    assert row["created_at"] == csv_row["created_at"] == "2026-11-29T00:04:00Z"
    assert row["event_id"] == csv_row["event_id"] == "e,02"
    assert (
        row["detail"]
        == json.loads(csv_row["detail"])
        == {"frames": [7, 11], "label": "打球"}
    )
    assert row["note"] == csv_row["note"] == "映像\n確認済み"
    assert row["included"] is json.loads(csv_row["included"]) is True
    assert row["absent"] is None and csv_row["absent"] == r"\N"
    assert row["empty"] == csv_row["empty"] == ""
    assert row["literal"] == r"\N" and csv_row["literal"] == r"\\N"


def test_incomplete_csv_export_leaves_no_published_directory(tmp_path: Path) -> None:
    destination = tmp_path / "export"
    with pytest.raises(KeyError, match="source_event_id|clip_id"):
        write_csv_export(destination, {"points": [], "ball_events": [{"id": 7}]})
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("existing", ["directory", "dangling-symlink"])
def test_csv_cli_rejects_existing_destination_before_connecting(
    existing: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    destination = tmp_path / "export"
    if existing == "directory":
        destination.mkdir()
    else:
        destination.symlink_to(tmp_path / "missing")

    assert (
        main(
            [
                "export",
                "--match-context",
                "1",
                "--format",
                "csv",
                "--output-dir",
                str(destination),
            ]
        )
        == 1
    )
    assert (
        f"Export destination already exists: {destination}" in capsys.readouterr().err
    )
    assert list(tmp_path.iterdir()) == [destination]


def test_metrics_cli_and_missing_database_configuration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert main(["metrics", "list"]) == 0
    assert "rally.contact_count\t1" in capsys.readouterr().out.splitlines()
    assert main(["metrics", "export", "--format", "json"]) == 0
    definitions = json.loads(capsys.readouterr().out)
    assert main(["metrics", "export", "--format", "csv"]) == 0
    rows = list(csv.DictReader(StringIO(capsys.readouterr().out)))
    assert [row["metric_id"] for row in rows] == [
        row["metric_id"] for row in definitions
    ]
    assert json.loads(rows[0]["rules"]) == definitions[0]["rules"]
    assert main(["query", "points"]) == 1
    result = capsys.readouterr()
    assert not result.out and "Set DATABASE_URL" in result.err


def test_database_url_selects_psycopg_and_rejects_other_backends() -> None:
    url = database_url("postgresql://user:p%25ss@localhost/db")
    assert url.drivername == "postgresql+psycopg"
    assert url.password == "p%ss"
    with pytest.raises(ValueError, match="must use PostgreSQL"):
        database_url("sqlite:///analytics.db")
