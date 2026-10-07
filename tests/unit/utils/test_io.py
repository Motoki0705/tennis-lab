from __future__ import annotations

from pathlib import Path

from src.utils.io import (
    JSONDict,
    read_jsonl,
    relative_path,
    save_json_atomic,
    write_jsonl,
)


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "records.jsonl"
    records: list[JSONDict] = [{"id": "a", "text": "tennis"}, {"id": "b", "value": 2}]

    write_jsonl(path, records)

    assert read_jsonl(path) == records


def test_save_json_atomic_replaces_target(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    save_json_atomic({"value": 1}, path)
    save_json_atomic({"value": 2}, path)

    assert path.read_text(encoding="utf-8") == '{\n  "value": 2\n}\n'
    assert not path.with_suffix(".json.tmp").exists()


def test_relative_path_resolves_paths(tmp_path: Path) -> None:
    root = tmp_path / "root"
    child = root / "a" / "b.txt"
    child.parent.mkdir(parents=True)
    child.write_text("x", encoding="utf-8")

    assert relative_path(child, root) == "a/b.txt"


def test_strict_atomic_artifact_json_keeps_previous_value_on_failure(
    tmp_path: Path,
) -> None:
    from dataclasses import dataclass

    import numpy as np
    import pytest

    from src.utils.io import json_value, load_json, write_json_atomic

    @dataclass
    class Record:
        location: Path
        samples: np.ndarray

    path = tmp_path / "receipt.json"
    record = Record(tmp_path, np.array([1, 2]))
    write_json_atomic(path, {"record": record, "score": np.float32(0.5)})
    expected = path.read_bytes()
    assert load_json(path) == {
        "record": {"location": str(tmp_path), "samples": [1, 2]},
        "score": 0.5,
    }
    with pytest.raises(ValueError, match="JSON"):
        write_json_atomic(path, {"invalid": np.nan})
    with pytest.raises(TypeError, match="Unsupported"):
        write_json_atomic(path, {"invalid": object()})
    assert path.read_bytes() == expected
    assert list(tmp_path.iterdir()) == [path]
    assert json_value((np.int64(3),)) == [3]
