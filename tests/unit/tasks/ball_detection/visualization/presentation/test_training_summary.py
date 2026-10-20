"""Compact measured evidence stays reproducible without the deleted raw logs."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from assets.ball_detection.learning_stages.export_scalars import export_scalars
from src.tasks.ball_detection.visualization.presentation.training_summary import (
    read_training_summary,
)
from src.utils.paths import PROJECT_ROOT

_ASSET = PROJECT_ROOT / "assets/ball_detection/learning_stages"
_BUNDLES = {
    "pretraining": PROJECT_ROOT / "knowledge/runs/run-i1050-convnext-v2-pretrain-s42",
    "posttraining": PROJECT_ROOT
    / "knowledge/runs/run-i1049-convnext-query-posttrain-s42",
}


def _record_source(path: Path, bundle: Path) -> None:
    manifest = json.loads((bundle / "artifact-manifest.json").read_text())
    (record,) = manifest["files"]
    record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    record["size_bytes"] = path.stat().st_size
    (bundle / "artifact-manifest.json").write_text(json.dumps(manifest))


@pytest.fixture
def sources(tmp_path: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    """Independent JSONL inputs with extra fields that must not be retained."""
    raw_files, bundles = {}, {}
    measured = read_training_summary(_ASSET / "scalars.json", _BUNDLES)
    for stage, source in _BUNDLES.items():
        bundle = tmp_path / stage
        bundle.mkdir()
        shutil.copyfile(
            source / "artifact-manifest.json", bundle / "artifact-manifest.json"
        )
        raw = bundle / "original.jsonl"
        raw.write_text(
            "\n".join(
                json.dumps(
                    {**row, "unused_timing": index + 0.25, "unused_vector": [1, 2]}
                )
                for index, row in enumerate(measured[stage])
            )
            + "\n"
        )
        _record_source(raw, bundle)
        raw_files[stage], bundles[stage] = raw, bundle
    return raw_files, bundles


def test_extraction_preserves_every_selected_value_and_never_changes_sources(
    sources: tuple[dict[str, Path], dict[str, Path]], tmp_path: Path
) -> None:
    raw_files, bundles = sources
    before = {stage: path.read_bytes() for stage, path in raw_files.items()}
    output = tmp_path / "summary"
    export_scalars(raw_files, bundles, output)
    result = read_training_summary(output / "scalars.json", bundles)
    original = read_training_summary(_ASSET / "scalars.json", _BUNDLES)
    assert result == original
    assert {stage: path.read_bytes() for stage, path in raw_files.items()} == before
    receipt = json.loads((output / "derivation.json").read_text())
    assert all(source["count"] == 1200 for source in receipt["sources"].values())
    assert (output / "scalars.json").stat().st_size <= 256 * 1024
    with pytest.raises(FileExistsError, match="already exists"):
        export_scalars(raw_files, bundles, output)


@pytest.mark.parametrize(
    "damage,message",
    [
        ("hash", "recorded source"),
        ("size", "recorded source"),
        ("count", "column/count mismatch"),
        ("step", "step mismatch"),
        ("missing_column", "Missing training columns"),
    ],
)
def test_extraction_rejects_source_count_and_step_mismatches_before_writing(
    sources: tuple[dict[str, Path], dict[str, Path]],
    tmp_path: Path,
    damage: str,
    message: str,
) -> None:
    raw_files, bundles = sources
    raw, bundle = raw_files["pretraining"], bundles["pretraining"]
    if damage == "size":
        manifest = json.loads((bundle / "artifact-manifest.json").read_text())
        manifest["files"][0]["size_bytes"] += 1
        (bundle / "artifact-manifest.json").write_text(json.dumps(manifest))
    elif damage == "hash":
        raw.write_bytes(raw.read_bytes() + b"\n")
    else:
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        if damage == "count":
            rows.pop()
        elif damage == "step":
            rows[1]["global_step"] += 1
        else:
            del rows[0]["train_loss"]
        raw.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        _record_source(raw, bundle)
    output = tmp_path / "summary"
    with pytest.raises(ValueError, match=message):
        export_scalars(raw_files, bundles, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "damage,message",
    [
        ("hash", "SHA-256 differs"),
        ("source", "source/columns mismatch"),
        ("columns", "column/count mismatch"),
        ("count", "column/count mismatch"),
        ("step", "step mismatch"),
        ("last_step", "step mismatch"),
        ("step_type", "Invalid training scalar"),
        ("schema", "summary schema"),
    ],
)
def test_runtime_rejects_changed_assets_and_inconsistent_derivation(
    tmp_path: Path, damage: str, message: str
) -> None:
    path = tmp_path / "scalars.json"
    payload: dict[str, Any] = json.loads((_ASSET / "scalars.json").read_text())
    receipt = json.loads((_ASSET / "derivation.json").read_text())
    run = payload["runs"]["pretraining"]
    if damage == "hash":
        run["columns"]["train_loss"][0] += 1
    elif damage == "source":
        run["source_sha256"] = "0" * 64
    elif damage == "columns":
        del run["columns"]["train_loss"]
    elif damage == "count":
        receipt["sources"]["pretraining"]["count"] -= 1
    elif damage == "step":
        run["columns"]["global_step"][1] += 1
    elif damage == "last_step":
        receipt["sources"]["pretraining"]["steps"]["last"] += 1
    elif damage == "schema":
        payload["schema_version"] = True
    else:
        run["columns"]["global_step"][0] = 50.0
    path.write_text(json.dumps(payload))
    if damage != "hash":
        receipt["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name("derivation.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=message):
        read_training_summary(path, _BUNDLES)
