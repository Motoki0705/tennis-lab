"""A distinct human-confirmed owner schema with verifiable source diagnostics."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from src.synthetic_data_generation.alignment.contracts import AlignmentResult
from src.synthetic_data_generation.alignment.heatmaps import validate_line_heatmaps
from src.synthetic_data_generation.alignment.manual.geometry import build_manual_result
from src.synthetic_data_generation.alignment.manual.models import LayoutEdit
from src.synthetic_data_generation.alignment.manual.source import (
    ManualSource,
    file_digest,
)

CONFIRMATION_FILE = "manual-confirmation.json"
SOURCE_ARCHIVE_SCHEMA = "manual_alignment_source_archive_v1"


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected ordinary JSON file: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object.")
    return result


def load_manual_source(root: Path) -> ManualSource:
    with np.load(root / "ground-line-map.npz", allow_pickle=False) as archive:
        if (
            set(archive.files) != {"schema", "source_json"}
            or str(archive["schema"].item()) != SOURCE_ARCHIVE_SCHEMA
        ):
            raise ValueError("Invalid manual source archive.")
        return ManualSource.from_dict(json.loads(str(archive["source_json"].item())))


def geometry_payload(result: AlignmentResult) -> dict[str, Any]:
    return {
        "schema": "human_confirmed_court_geometry_v1",
        "metric_scene_adapter": result.metric_adapter.to_dict(),
        "layout": result.layout.to_dict(),
    }


def diagnostic_payload(result: AlignmentResult) -> dict[str, Any]:
    return {
        "schema": "manual_alignment_diagnostics_v1",
        "authority": "human_confirmed",
        "measurement": "64 endpoint-inclusive samples per regulation line; nearest measured projected evidence per camera partition; distances in corrected metres",
        "holdout_note": "Human editing may inspect all views; holdout is descriptive and no longer an independent acceptance test.",
        "heatmap_coordinates": "immutable source ground-plane UV; divide source distances by confirmed scale for corrected metres",
        "candidates": [candidate.to_dict() for candidate in result.candidates],
    }


def write_manual_outputs(
    root: Path,
    *,
    source_owner: Path,
    source: ManualSource,
    edit: LayoutEdit,
    confirmed_at: str,
    revision: str,
) -> AlignmentResult:
    if any(root.iterdir()):
        raise ValueError("Manual staging directory must be empty.")
    heatmaps = validate_line_heatmaps(source_owner / "line-heatmaps")
    result = build_manual_result(source, heatmaps, edit)
    shutil.copytree(source_owner / "line-heatmaps", root / "line-heatmaps")
    np.savez_compressed(
        root / "ground-line-map.npz",
        schema=np.asarray(SOURCE_ARCHIVE_SCHEMA),
        source_json=np.asarray(json.dumps(source.to_dict(), allow_nan=False)),
    )
    write_json(root / "alignment.json", result.to_dict())
    write_json(root / "court-geometry.json", geometry_payload(result))
    (root / "diagnostics").mkdir()
    write_json(root / "diagnostics" / "manual-metrics.json", diagnostic_payload(result))
    write_json(
        root / CONFIRMATION_FILE,
        {
            "schema": "manual_court_confirmation_v1",
            "human_confirmed": True,
            "confirmed_at": confirmed_at,
            "source_revision": revision,
            "layout": edit.model_dump(),
            "source_archive_sha256": file_digest(root / "ground-line-map.npz"),
            "heatmaps_sha256": file_digest(root / "line-heatmaps" / "heatmaps.npz"),
        },
    )
    validate_manual_outputs(root)
    return result


def validate_manual_outputs(root: Path) -> AlignmentResult:
    """Recompute geometry AND measured scores; never turn failed gates into passes."""
    expected = {
        "alignment.json",
        "ground-line-map.npz",
        "court-geometry.json",
        "diagnostics",
        "line-heatmaps",
        CONFIRMATION_FILE,
    }
    if {path.name for path in root.iterdir()} != expected or any(
        path.is_symlink() for path in root.rglob("*")
    ):
        raise ValueError("Manual alignment inventory mismatch.")
    confirmation = read_json(root / CONFIRMATION_FILE)
    if set(confirmation) != {
        "schema",
        "human_confirmed",
        "confirmed_at",
        "source_revision",
        "layout",
        "source_archive_sha256",
        "heatmaps_sha256",
    }:
        raise ValueError("Invalid manual confirmation fields.")
    if (
        confirmation["schema"] != "manual_court_confirmation_v1"
        or confirmation["human_confirmed"] is not True
    ):
        raise ValueError("Manual alignment requires explicit human confirmation.")
    from datetime import datetime

    if (
        datetime.fromisoformat(confirmation["confirmed_at"]).tzinfo is None
        or not confirmation["source_revision"]
    ):
        raise ValueError("Confirmation requires a timezone and source revision.")
    if confirmation["source_archive_sha256"] != file_digest(
        root / "ground-line-map.npz"
    ) or confirmation["heatmaps_sha256"] != file_digest(
        root / "line-heatmaps" / "heatmaps.npz"
    ):
        raise ValueError("Confirmed evidence digest mismatch.")
    source = load_manual_source(root)
    result = build_manual_result(
        source,
        validate_line_heatmaps(root / "line-heatmaps"),
        LayoutEdit.model_validate(confirmation["layout"]),
    )
    if read_json(root / "alignment.json") != result.to_dict():
        raise ValueError(
            "Manual alignment disagrees with confirmed placement and measured evidence."
        )
    if read_json(root / "court-geometry.json") != geometry_payload(result):
        raise ValueError("Manual court geometry disagrees with confirmed alignment.")
    if {path.name for path in (root / "diagnostics").iterdir()} != {
        "manual-metrics.json"
    } or read_json(root / "diagnostics" / "manual-metrics.json") != diagnostic_payload(
        result
    ):
        raise ValueError("Manual diagnostics disagree with measured evidence.")
    return result
