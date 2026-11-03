"""Hand-calculated evaluation truth, with no dependency on local sample assets."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tennis_scene.ground_truth import (
    ReferenceIdentity,
    UserReference,
    load_user_reference,
)

FIXTURE = Path(__file__).parent / "fixtures" / "reference.json"


def identity() -> ReferenceIdentity:
    return ReferenceIdentity(
        "synthetic/asymmetric",
        ("right", "left"),
        3,
        29.97,
        (80, 40),
        {"right": "a" * 64, "left": "b" * 64},
    )


@pytest.fixture
def document(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    shutil.copytree(FIXTURE.parent / "source", tmp_path / "source")
    return tmp_path / "reference.json", json.loads(FIXTURE.read_text())


def read_changed(document: tuple[Path, dict[str, Any]]) -> UserReference:
    path, data = document
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_user_reference(
        path,
        expected=identity(),
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_explicit_axes_and_user_approved_estimates() -> None:
    truth = load_user_reference(
        FIXTURE,
        expected=identity(),
        expected_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
    )
    # JSON row order is deliberately unrelated to either axis.
    selected = truth.select(camera_ids=("left", "right"), frame_indices=(1, 0, 2))
    assert selected.camera_ids == ("left", "right")
    assert selected.frame_indices == (1, 0, 2)
    np.testing.assert_equal(
        selected.uv_px,
        [
            [[np.nan, np.nan], [61.25, 5.5], [np.nan, np.nan]],
            [[17.25, 8.5], [3.5, 27.0], [np.nan, np.nan]],
        ],
    )
    np.testing.assert_array_equal(
        selected.valid_for_scoring, [[False, True, False], [True, True, False]]
    )
    np.testing.assert_array_equal(
        selected.reference_kind,
        [
            ["unknown", "observed", "occluded"],
            ["estimated", "observed", "excluded_extrapolation"],
        ],
    )
    assert truth.identity.clip_id == "synthetic/asymmetric"
    assert truth.identity.video_sha256 == {"right": "a" * 64, "left": "b" * 64}
    estimate = truth.position("right", 1)
    assert estimate.source_status == "unknown"
    assert estimate.estimation == {
        "position_source": "estimated",
        "estimate_method": "poly_3_10",
        "estimate_quality": "lower",
        "estimate_model_spread_px": "6.37",
    }
    assert truth.position("right", 2).uv_px is None
    assert [
        (e["camera_id"], e["frame_index"], e["source_file"], e["note"])
        for e in truth.events
    ] == [
        ("right", 0, "annotations.csv", "Contact uncertain by one frame."),
        ("left", 1, "annotations.csv", "Same event, another camera."),
    ]
    assert len(truth.estimation_segments) == 1
    assert truth.estimation_segments[0]["max_model_spread_px"] == "6.37"
    assert truth.estimation_segments[0]["validation_cases"] == "141"
    assert truth.metadata["provenance"]["kind"] == "user_supplied_reference"
    with pytest.raises(TypeError):
        truth.metadata["provenance"]["kind"] = "model"
    with pytest.raises(ValueError, match="read-only"):
        selected.valid_for_scoring[0, 0] = True


@pytest.mark.parametrize(
    "change, message",
    [
        ({"clip_id": "another/clip"}, "clip_id"),
        ({"camera_ids": ("left", "right")}, "camera_ids"),
        ({"num_frames": 4}, "num_frames"),
        ({"fps": 30.0}, "fps"),
        ({"source_size_wh": (40, 80)}, "source_size_wh"),
        ({"video_sha256": {"right": "b" * 64, "left": "a" * 64}}, "video_sha256"),
    ],
)
def test_identity_must_match_caller(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_user_reference(
            FIXTURE,
            expected=replace(identity(), **change),
            expected_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        )


@pytest.mark.parametrize(
    "defect, message",
    [
        ("duplicate_row", "duplicate camera/frame"),
        ("missing_row", "full camera/frame"),
        ("unknown_camera", "camera_id"),
        ("frame_outside", "frame_index"),
        ("boolean_frame", "frame_index"),
        ("estimate_missing", "scoring"),
        ("estimate_as_observed", "source_status|estimation"),
        ("tail_valid", "scoring|exclusion"),
        ("tail_relabelled", "exclusion"),
        ("invalid_uv", "scoring"),
        ("negative_uv", "uv_px"),
        ("uv_outside", "uv_px"),
        ("uv_wrong_length", "uv_px"),
        ("uv_boolean", "finite number"),
        ("uv_string", "finite number"),
        ("nan", "non-finite"),
        ("overflow", "finite number"),
        ("missing_field", "expected fields"),
        ("extra_field", "expected fields"),
        ("wrong_schema", "schema"),
        ("wrong_source_kind", "provenance.kind"),
        ("missing_provenance", "estimation"),
        ("wrong_counts", "counts_by_camera"),
        ("event_camera", "camera_id"),
        ("segment_outside", "segment"),
        ("duplicate_segment", "duplicate segment"),
        ("source_file_unknown", "source_file"),
        ("source_hash", "source.*SHA-256"),
        ("source_size", "source.*size"),
        ("source_path_escape", "source.*path"),
    ],
)
def test_rejects_invalid_reference(document, defect: str, message: str) -> None:
    _, data = document
    rows = data["positions"]
    estimate, observed, excluded = rows[1], rows[2], rows[3]
    source = data["provenance"]["source_files"][0]
    if defect == "duplicate_row":
        rows.append(dict(observed))
    elif defect == "missing_row":
        rows.pop()
    elif defect == "unknown_camera":
        observed["camera_id"] = "absent"
    elif defect == "frame_outside":
        observed["frame_index"] = 3
    elif defect == "boolean_frame":
        observed["frame_index"] = True
    elif defect == "estimate_missing":
        estimate["valid_for_scoring"] = False
    elif defect == "estimate_as_observed":
        estimate["reference_kind"] = "observed"
    elif defect == "tail_valid":
        excluded.update(valid_for_scoring=True, uv_px=[78, 39])
    elif defect == "tail_relabelled":
        excluded.update(
            reference_kind="estimated", valid_for_scoring=True, uv_px=[78, 39]
        )
    elif defect == "invalid_uv":
        rows[0]["uv_px"] = [1, 2]
    elif defect == "negative_uv":
        observed["uv_px"] = [-0.01, 2]
    elif defect == "uv_outside":
        observed["uv_px"] = [80, 2]
    elif defect == "uv_wrong_length":
        observed["uv_px"] = [1, 2, 3]
    elif defect == "uv_boolean":
        observed["uv_px"] = [True, 2]
    elif defect == "uv_string":
        observed["uv_px"] = ["1", 2]
    elif defect == "nan":
        observed["uv_px"] = [float("nan"), 2]
    elif defect == "overflow":
        observed["uv_px"] = [10**400, 2]
    elif defect == "missing_field":
        observed.pop("valid_for_scoring")
    elif defect == "extra_field":
        observed["confidence"] = 1.0
    elif defect == "wrong_schema":
        data["schema"] = "another.v1"
    elif defect == "wrong_source_kind":
        data["provenance"]["kind"] = "model"
    elif defect == "missing_provenance":
        estimate["estimation"] = None
    elif defect == "wrong_counts":
        data["counts_by_camera"]["right"]["estimated"] = 0
    elif defect == "event_camera":
        data["events"][0]["camera_id"] = "absent"
    elif defect == "segment_outside":
        data["estimation_segments"][0]["end_frame"] = "3"
    elif defect == "duplicate_segment":
        data["estimation_segments"] *= 2
    elif defect == "source_file_unknown":
        observed["source_file"] = "absent.csv"
    elif defect == "source_hash":
        source["sha256"] = "0" * 64
    elif defect == "source_size":
        source["size_bytes"] += 1
    elif defect == "source_path_escape":
        source["path"] = "../annotations.csv"
    else:
        raise AssertionError(defect)
    with pytest.raises(ValueError, match=message):
        read_changed(document)


def test_json_integrity_and_duplicate_object_keys(document) -> None:
    path, _ = document
    path.write_bytes(FIXTURE.read_bytes())
    with pytest.raises(ValueError, match="reference SHA-256"):
        load_user_reference(path, expected=identity(), expected_sha256="0" * 64)
    path.write_text('{"schema":"first", "schema":"second"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_user_reference(
            path,
            expected=identity(),
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "1e999"])
def test_nonfinite_json_coordinate(document, literal: str) -> None:
    path, _ = document
    # 1e999 is valid JSON syntax but overflows Python's float, unlike NaN/Infinity.
    path.write_text(FIXTURE.read_text().replace("61.25", literal))
    with pytest.raises(ValueError, match="non-finite|finite number"):
        load_user_reference(
            path,
            expected=identity(),
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def test_native_pixel_boundary_values_are_not_missing(document) -> None:
    _, data = document
    data["positions"][2]["uv_px"] = [0, 0]
    data["positions"][5]["uv_px"] = [79, 39]
    truth = read_changed(document)
    selected = truth.select(camera_ids=("right", "left"), frame_indices=(0,))
    np.testing.assert_array_equal(selected.uv_px, [[[79, 39]], [[0, 0]]])
    np.testing.assert_array_equal(selected.valid_for_scoring, [[True], [True]])


@pytest.mark.parametrize(
    "cameras, frames",
    [
        (("absent",), (0,)),
        (("right",), (-1,)),
        (("right",), (3,)),
        (("right",), (True,)),
        (("right", "right"), (0,)),
        (("right",), (0, 0)),
        ((), (0,)),
        (("right",), ()),
    ],
)
def test_selection_refuses_ambiguous_or_nonexistent_axes(cameras, frames) -> None:
    truth = load_user_reference(
        FIXTURE,
        expected=identity(),
        expected_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError):
        truth.select(camera_ids=cameras, frame_indices=frames)
