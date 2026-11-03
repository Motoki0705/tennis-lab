"""Shared owner identities and source-pixel location metrics without presence scores."""

import hashlib

import pytest

from src.tasks.ball_detection.evaluation.temporal_metrics import (
    location_metrics,
    prediction_identities,
)


def test_owner_identities_keep_compact_manifest_encoding_and_label_membership() -> None:
    records = [
        dict(clip_id="clip", frame_index=4, native_fps="30000/1001", kind="masked"),
        dict(clip_id="clip", frame_index=5, native_fps="30000/1001", kind="observed"),
        dict(clip_id="clip", frame_index=6, native_fps="30000/1001", kind="reviewed_empty"),
        dict(clip_id="clip", frame_index=7, native_fps="30000/1001", kind="out_of_frame"),
    ]
    target = (
        b'["clip",4,1,"30000/1001"]\n'
        b'["clip",5,1,"30000/1001"]\n'
        b'["clip",6,1,"30000/1001"]\n'
        b'["clip",7,1,"30000/1001"]\n'
    )
    teacher = (
        b'["clip",4,1,"30000/1001",0]\n'
        b'["clip",5,1,"30000/1001",1]\n'
        b'["clip",6,1,"30000/1001",2]\n'
        b'["clip",7,1,"30000/1001",3]\n'
    )
    scored = (
        b'["clip",5,1,"30000/1001"]\n'
        b'["clip",6,1,"30000/1001"]\n'
        b'["clip",7,1,"30000/1001"]\n'
    )
    assert prediction_identities(records) == {
        "target_identity_sha256": hashlib.sha256(target).hexdigest(),
        "teacher_identity_sha256": hashlib.sha256(teacher).hexdigest(),
        "scored_frame_sha256": hashlib.sha256(scored).hexdigest(),
    }


def test_location_metrics_need_no_presence_scores_and_count_only_observed_errors() -> None:
    records = [dict(kind="observed", error_px=error) for error in (0., 20., 40., 100.)]
    records += [
        dict(kind="reviewed_empty", error_px=None),
        dict(kind="out_of_frame", error_px=None),
        dict(kind="masked", error_px=None),
    ]
    report = location_metrics(records, distance_px=20.)
    assert report["support"] == dict(
        frames=7, scored_frames=6, observed=4, reviewed_empty=1, out_of_frame=1, masked=1
    )
    assert report["mean_error_px"] == 40. and report["median_error_px"] == 30.
    assert report["p95_error_px"] == pytest.approx(91.)
    assert report["within_distance_rate"] == .5
    assert report["distance_threshold_source_px"] == 20.
    empty = location_metrics(records[4:], distance_px=20.)
    assert empty["support"]["frames"] == 3 and empty["support"]["scored_frames"] == 2
    for name in ("mean_error_px", "median_error_px", "p95_error_px", "within_distance_rate"):
        assert empty[name] is None
