"""Report rendering keeps one number consistent across JSON, CSV, and TeX."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from src.tasks.court_detection.evaluation.reporting import (
    csv_rows,
    render_csv,
    render_tex,
    write_report,
)

_AGGREGATES = [
    {
        "model": "ours",
        "domain": "real_validation",
        "keypoints": {
            "sample_count": 2,
            "ground_truth_visible_keypoints": 28,
            "predicted_valid_keypoints_among_visible": 26,
            "completeness": 26 / 28,
            "pck": {"0.005": 0.5, "0.01": 0.9, "0.02": 0.95, "0.05": 1.0},
            "pair_error_px": {"count": 26, "mean": 3.0, "median": 2.5, "q90": 5.0},
            "pair_error_diagonal": {
                "count": 26,
                "mean": 0.002,
                "median": 0.002,
                "q90": 0.004,
            },
        },
        "alignment": {
            "sample_count": 2,
            "ground_truth_homography_success": 2,
            "predicted_homography_success": 1,
            "predicted_homography_success_rate": 0.5,
            "predicted_homography_failure_reasons": {"insufficient_valid_keypoints": 1},
            "predicted_homography_inliers_on_success": {
                "count": 2,
                "mean": 13.0,
                "median": 13.0,
                "q90": 14.0,
            },
            "line_reprojection_px": {
                "count": 1,
                "mean": 4.0,
                "median": 4.0,
                "q90": 4.0,
            },
            "line_reprojection_in_frame_px": {
                "count": 1,
                "mean": 3.0,
                "median": 3.0,
                "q90": 3.0,
            },
            "line_samples_in_frame": 10,
            "line_samples_in_frame_fraction": 0.5,
            "line_samples_total": 20,
            "line_reprojection_diagonal": {
                "count": 1,
                "mean": 0.003,
                "median": 0.003,
                "q90": 0.003,
            },
            "doubles_polygon_iou": {"count": 1, "mean": 0.9, "median": 0.9, "q90": 0.9},
            "undefined_reasons": {"no_valid_prediction_court_homography": 1},
        },
        "per_channel": [
            {
                "index": 0,
                "name": "far_doubles_left",
                "ground_truth_visible": 2,
                "predicted_valid_among_visible": 2,
                "pck": {"0.005": 0.5, "0.01": 1.0},
                "pair_error_px": {"count": 2, "mean": 1.0, "median": 1.0, "q90": 1.5},
            }
        ],
        "strata": {"scene": {"tennis_court_detector": {}}},
    },
    {
        "model": "tcd",
        "domain": "real_validation",
        "keypoints": {
            "sample_count": 2,
            "ground_truth_visible_keypoints": 28,
            "predicted_valid_keypoints_among_visible": 28,
            "completeness": 1.0,
            "pck": {"0.005": 0.6, "0.01": 0.92, "0.02": 0.97, "0.05": 1.0},
            "pair_error_px": {"count": 28, "mean": 2.0, "median": 1.5, "q90": 4.0},
            "pair_error_diagonal": {
                "count": 28,
                "mean": 0.001,
                "median": 0.001,
                "q90": 0.003,
            },
        },
        "alignment": {
            "sample_count": 2,
            "ground_truth_homography_success": 2,
            "predicted_homography_success": 2,
            "predicted_homography_success_rate": 1.0,
            "predicted_homography_failure_reasons": {},
            "predicted_homography_inliers_on_success": {
                "count": 2,
                "mean": 14.0,
                "median": 14.0,
                "q90": 14.0,
            },
            "line_reprojection_px": {
                "count": 2,
                "mean": 1.5,
                "median": 1.5,
                "q90": 1.6,
            },
            "line_reprojection_in_frame_px": {
                "count": 2,
                "mean": 1.2,
                "median": 1.2,
                "q90": 1.3,
            },
            "line_samples_in_frame": 12,
            "line_samples_in_frame_fraction": 0.6,
            "line_samples_total": 20,
            "line_reprojection_diagonal": {
                "count": 2,
                "mean": 0.001,
                "median": 0.001,
                "q90": 0.001,
            },
            "doubles_polygon_iou": {
                "count": 2,
                "mean": 0.95,
                "median": 0.95,
                "q90": 0.96,
            },
            "undefined_reasons": {},
        },
        "per_channel": [
            {
                "index": 0,
                "name": "far_doubles_left",
                "ground_truth_visible": 2,
                "predicted_valid_among_visible": 2,
                "pck": {"0.005": 0.6, "0.01": 1.0},
                "pair_error_px": {"count": 2, "mean": 0.5, "median": 0.5, "q90": 0.7},
            }
        ],
        "strata": {"scene": {"tennis_court_detector": {}}},
    },
]
_DISPLAY = {"real_validation": "real_validation"}


def _as_mapping(value: object) -> Mapping[str, Any]:
    """Narrow one aggregate into an indexable mapping for assertions."""
    assert isinstance(value, Mapping)
    return cast("Mapping[str, Any]", value)


def test_write_report_publishes_all_three_formats(tmp_path: Path) -> None:
    paths = write_report(
        tmp_path,
        aggregates=_AGGREGATES,
        display_names=_DISPLAY,
        manifest_fingerprint="d" * 64,
        command="benchmark_alignment --datasets real_validation",
    )

    assert set(paths) == {"json", "csv", "tex"}
    for path in paths.values():
        assert path.is_file() and path.stat().st_size > 0
    document = _as_mapping(json.loads(paths["json"].read_text(encoding="utf-8")))
    assert document["manifest_fingerprint"] == "d" * 64
    notes = _as_mapping(document["notes"])
    assert "held-out" in notes["real_validation"]
    assert [
        _as_mapping(item)["model"]
        for item in cast("list[object]", document["aggregates"])
    ] == ["ours", "tcd"]


def test_csv_exposes_one_row_per_metric_leaf_with_counts(tmp_path: Path) -> None:
    rows = csv_rows(_AGGREGATES, display_names=_DISPLAY)
    rendered = render_csv(rows)

    header = rendered.splitlines()[0]
    assert header == (
        "domain,domain_display_name,model,metric_group,metric,subkey,value,count"
    )
    assert (
        "real_validation,real_validation,ours,keypoints,completeness,,0.928571,"
        in rendered
    )
    # Distribution leaves carry the count of the block they belong to, so a
    # reader of the CSV alone still sees the denominator.
    assert "pair_error_px,median,2.5,26" in rendered
    assert "pair_error_px,mean,3,26" in rendered
    assert "pair_error_px,count,26,26" in rendered
    assert "line_reprojection_px,q90,4,1" in rendered
    assert "doubles_polygon_iou,mean,0.9,1" in rendered
    assert "per_channel,far_doubles_left,pck,0.005=0.5; 0.01=1,2" in rendered


def test_tex_tables_are_include_ready_and_mark_undefined_values() -> None:
    tex = render_tex(
        _AGGREGATES,
        display_names=_DISPLAY,
        manifest_fingerprint="d" * 64,
        command="benchmark_alignment",
    )

    assert tex.count(r"\begin{tabular}") == 2
    assert tex.count(r"\end{tabular}") == 2
    assert r"\input" not in tex
    assert "not an official test set" in tex
    assert "0.9000" in tex  # doubles IoU mean for ours


def test_undefined_metrics_render_as_null_and_dashes() -> None:
    aggregates = json.loads(json.dumps(_AGGREGATES))
    aggregates[0]["alignment"]["line_reprojection_px"] = {
        "count": 0,
        "mean": None,
        "median": None,
        "q90": None,
    }

    rendered = render_csv(csv_rows(aggregates, display_names=_DISPLAY))
    tex = render_tex(
        aggregates,
        display_names=_DISPLAY,
        manifest_fingerprint="d" * 64,
        command="benchmark_alignment",
    )

    assert "line_reprojection_px,mean,," in rendered
    line_rows = [
        line
        for line in tex.splitlines()
        if line.startswith("real\\_validation & ours &")
    ]
    assert "--" in line_rows[-1]
