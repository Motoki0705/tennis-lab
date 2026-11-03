"""The integrated display binds counts to the foundation's scoring summaries."""

import importlib.util
import sys
from pathlib import Path


def test_fold_table_uses_scoring_coverage_and_keeps_empty_errors_unmeasured(
    monkeypatch,
):
    directory = (
        Path(__file__).resolve().parents[5]
        / "knowledge/runs/run-ball3d-integrated-comparison-20261104"
    )
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.delitem(sys.modules, "common", raising=False)
    spec = importlib.util.spec_from_file_location(
        "integrated_display_v2_test", directory / "report_v2.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def summary(count, eligible, median, p90):
        return {
            "count": count,
            "label_count": eligible,
            "scored_label_coverage": count / eligible,
            "median": median,
            "p90": p90,
            "above_threshold_rate": 0.0 if count else None,
        }

    metrics = {
        "without_cam0": summary(1, 3, 4.0, 4.0),
        "without_cam1": summary(1, 2, 12.0, 12.0),
        "without_cam2": summary(0, 2, None, None),
        "all_views": summary(4, 7, 6.0, 14.0),
    }
    folds = {
        key: {
            segment: {
                "reprojection_px": {
                    k: v
                    for k, v in m.items()
                    if k not in {"label_count", "scored_label_coverage"}
                },
                "reprojection_by_reference_kind": {"all": m},
                "valid_3d_rate": 0.4,
            }
            for segment in module.ORIGINAL.SEGMENTS
        }
        for key, m in metrics.items()
    }
    scores = {
        "methods": {
            "method": {
                "lovo_reprojection_px": {
                    s: summary(2, 7, 8.0, 11.2) for s in module.ORIGINAL.SEGMENTS
                },
                "folds": folds,
            }
        }
    }
    rows = module.metric_rows(
        scores,
        {"methods": [{"name": "method", "category": "method"}], "diagnostics": []},
    )
    chosen = {r["held_out_camera"]: r for r in rows if r["segment"] == "evaluation"}
    assert (
        chosen["pooled"]["count"],
        chosen["pooled"]["eligible"],
        chosen["pooled"]["median_px"],
    ) == (2, 7, 8.0)
    assert chosen["cam0"]["coverage"] == 1 / 3
    assert chosen["cam1"]["median_px"] == 12.0
    assert (
        chosen["cam2"]["count"] == 0
        and chosen["cam2"]["median_px"] is None
        and chosen["cam2"]["coverage"] == 0.0
    )
    assert (
        chosen["all_views"]["coverage"] == 4 / 7
        and chosen["all_views"]["all_view_valid_rate"] == 0.4
    )
    assert (
        "label_count"
        not in scores["methods"]["method"]["folds"]["without_cam0"]["evaluation"][
            "reprojection_px"
        ]
    )
