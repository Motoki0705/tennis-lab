"""Numerical checks for the paired comparison and its decision rule."""

import pytest
from report import classify, metrics, paired_video_interval


def test_summary_pools_visible_points_and_counts_failed_h_in_the_denominator():
    rows = [
        {
            "video_id": "a",
            "kp_errors_px512": [0.0, 12.0],
            "h_status": "ok",
            "h_at_1pct": False,
        },
        {
            "video_id": "b",
            "kp_errors_px512": [24.0],
            "h_status": "failed",
            "h_at_1pct": False,
        },
    ]
    summary = metrics(rows)
    assert summary["images"] == 2
    assert summary["visible_points"] == 3
    assert summary["kp_mean_px512"] == 12.0
    assert summary["h_formed_pct"] == 50.0
    assert summary["h_at_1pct_pct"] == 0.0
    assert metrics([])["h_at_1pct_pct"] is None


def test_paired_interval_draws_videos_and_uses_both_seed_pairs():
    def rows(a, b):
        return [
            {
                "video_id": name,
                "kp_errors_px512": [value],
                "h_status": "ok",
                "h_at_1pct": False,
            }
            for name, value in [("a", a), ("a", a), ("b", b)]
        ]

    intervals = paired_video_interval(
        {
            ("narrow", 42): rows(10, 10),
            ("broad", 42): rows(12, 14),
            ("narrow", 43): rows(10, 10),
            ("broad", 43): rows(6, 16),
        }
    )
    # More than 2.5% of draws repeat only a or only b. Their seed means are
    # (2 - 4) / 2 and (4 + 6) / 2; H never changes among the paired rows.
    assert intervals["kp_mean_px512"] == pytest.approx([-1, 5])
    assert intervals["h_at_1pct_pct"] == [0, 0]


@pytest.mark.parametrize(
    ("seed42", "seed43", "expected"),
    [
        ((-1, 1), (-2, 1), "supported"),
        ((1, 0), (-2, 0), "partially_supported"),
        ((1, -1), (0, 0), "unsupported"),
    ],
)
def test_judgment_uses_the_two_preregistered_test_populations(seed42, seed43, expected):
    rows = [
        {
            "split": "test",
            "stratum": stratum,
            "broad_minus_narrow_by_seed": {
                str(seed): {
                    "kp_mean_px512": diff[0],
                    "h_at_1pct_pct": diff[1],
                    "h_formed_pct": 10,
                }
                for seed, diff in [(42, seed42), (43, seed43)]
            },
        }
        for stratum in ("all", "challenging")
    ]
    assert classify(rows) == expected
