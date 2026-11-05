from src.tennis_scene.visualization.ball3d.tables import comparison_tables


def test_main_table_reports_lovo_counts_coverage_separately_from_all_view_validity():
    # Review example: these two rates have different numerators and denominators.
    lovo = dict(
        median=9.34,
        p90=27.08,
        above_threshold_rate=135 / 825,
        count=825,
        label_count=1462,
        scored_label_coverage=825 / 1462,
    )
    common = dict(lovo, count=740, scored_label_coverage=740 / 1462)
    full = dict(
        valid_3d_rate=420 / 505,
        reprojection_px={"median": 4.625},
        gross_reprojection_frame_rate=0.0976,
        speed_mps={"above_threshold_rate": 0.0},
        acceleration_mps2={"above_threshold_rate": 0.1462},
    )
    scores = {
        "config": {"gross_reprojection_px": 20},
        "common_lovo_cohort": {"methods": ["baseline", "other"]},
        "methods": {
            "baseline": {
                "lovo_reprojection_px": {
                    s: lovo for s in ("tuning", "evaluation", "all")
                },
                "common_lovo_reprojection_px": {
                    s: common for s in ("tuning", "evaluation", "all")
                },
                "folds": {
                    "all_views": {s: full for s in ("tuning", "evaluation", "all")}
                },
            }
        },
    }
    primary, paired = comparison_tables(scores)
    row = dict(zip(primary.headers, primary.rows[1], strict=True))
    assert row["LOVO scored"] == "825" and row["LOVO eligible"] == "1462"
    assert row["LOVO coverage"] == "56.43%"
    assert row["All-view 3D valid rate"] == "83.17%"
    paired_row = dict(zip(paired.headers, paired.rows[1], strict=True))
    assert paired_row["Common LOVO scored"] == "740"
    assert paired_row["Common coverage"] == "50.62%"
