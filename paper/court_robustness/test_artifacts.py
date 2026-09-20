"""Meaningful checks of the projection and immutable source/figure contracts."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest
from common import ROOT, sha256, sources
from homography_evidence import read_results
from make_comparisons import external_panels, fit_panel
from make_scene_figures import (
    SELECTION,
    project_segment,
    render_overlay,
    validate_projection,
)
from PIL import Image, ImageOps


@pytest.mark.parametrize("sid", list(SELECTION))
def test_camera_and_alignment_match_saved_labels(sid: str) -> None:
    bundle = json.loads((ROOT / f"evidence/scene_sources/{sid}.json").read_text())
    groups = set()
    for view in bundle["views"]:
        sample = view["sample"]
        groups.add(sample["trajectory_group_id"])
        assert validate_projection(bundle, sample) < 1e-6
        rgb = Image.open(ROOT / view["bundled_rgb"]).convert("RGB")
        assert sha256(ROOT / view["bundled_rgb"]) == view["rgb_sha256"]
        actual = render_overlay(bundle, sample, rgb)
        saved = Image.open(ROOT / view["figure"]).convert("RGB")
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(saved))
        assert np.count_nonzero(np.asarray(actual) != np.asarray(rgb)) > 100
    assert len(groups) == 3


def test_foreign_camera_is_rejected() -> None:
    bundle = json.loads((ROOT / "evidence/scene_sources/B00.json").read_text())
    sample = copy.deepcopy(bundle["views"][0]["sample"])
    sample["camera"]["camera_to_scene"][3] += 1.0
    with pytest.raises(ValueError, match="projection mismatch"):
        validate_projection(bundle, sample)


def test_stale_alignment_is_rejected() -> None:
    bundle = json.loads((ROOT / "evidence/scene_sources/B01.json").read_text())
    bundle["alignment"]["layout"]["courts"][0]["scene_from_court"][3] += 0.5
    with pytest.raises(ValueError, match="metric coordinates"):
        validate_projection(bundle, bundle["views"][0]["sample"])


def test_near_plane_crossing_is_clipped_without_mirroring() -> None:
    line = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, 1.0]])
    result = project_segment(line, np.eye(3), near=0.1)
    np.testing.assert_allclose(result, [[10.0, 0.0], [1.0, 0.0]])
    assert (
        project_segment(np.array([[1.0, 0.0, -2.0], [1.0, 0.0, -1.0]]), np.eye(3))
        is None
    )
    np.testing.assert_array_equal(line, [[1.0, 0.0, -1.0], [1.0, 0.0, 1.0]])


@pytest.mark.parametrize("record", sources(), ids=lambda r: r["id"])
def test_external_panels_reproduce_saved_predictions(record: dict) -> None:
    image = ImageOps.exif_transpose(Image.open(ROOT / record["paper_path"])).convert(
        "RGB"
    )
    ident = record["id"]
    with (
        np.load(ROOT / f"evidence/predictions/{ident}_baseline.npz") as b,
        np.load(ROOT / f"evidence/predictions/{ident}_ours.npz") as o,
    ):
        original_line = o["line_probability"].copy()
        panels = external_panels(image, b, o, read_results()["images"][record["id"]])
        for name, panel in panels.items():
            saved = Image.open(ROOT / f"figures/{ident}_{name}.png").convert("RGB")
            np.testing.assert_array_equal(
                np.asarray(fit_panel(panel, (720, 500))), np.asarray(saved)
            )
        np.testing.assert_array_equal(o["line_probability"], original_line)
        assert panels["input"].size == image.size
        # A raw LINE overlay must not alter any subthreshold background pixel.
        import cv2

        active = cv2.resize(original_line, image.size) >= 0.5
        expected_mask = np.repeat((active.astype(np.uint8) * 255)[..., None], 3, axis=2)
        np.testing.assert_array_equal(
            np.asarray(panels["ours_line_mask"]), expected_mask
        )
        np.testing.assert_array_equal(
            np.asarray(panels["ours_line_overlay"])[~active], np.asarray(image)[~active]
        )


def test_tcd_official_panel_displays_refined_keypoints() -> None:
    from make_comparisons import overlay

    record = sources()[0]
    image = Image.open(ROOT / record["paper_path"]).convert("RGB")
    with (
        np.load(ROOT / "evidence/predictions/local01_baseline.npz") as b,
        np.load(ROOT / "evidence/predictions/local01_ours.npz") as o,
    ):
        assert not bool(b["homography_found"])
        no_lines = np.full((14, 2), np.nan)
        expected = overlay(image, no_lines, (255, 98, 48), b["refined_kp"])
        unrefined = overlay(image, no_lines, (255, 98, 48), b["raw_kp"])
        actual = external_panels(image, b, o, read_results()["images"][record["id"]])[
            "baseline"
        ]
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        assert not np.array_equal(np.asarray(actual), np.asarray(unrefined))


def test_hybrid_homography_reproduces_saved_kp_line_and_generated_table() -> None:
    from homography_evidence import table_text

    results = read_results()
    assert table_text(results) == (ROOT / "tables/homography.tex").read_text()
    assert results["images"]["local02"]["top4_template_collinear"]
    for record in sources():
        item = results["images"][record["id"]]
        assert item["status"] == "ok"
        assert 4 <= item["inlier_count"] <= 8
        residuals = np.array(item["residuals_px"])
        selected = np.asarray(item["inliers"])
        assert np.all(residuals[selected] <= item["threshold_px"])
        assert np.all(
            np.asarray(item["kp_line_distance_px"])[selected] <= item["threshold_px"]
        )
        assert (
            np.flatnonzero(selected).tolist()
            == item["stages"]["hybrid"]["selection_history"][-1]
        )
        assert all(
            4 <= len(indices) <= 8
            for indices in item["stages"]["hybrid"]["selection_history"]
        )
        assert item["fit_inliers"] == item["inliers"]
        with np.load(ROOT / f"evidence/predictions/{record['id']}_ours.npz") as source:
            scores = source["kp_scores"][item["ranked_indices"]]
            assert np.all(np.diff(scores) <= 0)


@pytest.mark.parametrize(
    "change",
    ["parameters", "matrix", "rank", "source", "line_support", "selected_history"],
)
def test_hybrid_evidence_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    import homography_evidence as module

    original = module.BUNDLE
    result = json.loads((original / "results.json").read_text())
    item = result["images"]["local02"]
    if change == "parameters":
        result["parameters"]["hybrid"]["max_kp"] = 14
    elif change == "matrix":
        item["matrix"][0][2] += 1
    elif change == "rank":
        item["ranked_indices"] = item["ranked_indices"][::-1]
    elif change == "line_support":
        item["stages"]["hybrid"]["line"]["reverse_support"] = 0
    elif change == "selected_history":
        item["stages"]["hybrid"]["selection_history"][-1] = list(range(14))
    else:
        item["source_sha256"] = "0" * 64
    (tmp_path / "results.json").write_text(json.dumps(result))
    (tmp_path / "template.json").write_bytes((original / "template.json").read_bytes())
    monkeypatch.setattr(module, "BUNDLE", tmp_path)
    with pytest.raises(ValueError, match="homography"):
        module.read_results()


def make_build_fixture(root: Path) -> None:
    from build_paper import source_digests
    from common import write_json

    (root / "figures").mkdir()
    (root / "tables").mkdir()
    (root / "report.tex").write_text("source fixture")
    (root / "report.pdf").write_bytes(b"pdf byte fixture")
    (root / "figures/view.png").write_bytes(b"figure byte fixture")
    (root / "tables/measurements.tex").write_text("table fixture")
    write_json(
        root / "evidence/build.json",
        {
            "schema": "court_paper_build_v1",
            "source_sha256": source_digests(root),
            "pdf_sha256": sha256(root / "report.pdf"),
            "layout_glyph_reference_checks": "passed",
        },
    )


def test_offline_build_binding_needs_no_temporary_log(tmp_path: Path) -> None:
    from verify_artifacts import validate_build_receipt

    make_build_fixture(tmp_path)
    assert not (tmp_path / "report.log").exists()
    validate_build_receipt(tmp_path)


@pytest.mark.parametrize(
    "changed",
    ["report.pdf", "report.tex", "figures/view.png", "tables/measurements.tex"],
)
def test_build_binding_rejects_modified_artifact(tmp_path: Path, changed: str) -> None:
    from verify_artifacts import validate_build_receipt

    make_build_fixture(tmp_path)
    (tmp_path / changed).write_bytes(b"altered bytes")
    with pytest.raises(ValueError, match="differs|differ"):
        validate_build_receipt(tmp_path)


def test_local_check_rejects_different_checkpoint(tmp_path: Path) -> None:
    from verify_artifacts import validate_local_weights

    path = tmp_path / "weights.ckpt"
    path.write_bytes(b"different checkpoint bytes")
    with pytest.raises(ValueError, match="Checkpoint SHA-256 differs"):
        validate_local_weights({"checkpoint": str(path), "ours_sha256": "0" * 64})


def test_method_ransac_projection_and_aggregate_match_saved_run() -> None:
    from alignment_evidence import read_bundle, validate_geometry

    manifest, arrays = read_bundle()
    result = validate_geometry(manifest, arrays)
    assert result["point_count"] == 217407
    assert result["fit_views"] == 32 and result["holdout_views"] == 16
    assert result["max_projection_difference_metres"] < 1e-9
    assert result["aggregate_matches_exactly"]


def test_method_rejects_camera_intrinsics_not_used_for_inference() -> None:
    from alignment_evidence import SELECTED, read_bundle, validate_geometry

    manifest, arrays = read_bundle()
    camera = next(c for c in manifest["cameras"] if c["camera_id"] == SELECTED[0])
    camera["intrinsics"][2] += 20
    with pytest.raises(ValueError, match="Ray-plane projection differs"):
        validate_geometry(manifest, arrays)


def test_method_aggregate_rejects_holdout_views() -> None:
    from alignment_evidence import aggregate, read_bundle

    manifest, arrays = read_bundle()
    holdout = int(np.flatnonzero(~arrays["included_in_aggregate"])[0])
    with pytest.raises(ValueError, match="Holdout view"):
        aggregate(manifest, arrays, [holdout])


@pytest.mark.parametrize("name", ["ransac", "projection"])
def test_method_figures_reproduce_actual_bundled_evidence(
    tmp_path: Path, name: str
) -> None:
    import matplotlib.pyplot as plt
    from alignment_evidence import read_bundle
    from make_alignment_figures import projection_figure, ransac_figure

    manifest, arrays = read_bundle()
    builder = ransac_figure if name == "ransac" else projection_figure
    figure, _ = builder(manifest, arrays)
    output = tmp_path / f"{name}.png"
    figure.savefig(output, dpi=210, facecolor="white")
    plt.close(figure)
    np.testing.assert_array_equal(
        np.asarray(Image.open(output)),
        np.asarray(Image.open(ROOT / f"figures/method_{name}.png")),
    )


def test_sfm_tracks_use_frame_names_and_reject_truncation(tmp_path: Path) -> None:
    import struct

    from drift_evidence import read_tracks

    images = bytearray(struct.pack("<Q", 2))
    for ident, frame in ((100, 12), (7, 5)):
        images.extend(struct.pack("<idddddddi", ident, 1, 0, 0, 0, 0, 0, 0, 1))
        images.extend(f"frame_{frame:06d}.png\0".encode())
        images.extend(struct.pack("<Qddq", 1, 10, 20, 9000))
    (tmp_path / "images.bin").write_bytes(images)
    point = struct.pack("<Q", 1) + struct.pack("<QdddBBBd", 9000, 1, 2, 3, 7, 8, 9, 0.3)
    point += struct.pack("<Qiiii", 2, 100, 0, 7, 0)
    (tmp_path / "points3D.bin").write_bytes(point)
    xyz, rgb, tracks, frames = read_tracks(tmp_path)
    assert frames == [5, 12]
    assert tracks["first_frame"].tolist() == [5]
    assert tracks["last_frame"].tolist() == [12]
    assert tracks["point_id"].tolist() == [9000]
    np.testing.assert_array_equal(xyz, [[1, 2, 3]])
    np.testing.assert_array_equal(rgb, [[7, 8, 9]])
    (tmp_path / "points3D.bin").write_bytes(point[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        read_tracks(tmp_path)


def test_sfm_export_mapping_requires_unique_points_and_matching_rgb() -> None:
    from drift_evidence import match_export

    xyz = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    rgb = np.array([[0, 128, 255], [255, 64, 0]], dtype=np.uint8)
    exported = np.c_[xyz, rgb / 255].astype(np.float32)[::-1]
    order, _ = match_export(xyz, rgb, exported, np.eye(4))
    assert order.tolist() == [1, 0]
    with pytest.raises(ValueError, match="bijectively"):
        match_export(np.repeat(xyz[:1], 2, axis=0), rgb, exported, np.eye(4))
    with pytest.raises(ValueError, match="bijectively"):
        match_export(xyz, rgb[::-1], exported, np.eye(4))


def test_sfm_height_comparison_pairs_cells_and_excludes_cross_period_tracks() -> None:
    from drift_evidence import paired_heights

    # Shared cell at U=0.1; very different surfaces present in just one period.
    uv = np.array(
        [
            [0.1, 0],
            [0.1, 0],
            [0.1, 0],
            [0.1, 0],
            [1.1, 0],
            [1.1, 0],
            [2.1, 0],
            [2.1, 0],
            [0.1, 0],
        ]
    )
    height = np.array([0, 0, 0.02, 0.02, 10, 10, -10, -10, 999])
    first = np.array([0, 0, 10, 10, 0, 0, 10, 10, 9])
    last = np.array([1, 1, 11, 11, 1, 1, 11, 11, 10])
    result = paired_heights(
        uv, height, first, last, np.ones(9, bool), np.array([0, 10, 20]), 0.5, 2
    )
    assert result["shared_cell_count"] == 1
    assert result["qualified_point_counts"] == [4, 4]
    np.testing.assert_allclose(result["median_delta_m"], [0, 0.02])


def test_sfm_no_common_ground_is_explicitly_unmeasured_not_zero() -> None:
    from drift_evidence import paired_heights

    result = paired_heights(
        np.array([[0, 0], [1, 0]]),
        np.array([0, 0.02]),
        np.array([0, 10]),
        np.array([1, 11]),
        np.ones(2, bool),
        np.array([0, 10, 20]),
        0.5,
        1,
    )
    assert result["status"] == "no_shared_cells"
    assert result["shared_cell_count"] == 0
    assert result["median_delta_m"] is None
    assert result["iqr_delta_m"] is None


def test_sfm_periods_preserve_missing_frame_ids_and_last_registered_frame() -> None:
    from drift_evidence import period_edges

    frames = [0, 1, 5, 9]
    assert period_edges(frames, 2) == [0, 5, 10]
    assert period_edges(frames, 4) == [0, 1, 5, 9, 10]
    with pytest.raises(ValueError, match="unique chronological"):
        period_edges([0, 5, 1, 9], 2)


def test_sfm_ground_reference_rejects_non_coplanar_saved_courts() -> None:
    from drift_evidence import ground_reference

    alignment = json.loads((ROOT / "evidence/scene_sources/B01.json").read_text())
    ground_reference(alignment)
    alignment["alignment"]["layout"]["courts"][1]["scene_from_court"][11] += 0.1
    with pytest.raises(ValueError, match="common ground plane"):
        ground_reference(alignment)


def test_sfm_all_four_scenes_have_supported_primary_measurements() -> None:
    from drift_evidence import SCENE_IDS, validate

    result = validate()
    assert tuple(result) == SCENE_IDS
    assert sum(r["point_count"] for r in result.values()) == 428265
    for item in result.values():
        for protocol in ("primary", "sensitivity"):
            assert item[protocol]["status"] == "measured"
            assert item[protocol]["shared_cell_count"] > 0
            assert len(item[protocol]["median_delta_m"]) == 2
    for sid in ("B01", "B02", "B03"):
        assert result[sid]["four_period_audit"]["median_delta_m"] is None


def test_sfm_drift_figure_reproduces_measured_bundle(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt
    from drift_evidence import validate
    from make_drift_figure import drift_figure, drift_table

    validate()
    assert drift_table() == (ROOT / "tables/sfm_drift.tex").read_text()
    figure = drift_figure()
    output = tmp_path / "drift.png"
    figure.savefig(output, dpi=210, facecolor="white")
    plt.close(figure)
    np.testing.assert_array_equal(
        np.asarray(Image.open(output)),
        np.asarray(Image.open(ROOT / "figures/sfm_temporal_drift.png")),
    )
