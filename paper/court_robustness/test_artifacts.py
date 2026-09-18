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
        panels = external_panels(image, b, o)
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
        actual = external_panels(image, b, o)["baseline"]
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        assert not np.array_equal(np.asarray(actual), np.asarray(unrefined))


def make_build_fixture(root: Path) -> None:
    from build_paper import source_digests
    from common import write_json

    (root / "figures").mkdir()
    (root / "report.tex").write_text("source fixture")
    (root / "report.pdf").write_bytes(b"pdf byte fixture")
    (root / "figures/view.png").write_bytes(b"figure byte fixture")
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


@pytest.mark.parametrize("changed", ["report.pdf", "report.tex", "figures/view.png"])
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
