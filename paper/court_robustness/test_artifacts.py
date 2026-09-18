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
        np.testing.assert_array_equal(
            np.asarray(panels["ours_line_overlay"])[~active], np.asarray(image)[~active]
        )
