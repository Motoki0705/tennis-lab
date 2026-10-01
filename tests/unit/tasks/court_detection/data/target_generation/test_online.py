"""Dense targets share one source fit and follow the sampled image transform."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from PIL import Image

from src.tasks.court_detection.data.contracts import (
    CourtInstance2D,
    CourtKeypointChannels,
    CourtRawSample,
    CourtSampleMetadata,
)
from src.tasks.court_detection.data.target_generation.online import (
    generate_online_targets,
)
from src.tasks.court_detection.target_schemas import (
    LINE_TARGET_SCHEMA,
    SEGMENTATION_TARGET_SCHEMA,
    SEMANTIC_LINE_TARGET_SCHEMA,
)
from src.utils.schema.court import (
    GROUND_COURT_KP_NAMES,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)


def _raw() -> CourtRawSample:
    points = court_keypoints_3d(STANDARD_COURT_CONFIG)[:14, :2]
    pixels = torch.stack(
        ((points[:, 0] / 12 + 0.5) * 255, (0.5 - points[:, 1] / 26) * 255), dim=-1
    )
    visible = torch.ones(14, dtype=torch.bool)
    instance = CourtInstance2D("court", torch.arange(14), pixels, visible, visible)
    channels = CourtKeypointChannels(
        GROUND_COURT_KP_NAMES,
        pixels[:, None],
        visible[:, None],
        torch.arange(14)[:, None],
        (1, 0, 3, 2, 6, 7, 4, 5, 9, 8, 11, 10, 12, 13),
    )
    return CourtRawSample(
        "sample",
        Image.new("RGB", (256, 256)),
        channels,
        (instance,),
        CourtSampleMetadata("tennis_court_detector", "test", "sample", None, {}),
    )


def test_all_dense_targets_without_disk_masks_share_one_homography(monkeypatch) -> None:
    from src.tasks.court_detection.data.target_generation import rasterization

    original = rasterization.compute_template_to_image_homography
    calls = []

    def capture(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(rasterization, "compute_template_to_image_homography", capture)
    targets = generate_online_targets(
        _raw(),
        {
            "seg": SEGMENTATION_TARGET_SCHEMA,
            "line": LINE_TARGET_SCHEMA,
            "semantic_line": SEMANTIC_LINE_TARGET_SCHEMA,
        },
    )
    assert len(calls) == 1
    assert set(torch.unique(targets["seg"]).tolist()) == set(range(7))
    assert set(torch.unique(targets["semantic_line"]).tolist()) == set(range(12))
    assert torch.equal(targets["line"][0] > 0, targets["semantic_line"] > 0)


def test_crop_transform_and_padding_do_not_extrapolate_supervision() -> None:
    raw = _raw()
    transform = torch.tensor(
        [[1.0, 0.0, 40.0], [0.0, 1.0, 30.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    schemas = {"line": LINE_TARGET_SCHEMA}
    original = generate_online_targets(raw, schemas)["line"]
    shifted = generate_online_targets(
        raw,
        schemas,
        source_to_output=transform,
        output_size_hw=(320, 320),
        content_size_hw=(286, 296),
    )["line"]
    assert torch.equal(shifted[:, 30:286, 40:296], original)
    assert not shifted[:, :30].any()
    assert not shifted[:, :, :40].any()
    assert not shifted[:, 286:].any()


def test_invisible_heatmap_points_still_define_dense_geometry() -> None:
    raw = _raw()
    assert raw.keypoint_channels is not None
    occluded = replace(
        raw,
        keypoint_channels=replace(
            raw.keypoint_channels, point_visible=torch.zeros((14, 1), dtype=torch.bool)
        ),
    )
    schemas = {"seg": SEGMENTATION_TARGET_SCHEMA}
    assert torch.equal(
        generate_online_targets(raw, schemas)["seg"],
        generate_online_targets(occluded, schemas)["seg"],
    )


def test_single_court_dense_targets_reject_multiple_instances() -> None:
    raw = _raw()
    multiple = replace(raw, court_instances=(raw.court_instances[0], replace(raw.court_instances[0], court_instance_id="second")))
    with pytest.raises(ValueError, match="exactly one selected court"):
        generate_online_targets(multiple, {"line": LINE_TARGET_SCHEMA})
