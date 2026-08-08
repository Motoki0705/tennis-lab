"""Independent negative coverage for the Court final-render inventory gate."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from src.synthetic_data_generation.dataset.court.assembler import (
    _validate_render_inventory,
    _validate_renderer_visibility_payload,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    DatasetSplit,
    PlannedCourtSample,
)
from src.synthetic_data_generation.dataset.court.shards import CourtRenderedSample
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera


def _sample() -> PlannedCourtSample:
    camera = SceneCamera(
        camera_id="sample-000000",
        source_frame_index=0,
        width=4,
        height=3,
        intrinsics=(4.0, 0.0, 1.5, 0.0, 4.0, 1.0, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.identity(),
        image_path="generated/sample-000000.png",
    )
    return PlannedCourtSample(
        sample_index=0,
        sample_id=camera.camera_id,
        trajectory_group_id="group-a",
        trajectory_id="trajectory-a",
        view_id="view-a",
        trajectory_frame_index=0,
        split=DatasetSplit.TRAIN,
        shard_id="shard-000",
        camera_center_scene_m=(0.0, 0.0, 0.0),
        camera=camera,
    )


def _rendered(root: Path, sample: PlannedCourtSample) -> CourtRenderedSample:
    sample_root = root / sample.sample_id
    sample_root.mkdir(parents=True)
    np.save(sample_root / "rgb.npy", np.zeros((3, 4, 3), dtype=np.float32))
    np.save(sample_root / "alpha.npy", np.ones((3, 4, 1), dtype=np.float32))
    np.save(sample_root / "depth.npy", np.ones((3, 4, 1), dtype=np.float32))
    Image.new("RGB", (4, 3)).save(sample_root / "rgb.png")
    Image.new("L", (4, 3)).save(sample_root / "alpha.png")
    return CourtRenderedSample(
        sample=sample,
        rgb_path=sample_root / "rgb.npy",
        rgb_preview_path=sample_root / "rgb.png",
        alpha_path=sample_root / "alpha.npy",
        alpha_preview_path=sample_root / "alpha.png",
        depth_path=sample_root / "depth.npy",
    )


def test_render_inventory_rejects_missing_duplicate_and_overlapping_results(
    tmp_path: Path,
) -> None:
    sample = _sample()
    plan = SimpleNamespace(samples=(sample,))
    rendered = _rendered(tmp_path, sample)

    with pytest.raises(ValueError, match="partition mismatch.*missing"):
        _validate_render_inventory(
            plan,
            (),
            pre_render_rejected_sample_ids=(),
        )
    with pytest.raises(ValueError, match="Duplicate renderer sample ID"):
        _validate_render_inventory(
            plan,
            (rendered, rendered),
            pre_render_rejected_sample_ids=(),
        )
    with pytest.raises(ValueError, match="partition mismatch"):
        _validate_render_inventory(
            plan,
            (rendered,),
            pre_render_rejected_sample_ids=(sample.sample_id,),
        )
    with pytest.raises(ValueError, match="rejection inventory contains duplicates"):
        _validate_render_inventory(
            plan,
            (),
            pre_render_rejected_sample_ids=(sample.sample_id, sample.sample_id),
        )


def test_render_inventory_rejects_renderer_metadata_drift(tmp_path: Path) -> None:
    expected = _sample()
    plan = SimpleNamespace(samples=(expected,))
    changed = replace(expected, split=DatasetSplit.VALIDATION)
    rendered = _rendered(tmp_path, changed)

    with pytest.raises(ValueError, match="Renderer sample metadata changed"):
        _validate_render_inventory(
            plan,
            (rendered,),
            pre_render_rejected_sample_ids=(),
        )


def test_render_inventory_accepts_exact_renderer_or_rejection_partition(
    tmp_path: Path,
) -> None:
    rendered_sample = _sample()
    rejected_sample = replace(
        rendered_sample,
        sample_index=1,
        sample_id="sample-000001",
        camera=replace(
            rendered_sample.camera,
            camera_id="sample-000001",
            source_frame_index=1,
            image_path="generated/sample-000001.png",
        ),
    )
    plan = SimpleNamespace(samples=(rendered_sample, rejected_sample))
    rendered = _rendered(tmp_path, rendered_sample)

    _validate_render_inventory(
        plan,
        (rendered,),
        pre_render_rejected_sample_ids=(rejected_sample.sample_id,),
    )


@pytest.mark.parametrize("mutated_field", ["alpha", "depth"])
def test_renderer_semantic_visibility_rejects_valid_range_array_mutation(
    mutated_field: str,
) -> None:
    projection = {
        "visible_point_count": 1,
        "visible_class_names": ["doubles_left"],
        "courts": [
            {
                "classes": [
                    {
                        "class_name": "doubles_left",
                        "renderer_visible": True,
                        "points": [
                            {
                                "uv": [1.0, 1.0],
                                "in_frame": True,
                                "renderer_visible": True,
                            },
                            {
                                "uv": [10.0, 10.0],
                                "in_frame": False,
                                "renderer_visible": False,
                            },
                        ],
                    }
                ]
            }
        ],
    }
    alpha: NDArray[np.float32] = np.ones((3, 4, 1), dtype=np.float32)
    depth: NDArray[np.float32] = np.ones((3, 4, 1), dtype=np.float32)
    if mutated_field == "alpha":
        alpha.fill(0.0)
    else:
        depth.fill(0.0)

    with pytest.raises(ValueError, match="renderer-visible point disagrees"):
        _validate_renderer_visibility_payload(
            projection,
            alpha=alpha,
            depth=depth,
        )
