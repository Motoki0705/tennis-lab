"""Tests for the public BLCS-to-NHT Gaussian composition request."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.synthetic_data_generation.alignment import MetricSceneAdapter
from src.synthetic_data_generation.dataset.blcs.rendering.request import (
    write_blcs_nht_composition_request,
)
from src.synthetic_data_generation.dataset.blcs.timeline import build_blcs_plans


def test_request_serializes_asset_local_gaussians_and_nht_rigid_timeline(
    tmp_path: Path,
    two_court_layout,
    default_camera_profile,
    blcs_assets,
    blcs_trajectory_factory,
) -> None:
    plan = build_blcs_plans(
        (blcs_trajectory_factory("trajectory-0", frame_count=3),),
        dataset_scene_id="B00",
        layout=two_court_layout,
        camera_config=default_camera_profile,
        assets=blcs_assets,
        seed=9,
        chunk_size_frames=2,
    )[0]
    nht_from_metric = np.eye(4, dtype=np.float64)
    nht_from_metric[:3, :3] *= 0.25
    nht_from_metric[:3, 3] = (1.0, -2.0, 0.5)
    adapter = MetricSceneAdapter.from_nht_scene_from_metric_scene(nht_from_metric)

    files = write_blcs_nht_composition_request(
        tmp_path / "composition",
        plan=plan,
        assets=blcs_assets,
        metric_adapter=adapter,
    )

    payload = json.loads(files.request_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "nht_composed_render_request_v1"
    assert payload["asset"] == {
        "asset_id": blcs_assets.ball.asset_id,
        "coordinate_space": "right_handed_asset_local_metres",
        "appearance_model": "direct_linear_rgb",
        "gaussian_count": blcs_assets.ball.gaussian_count,
        "tensors": "ball-gaussians.npz",
    }
    assert payload["timeline"]["frame_count"] == 3
    assert payload["timeline"]["object_ids"] == ["ball-001"]
    assert payload["timeline"]["instance_ids"] == [1]
    assert payload["timeline"]["chunks"] == [
        {"chunk_index": 0, "frame_indices": [0, 1]},
        {"chunk_index": 1, "frame_indices": [2]},
    ]

    with np.load(files.asset_path, allow_pickle=False) as asset:
        assert set(asset.files) == {
            "means_m",
            "quats_wxyz",
            "log_scales_m",
            "opacity_logits",
            "colors_linear_rgb",
        }
        assert all(asset[name].dtype == np.float32 for name in asset.files)
        np.testing.assert_allclose(
            np.linalg.norm(asset["means_m"], axis=1),
            blcs_assets.settings.radius_m,
            atol=1.0e-7,
        )
    with np.load(files.timeline_path, allow_pickle=False) as timeline:
        assert timeline["transforms_nht_from_asset"].dtype == np.float64
        assert timeline["present"].dtype == np.bool_
        assert timeline["instance_ids"].dtype == np.int32
        np.testing.assert_array_equal(timeline["present"], plan.source.present)
        first_instance = plan.composition.frames[0].instances[0]
        metric_from_asset = first_instance.scene_from_asset.rigid.matrix()
        metric_from_asset[:3, :3] *= first_instance.scene_from_asset.scale
        np.testing.assert_allclose(
            timeline["transforms_nht_from_asset"][0, 0],
            nht_from_metric @ metric_from_asset,
            atol=1.0e-10,
        )


def test_request_refuses_to_replace_an_existing_directory(
    tmp_path: Path,
    two_court_layout,
    default_camera_profile,
    blcs_assets,
    blcs_trajectory_factory,
) -> None:
    plan = build_blcs_plans(
        (blcs_trajectory_factory("trajectory-0", frame_count=1),),
        dataset_scene_id="B00",
        layout=two_court_layout,
        camera_config=default_camera_profile,
        assets=blcs_assets,
        seed=1,
        chunk_size_frames=1,
    )[0]
    directory = tmp_path / "composition"
    directory.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        write_blcs_nht_composition_request(
            directory,
            plan=plan,
            assets=blcs_assets,
            metric_adapter=MetricSceneAdapter.from_nht_scene_from_metric_scene(
                np.eye(4, dtype=np.float64)
            ),
        )
