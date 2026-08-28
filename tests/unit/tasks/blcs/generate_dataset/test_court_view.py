"""BLCS generator and artifact tests for the shared CourtKP20 contract."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from src.tasks.base.generate_dataset import (
    CAMERA_VIEW_V2_SELECTOR,
    COURT_VIEW_METADATA_KEY,
    PHYSICAL_V1_SELECTOR,
    CourtKeypointContract,
    CourtKeypointContractMismatchError,
    CourtViewRecord,
    resolve_court_keypoint_contract,
)
from src.tasks.blcs.generate_dataset.io.dataset_io import (
    BLCSDatasetWriter,
    load_scene,
)
from src.tasks.blcs.generate_dataset.scene_generator import (
    BLCSSceneData,
    BLCSSceneGenerator,
    CameraData,
    GeneratorConfig,
)
from src.utils.projection.camera_projector import Camera, CameraView
from src.utils.schema.court import COURT_KP20_HALF_TURN_INDEX
from src.utils.schema.court_normalization import (
    normalize_court_position,
    normalize_court_velocity,
)


def _view(y: float) -> CameraView:
    center = torch.tensor([3.0, y, 5.0])
    camera = Camera(
        C=center,
        R=torch.eye(3),
        f=100.0,
        cx=50.0,
        cy=40.0,
        w=100,
        h=80,
    )
    return CameraView(
        camera=camera,
        camera_params={
            "C": center.tolist(),
            "R": torch.eye(3).tolist(),
            "f": 100.0,
            "cx": 50.0,
            "cy": 40.0,
            "w": 100,
            "h": 80,
        },
        court_kp_uv=torch.arange(40, dtype=torch.float32).reshape(20, 2),
        court_kp_vis=(torch.arange(20) % 2 == 0),
        points_uv=torch.zeros(3, 2),
        points_vis=torch.ones(3, dtype=torch.bool),
    )


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        (PHYSICAL_V1_SELECTOR, tuple(range(20))),
        (CAMERA_VIEW_V2_SELECTOR, COURT_KP20_HALF_TURN_INDEX),
    ],
)
def test_camera_data_applies_selected_mapping_once(
    selector: str,
    expected: tuple[int, ...],
) -> None:
    generator = BLCSSceneGenerator.__new__(BLCSSceneGenerator)
    generator.config = cast(
        GeneratorConfig,
        SimpleNamespace(
            court_keypoint_contract=resolve_court_keypoint_contract(selector)
        ),
    )
    physical = _view(12.0)

    camera = generator._camera_view_to_data(physical, camera_id="cam_0")

    np.testing.assert_array_equal(
        camera.court_kp_uv,
        physical.court_kp_uv.numpy()[np.asarray(expected)],
    )
    np.testing.assert_array_equal(
        camera.court_kp_vis,
        physical.court_kp_vis.numpy()[np.asarray(expected)],
    )
    assert camera.court_view is not None
    assert camera.court_view.semantic_to_physical == expected


def _scene(camera: CameraData) -> BLCSSceneData:
    position = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    velocity = torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32)
    return BLCSSceneData(
        scene_id="scene_000000",
        initial_from_cell=0,
        initial_from_side="near",
        rally_length=1,
        end_reason="finished",
        winner_side=None,
        shots=[],
        ball_pos_world=position,
        ball_pos_norm=normalize_court_position(position),
        ball_vel_world=velocity,
        ball_vel_norm=normalize_court_velocity(velocity),
        cameras=[camera],
        num_cameras_sampled=1,
        fps_out=30,
        sim_fps=120,
        physics_config_dict={},
        court_config_dict={},
        num_balls=1,
    )


def _write_v2_scene(
    tmp_path: Path,
) -> tuple[Path, CourtKeypointContract, CourtViewRecord]:
    generator = BLCSSceneGenerator.__new__(BLCSSceneGenerator)
    contract = resolve_court_keypoint_contract(CAMERA_VIEW_V2_SELECTOR)
    generator.config = cast(
        GeneratorConfig,
        SimpleNamespace(court_keypoint_contract=contract),
    )
    camera = generator._camera_view_to_data(_view(12.0), camera_id="cam_0")
    assert camera.court_view is not None
    writer = BLCSDatasetWriter(tmp_path, court_keypoint_contract=contract)
    scene_path = writer.save_scene(_scene(camera))
    writer.save_meta_json()
    return scene_path, contract, camera.court_view


def test_v2_writer_reader_publish_and_validate_camera_record(tmp_path: Path) -> None:
    scene_path, contract, expected_view = _write_v2_scene(tmp_path)

    loaded = load_scene(scene_path, court_keypoint_contract=contract)
    assert loaded["court_keypoint_contract"] == contract
    assert loaded["cameras"][0]["court_view"] == expected_view

    metadata = json.loads((scene_path / "meta.json").read_text(encoding="utf-8"))
    metadata.pop(COURT_VIEW_METADATA_KEY)
    (scene_path / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="court_keypoint_views"):
        load_scene(scene_path, court_keypoint_contract=contract)


def test_v2_reader_rejects_scalar_camera_count_mismatch(tmp_path: Path) -> None:
    scene_path, contract, _ = _write_v2_scene(tmp_path)
    scalars_path = scene_path / "scalars.json"
    scalars = json.loads(scalars_path.read_text(encoding="utf-8"))
    scalars["num_cameras"] = 2
    scalars_path.write_text(json.dumps(scalars), encoding="utf-8")

    with pytest.raises(ValueError, match="num_cameras must exactly match"):
        load_scene(scene_path, court_keypoint_contract=contract)


def test_v2_reader_rejects_scene_header_camera_count_mismatch(tmp_path: Path) -> None:
    scene_path, contract, _ = _write_v2_scene(tmp_path)
    metadata_path = scene_path / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["num_cameras"] = 999
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="num_cameras must exactly match"):
        load_scene(scene_path, court_keypoint_contract=contract)


def test_v2_reader_rejects_unstable_camera_slot_id(tmp_path: Path) -> None:
    scene_path, contract, _ = _write_v2_scene(tmp_path)
    metadata_path = scene_path / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    court_views = metadata[COURT_VIEW_METADATA_KEY]
    assert isinstance(court_views, list)
    assert isinstance(court_views[0], dict)
    court_views[0]["camera_id"] = "camera_from_another_scene"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="camera slot 0 requires stable ID 'cam_0'"):
        load_scene(scene_path, court_keypoint_contract=contract)


def test_v2_reader_rejects_camera_center_mismatch_before_array_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_path, contract, _ = _write_v2_scene(tmp_path)
    scalars_path = scene_path / "scalars.json"
    scalars = json.loads(scalars_path.read_text(encoding="utf-8"))
    camera_params = scalars["cam_0_params"]
    assert isinstance(camera_params, dict)
    camera_params["C"] = [3.0, 13.0, 5.0]
    scalars_path.write_text(json.dumps(scalars), encoding="utf-8")

    def _forbid_array_payload(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("BLCS arrays were read before CourtKP20 headers.")

    monkeypatch.setattr(np, "load", _forbid_array_payload)

    with pytest.raises(ValueError, match="camera center does not exactly match"):
        load_scene(scene_path, court_keypoint_contract=contract)


def test_writer_contract_mismatch_does_not_publish_partial_scene(
    tmp_path: Path,
) -> None:
    physical_contract = resolve_court_keypoint_contract(PHYSICAL_V1_SELECTOR)
    generator = BLCSSceneGenerator.__new__(BLCSSceneGenerator)
    generator.config = cast(
        GeneratorConfig,
        SimpleNamespace(court_keypoint_contract=physical_contract),
    )
    physical_camera = generator._camera_view_to_data(_view(12.0), camera_id="cam_0")
    writer = BLCSDatasetWriter(
        tmp_path,
        court_keypoint_contract=resolve_court_keypoint_contract(
            CAMERA_VIEW_V2_SELECTOR
        ),
    )
    root_metadata_before = (tmp_path / "meta.json").read_bytes()

    with pytest.raises(CourtKeypointContractMismatchError, match="uses"):
        writer.save_scene(_scene(physical_camera))

    assert (tmp_path / "meta.json").read_bytes() == root_metadata_before
    assert not (tmp_path / "scenes" / "scene_000000").exists()
    assert writer.scene_records == []
    assert writer.scene_counter == 0
