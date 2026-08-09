from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.synthetic_data_generation.rendering.nht.contracts import (
    NHTRenderCamera,
    NHTRenderCommandRequest,
    NHTRenderRequest,
)
from src.synthetic_data_generation.scene_contract import RigidTransform


def _camera(camera_id: str = "novel") -> NHTRenderCamera:
    return NHTRenderCamera(
        camera_id=camera_id,
        width=16,
        height=12,
        intrinsics=(10.0, 0.0, 8.0, 0.0, 10.0, 6.0, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.from_matrix(np.eye(4)),
    )


def test_render_request_writes_strict_public_payload(tmp_path: Path) -> None:
    request = NHTRenderRequest((_camera(),))
    path = request.write(tmp_path / "request.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "nht_render_request_v1"
    assert payload["cameras"][0]["intrinsics"]["model"] == "PINHOLE"
    assert payload["cameras"][0]["intrinsics"]["params"] == [10.0, 10.0, 8.0, 6.0]


def test_render_command_combines_observed_and_file_request(tmp_path: Path) -> None:
    scene = tmp_path / "B00/reconstruction/export/scene.json"
    scene.parent.mkdir(parents=True)
    scene.write_text("{}", encoding="utf-8")
    arbitrary = NHTRenderRequest((_camera(),))
    command = NHTRenderCommandRequest(
        scene_path=scene,
        output_directory=tmp_path / "renders/shard-0",
        observed_camera_ids=("frame_000000",),
        arbitrary_cameras=arbitrary,
        arbitrary_request_path=tmp_path / "requests/shard-0.json",
    )

    assert command.expected_camera_ids == ("frame_000000", "novel")
    assert command.argv() == (
        "nht-render",
        "--scene",
        str(scene),
        "--camera-id",
        "frame_000000",
        "--cameras",
        str(tmp_path / "requests/shard-0.json"),
        "--output",
        str(tmp_path / "renders/shard-0"),
    )


def test_render_command_rejects_overlapping_camera_ids(tmp_path: Path) -> None:
    scene = tmp_path / "B00/reconstruction/export/scene.json"
    scene.parent.mkdir(parents=True)
    scene.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must not overlap"):
        NHTRenderCommandRequest(
            scene_path=scene,
            output_directory=tmp_path / "render",
            observed_camera_ids=("same",),
            arbitrary_cameras=NHTRenderRequest((_camera("same"),)),
            arbitrary_request_path=tmp_path / "request.json",
        )
