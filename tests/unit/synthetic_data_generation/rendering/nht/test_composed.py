"""Tests for the strict public composed-render client and sparse result contract."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)
from src.synthetic_data_generation.rendering.nht import client, composed
from src.synthetic_data_generation.rendering.nht.composed import (
    NHTComposedChunkRecord,
    NHTComposedRenderClient,
    NHTComposedRenderCommandRequest,
)
from src.synthetic_data_generation.rendering.nht.contracts import (
    NHTRenderCamera,
    NHTRenderCommandRequest,
    NHTRenderRequest,
)
from src.synthetic_data_generation.scene_contract import RigidTransform


def _camera() -> NHTRenderCamera:
    return NHTRenderCamera(
        camera_id="novel",
        width=8,
        height=6,
        intrinsics=(5.0, 0.0, 4.0, 0.0, 5.0, 3.0, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.from_matrix(np.eye(4)),
    )


def _scene(scene_path: Path) -> StandardSceneExport:
    (scene_path.parent / "cameras.json").write_text("{}", encoding="utf-8")
    np.save(scene_path.parent / "points_scene.npy", np.empty((0, 6), np.float32))
    model = scene_path.parent / "model"
    model.mkdir(exist_ok=True)
    checkpoint = model / "model.pt"
    runtime = model / "runtime.json"
    checkpoint.write_bytes(b"model")
    runtime.write_text("{}", encoding="utf-8")
    points: NDArray[np.float32] = np.empty((0, 6), dtype=np.float32)
    points.setflags(write=False)
    return StandardSceneExport(
        scene_id="B00",
        export_root=scene_path.parent,
        scene_path=scene_path,
        cameras=(),
        points_scene=points,
        scene_from_sfm=tuple(float(value) for value in np.eye(4).ravel()),
        sfm_from_scene=tuple(float(value) for value in np.eye(4).ravel()),
        checkpoint_path=checkpoint,
        runtime_config_path=runtime,
    )


def _command(tmp_path: Path) -> NHTComposedRenderCommandRequest:
    scene_path = tmp_path / "B00/reconstruction/export/scene.json"
    scene_path.parent.mkdir(parents=True)
    scene_path.write_text("{}", encoding="utf-8")
    request_root = tmp_path / "datasets/blcs/staging/composition"
    request_root.mkdir(parents=True)
    (request_root / "asset.npz").write_bytes(b"asset")
    (request_root / "timeline.npz").write_bytes(b"timeline")
    composition_path = request_root / "composition.json"
    composition_path.write_text(
        json.dumps(
            {
                "schema": "nht_composed_render_request_v1",
                "asset": {
                    "asset_id": "ball",
                    "coordinate_space": "right_handed_asset_local_metres",
                    "appearance_model": "direct_linear_rgb",
                    "gaussian_count": 64,
                    "tensors": "asset.npz",
                },
                "timeline": {
                    "coordinate_space": "canonical NHT scene space",
                    "frame_count": 1,
                    "object_count": 1,
                    "object_ids": ["ball-001"],
                    "instance_ids": [1],
                    "tensors": "timeline.npz",
                    "chunks": [{"chunk_index": 0, "frame_indices": [0]}],
                },
                "visibility_threshold": 0.0001,
            }
        ),
        encoding="utf-8",
    )
    return NHTComposedRenderCommandRequest(
        base=NHTRenderCommandRequest(
            scene_path=scene_path,
            output_directory=tmp_path / "datasets/blcs/staging/render",
            arbitrary_cameras=NHTRenderRequest((_camera(),)),
            arbitrary_request_path=tmp_path
            / "datasets/blcs/staging/cameras.json",
        ),
        composition_request_path=composition_path.resolve(),
    )


def _write_result(output: Path, *, instance_id: int = 1) -> None:
    background = output / "background"
    camera = background / "novel"
    camera.mkdir(parents=True)
    np.save(camera / "rgb.npy", np.full((6, 8, 3), 0.1, np.float32))
    np.save(camera / "alpha.npy", np.ones((6, 8, 1), np.float32))
    np.save(camera / "depth.npy", np.full((6, 8, 1), 10.0, np.float32))
    Image.new("RGB", (8, 6)).save(camera / "rgb.png")
    Image.new("L", (8, 6)).save(camera / "alpha.png")
    (background / "render.json").write_text(
        json.dumps(
            {
                "schema": "nht_render_result_v1",
                "scene_schema": "nht_standard_scene_v1",
                "scene_id": "B00",
                "coordinate_space": "canonical NHT scene space",
                "export_validation": {},
                "renders": [
                    {
                        "camera_id": "novel",
                        "request_source": "arbitrary",
                        "width": 8,
                        "height": 6,
                        "rgb": "novel/rgb.npy",
                        "rgb_preview": "novel/rgb.png",
                        "alpha": "novel/alpha.npy",
                        "alpha_preview": "novel/alpha.png",
                        "depth": "novel/depth.npy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    chunk = output / "chunks/chunk-000000"
    chunk.mkdir(parents=True)
    np.savez(
        chunk / "composed.npz",
        frame_indices=np.asarray([0], dtype=np.int64),
        camera_indices=np.asarray([0], dtype=np.int32),
        offsets=np.asarray([0, 1], dtype=np.int64),
        pixel_indices=np.asarray([7], dtype=np.int32),
        rgb=np.asarray([[0.72, 0.92, 0.08]], dtype=np.float32),
        alpha=np.asarray([0.97], dtype=np.float32),
        depth=np.asarray([2.0], dtype=np.float32),
        instance_ids=np.asarray([instance_id], dtype=np.int32),
    )
    (output / "render.json").write_text(
        json.dumps(
            {
                "schema": "nht_composed_render_result_v1",
                "scene_schema": "nht_standard_scene_v1",
                "scene_id": "B00",
                "coordinate_space": "canonical NHT scene space",
                "background": "background/render.json",
                "composition": {
                    "request_schema": "nht_composed_render_request_v1",
                    "frame_count": 1,
                    "object_count": 1,
                    "asset_gaussian_count": 64,
                    "appearance_model": "direct_linear_rgb",
                    "rasterization": "joint_3dgs_eval3d_transmittance_v1",
                    "visibility_threshold": 0.0001,
                },
                "chunks": [
                    {
                        "chunk_id": "chunk-000000",
                        "frame_indices": [0],
                        "camera_ids": ["novel"],
                        "sample_count": 1,
                        "pixel_count": 1,
                        "arrays": "chunks/chunk-000000/composed.npz",
                    }
                ],
                "cuda_peak_bytes": 4096,
            }
        ),
        encoding="utf-8",
    )


def test_composed_client_invokes_public_flag_and_validates_lazy_sparse_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command(tmp_path)
    observed_argv: list[str] = []
    monkeypatch.setattr(client, "validate_standard_scene_export", _scene)

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed_argv.extend(argv)
        assert kwargs["shell"] is False
        _write_result(command.base.output_directory)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(composed.subprocess, "run", fake_run)

    result = NHTComposedRenderClient().render_composed(command)

    assert observed_argv == list(command.argv())
    assert observed_argv[observed_argv.index("--composition") + 1] == str(
        command.composition_request_path
    )
    assert result.scene_id == "B00"
    assert result.appearance_model == "direct_linear_rgb"
    assert result.rasterization == "joint_3dgs_eval3d_transmittance_v1"
    assert result.background.record("novel").load_arrays().rgb.shape == (6, 8, 3)
    arrays = result.chunks[0].load_arrays()
    assert arrays.pixel_indices.tolist() == [7]
    assert arrays.instance_ids.tolist() == [1]
    assert not arrays.rgb.flags.writeable


def test_composed_sparse_result_rejects_invalid_instance_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command(tmp_path)
    monkeypatch.setattr(client, "validate_standard_scene_export", _scene)

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        _write_result(command.base.output_directory, instance_id=2)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(composed.subprocess, "run", fake_run)
    result = NHTComposedRenderClient().render_composed(command)

    with pytest.raises(ValueError, match="instance IDs"):
        result.chunks[0].load_arrays()


def test_chunk_record_rejects_unsorted_pixels(tmp_path: Path) -> None:
    arrays_path = tmp_path / "composed.npz"
    np.savez(
        arrays_path,
        frame_indices=np.asarray([0], np.int64),
        camera_indices=np.asarray([0], np.int32),
        offsets=np.asarray([0, 2], np.int64),
        pixel_indices=np.asarray([4, 3], np.int32),
        rgb=np.ones((2, 3), np.float32),
        alpha=np.ones(2, np.float32),
        depth=np.ones(2, np.float32),
        instance_ids=np.ones(2, np.int32),
    )
    record = NHTComposedChunkRecord(
        chunk_id="chunk-000000",
        frame_indices=(0,),
        camera_ids=("novel",),
        sample_count=1,
        pixel_count=2,
        arrays_path=arrays_path,
        width=8,
        height=6,
        object_count=1,
    )

    with pytest.raises(ValueError, match="sorted, unique"):
        record.load_arrays()
