"""Unit tests for the CPU adapter that exports GVHMR world motion artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.submodules.configuration import BundledModelAssetPaths
from src.tennis_scene.motion_alignment import artifacts
from src.tennis_scene.motion_alignment.artifacts import (
    ALIGNED_COORDINATE_SYSTEM,
    RAW_SCHEMA_VERSION,
    confidence_diagnostics,
    load_world_motion,
    transform_smpl_parameters,
)
from src.utils.geometry.matrices import SMPL_Y_UP_TO_COURT_Z_UP, rotation_matrix_z
from src.utils.geometry.rotation_conversions import axis_angle_to_matrix

NUM_SMPL_VERTS = 6890
NUM_SMPLX_VERTS = 10475
NUM_BETAS = 10
NUM_FRAMES = 5
FPS = 30.0


def fake_vertices(frames: int) -> NDArray[np.float32]:
    """Deterministic stand-in for ``SmplVertexReconstructor.reconstruct``."""
    vertex_ids: NDArray[np.float32] = np.arange(NUM_SMPL_VERTS, dtype=np.float32)
    frame_ids: NDArray[np.float32] = np.arange(frames, dtype=np.float32)
    vertices: NDArray[np.float32] = np.zeros(
        (frames, NUM_SMPL_VERTS, 3), dtype=np.float32
    )
    vertices[:, :, 0] = vertex_ids[None, :] * 1e-3
    vertices[:, :, 1] = frame_ids[:, None]
    return vertices


@dataclass
class FakeReconstructor:
    """Records batch sizes so batching behaviour stays observable."""

    calls: list[int]

    def reconstruct(self, parameters: dict[str, torch.Tensor]) -> torch.Tensor:
        frames = int(parameters["body_pose"].shape[0])
        self.calls.append(frames)
        return torch.from_numpy(fake_vertices(frames))


def one_hot_regressor(rows: int, vertex_ids: NDArray[np.int64]) -> torch.Tensor:
    regressor: NDArray[np.float32] = np.zeros(
        (rows, NUM_SMPL_VERTS), dtype=np.float32
    )
    regressor[np.arange(rows), vertex_ids] = 1.0
    return torch.from_numpy(regressor)


def make_bundled_assets(root: Path) -> BundledModelAssetPaths:
    """Real (tiny) regressor files; the other asset paths are never read."""
    smpl_path = root / "smpl_neutral_J_regressor.pt"
    coco_path = root / "smpl_coco17_J_regressor.pt"
    torch.save(one_hot_regressor(24, np.arange(24, dtype=np.int64)), smpl_path)
    torch.save(
        one_hot_regressor(17, np.arange(24, 41, dtype=np.int64)), coco_path
    )
    return BundledModelAssetPaths(
        hmr2_mean_params=root / "hmr2_mean_params.npz",
        smplx_to_smpl=root / "smplx2smpl_sparse.pt",
        smpl_coco17_regressor=coco_path,
        smplx_verts437=root / "smplx_verts437.npz",
        smpl_neutral_joint_regressor=smpl_path,
    )


def make_body_models(root: Path) -> Path:
    """Fake SMPL-X model whose pelvis pivot is ``[0, 1 + 0.5 * beta0, 0]``."""
    body_models_dir = root / "body_models"
    smplx_dir = body_models_dir / "smplx"
    smplx_dir.mkdir(parents=True)
    v_template: NDArray[np.float64] = np.zeros((NUM_SMPLX_VERTS, 3), dtype=np.float64)
    v_template[0] = [0.0, 1.0, 0.0]
    shapedirs: NDArray[np.float64] = np.zeros(
        (NUM_SMPLX_VERTS, 3, NUM_BETAS), dtype=np.float64
    )
    shapedirs[0, 1, 0] = 0.5
    j_regressor: NDArray[np.float64] = np.zeros(
        (55, NUM_SMPLX_VERTS), dtype=np.float64
    )
    j_regressor[0, 0] = 1.0
    np.savez_compressed(
        smplx_dir / "SMPLX_NEUTRAL.npz",
        v_template=v_template,
        shapedirs=shapedirs,
        J_regressor=j_regressor,
    )
    return body_models_dir.resolve()


def write_record(
    path: Path,
    *,
    frames: int = NUM_FRAMES,
    metadata_overrides: dict[str, Any] | None = None,
    arrays: dict[str, NDArray[Any]] | None = None,
) -> Path:
    metadata: dict[str, Any] = {
        "schema_version": RAW_SCHEMA_VERSION,
        "coordinate_system": ALIGNED_COORDINATE_SYSTEM,
        "frame_count": frames,
        "native_fps": FPS,
        "source_id": "unit/test",
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    values: dict[str, NDArray[Any]] = {
        "metadata_json": np.asarray(json.dumps(metadata)),
        "body_pose": np.zeros((frames, 63), dtype=np.float32),
        "betas": np.full((frames, NUM_BETAS), 0.4, dtype=np.float32),
        "global_orient": np.zeros((frames, 3), dtype=np.float32),
        "transl": np.arange(frames * 3, dtype=np.float32).reshape(frames, 3) * 0.1,
        "K_fullimg": np.tile(np.eye(3, dtype=np.float32), (frames, 1, 1)),
        "keypoints_2d_px": np.zeros((frames, 17, 3), dtype=np.float32),
        "boxes_xys_px": np.zeros((frames, 3), dtype=np.float32),
        "observed_mask": np.ones((frames,), dtype=np.bool_),
    }
    if arrays:
        values.update(arrays)
    np.savez_compressed(path, **values)
    return path


def install_fake_reconstructor(
    monkeypatch: pytest.MonkeyPatch, reconstructor: FakeReconstructor
) -> None:
    def build(**_: Any) -> FakeReconstructor:
        return reconstructor

    monkeypatch.setattr(artifacts, "_build_reconstructor", build)


def test_confidence_diagnostics_counts_out_of_range_values() -> None:
    diagnostics = confidence_diagnostics(
        np.array([[0.5, -0.2, 1.7, np.nan]], dtype=np.float64)
    )
    assert diagnostics.total_values == 4
    assert diagnostics.below_zero == 1
    assert diagnostics.above_one == 1
    assert diagnostics.nonfinite == 1
    assert diagnostics.needs_clipping


def test_confidence_diagnostics_reports_clean_input() -> None:
    diagnostics = confidence_diagnostics(np.full((5, 17), 0.9))
    assert diagnostics.total_values == 85
    assert not diagnostics.needs_clipping


def test_transform_smpl_parameters_reproduces_rigid_map() -> None:
    rng = np.random.default_rng(0)
    frames = 3
    rest = rng.normal(size=(6, 3))
    pivot = np.array([0.3, 1.1, -0.2])
    scale = 1.8
    yaw = 0.7
    translation = np.array([1.0, -2.0, 0.5])
    axis_angles = rng.normal(size=(frames, 3)) * 0.4
    transl = rng.normal(size=(frames, 3))
    body_pose = rng.normal(size=(frames, 63)).astype(np.float32)
    betas = rng.normal(size=(frames, NUM_BETAS)).astype(np.float32)

    rotation = axis_angle_to_matrix(torch.from_numpy(axis_angles)).numpy()
    rest_local = rest - pivot[None, :]
    vertices = np.einsum("tij,vj->tvi", rotation, rest_local) + pivot + transl[:, None, :]

    court_map = (
        np.asarray(rotation_matrix_z(np.asarray(yaw, dtype=np.float32)), dtype=np.float64)
        @ np.asarray(SMPL_Y_UP_TO_COURT_Z_UP, dtype=np.float64)
    )
    expected = scale * np.einsum("ij,tvj->tvi", court_map, vertices) + translation

    transformed = transform_smpl_parameters(
        body_pose=body_pose,
        betas=betas,
        global_orient=axis_angles.astype(np.float32),
        transl=transl.astype(np.float32),
        rest_pelvis=np.tile(pivot, (frames, 1)).astype(np.float32),
        scale=scale,
        yaw=yaw,
        translation=translation,
    )

    out_rotation = axis_angle_to_matrix(
        torch.from_numpy(transformed["global_orient"].astype(np.float64))
    ).numpy()
    # The exported convention reconstructs with transl=0, then scales, then adds
    # the returned offset.
    reconstructed = np.einsum("tij,vj->tvi", out_rotation, rest_local) + pivot
    actual = transformed["body_scale"] * reconstructed + transformed["transl"][:, None, :]

    assert transformed["body_scale"] == pytest.approx(scale)
    np.testing.assert_allclose(actual, expected, atol=1e-4)
    np.testing.assert_array_equal(transformed["body_pose"], body_pose)
    np.testing.assert_array_equal(transformed["betas"], betas)


def test_transform_smpl_parameters_keeps_rotation_unscaled() -> None:
    frames = 4
    rng = np.random.default_rng(3)
    axis_angles = rng.normal(size=(frames, 3))
    yaw = 0.9

    transformed = transform_smpl_parameters(
        body_pose=np.zeros((frames, 63), dtype=np.float32),
        betas=np.zeros((frames, NUM_BETAS), dtype=np.float32),
        global_orient=axis_angles.astype(np.float32),
        transl=np.zeros((frames, 3), dtype=np.float32),
        rest_pelvis=np.zeros((frames, 3), dtype=np.float32),
        scale=4.0,
        yaw=yaw,
        translation=np.zeros(3),
    )

    rotation = axis_angle_to_matrix(
        torch.from_numpy(transformed["global_orient"].astype(np.float64))
    ).numpy()
    norms = np.linalg.norm(rotation, axis=1)
    np.testing.assert_allclose(norms, np.ones((frames, 3)), atol=1e-5)

    source = axis_angle_to_matrix(torch.from_numpy(axis_angles)).numpy()
    basis = np.asarray(SMPL_Y_UP_TO_COURT_Z_UP, dtype=np.float64)
    expected = (
        np.asarray(rotation_matrix_z(np.asarray(yaw, dtype=np.float32)), dtype=np.float64)
        @ basis
    ) @ source
    np.testing.assert_allclose(rotation, expected, atol=1e-6)

    heading_in = np.arctan2((basis @ source)[:, 1, 0], (basis @ source)[:, 0, 0])
    heading_out = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
    np.testing.assert_allclose(
        np.angle(np.exp(1j * (heading_out - heading_in - yaw))), 0.0, atol=1e-6
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"scale": 0.0}, "scale"),
        ({"scale": -1.0}, "scale"),
        ({"scale": float("nan")}, "scale"),
        ({"yaw": float("inf")}, "yaw"),
        ({"translation": np.zeros(2)}, "translation"),
        ({"transl": np.zeros((2, 3))}, "frame count"),
        ({"body_pose": np.full((3, 63), np.nan)}, "non-finite"),
    ],
)
def test_transform_smpl_parameters_rejects_invalid_inputs(
    overrides: dict[str, Any], message: str
) -> None:
    frames = 3
    arguments: dict[str, Any] = {
        "body_pose": np.zeros((frames, 63), dtype=np.float32),
        "betas": np.zeros((frames, NUM_BETAS), dtype=np.float32),
        "global_orient": np.zeros((frames, 3), dtype=np.float32),
        "transl": np.zeros((frames, 3), dtype=np.float32),
        "rest_pelvis": np.zeros((frames, 3), dtype=np.float32),
        "scale": 1.0,
        "yaw": 0.0,
        "translation": np.zeros(3),
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        transform_smpl_parameters(**arguments)


def test_load_world_motion_exports_court_frame_motion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = make_bundled_assets(tmp_path)
    body_models_dir = make_body_models(tmp_path)
    record = write_record(tmp_path / "cam1.gvhmr.npz")
    reconstructor = FakeReconstructor(calls=[])
    install_fake_reconstructor(monkeypatch, reconstructor)

    motion = load_world_motion(
        record, body_models_dir=body_models_dir, bundled_assets=assets
    )

    basis = np.asarray(SMPL_Y_UP_TO_COURT_Z_UP, dtype=np.float64)
    vertices = fake_vertices(NUM_FRAMES).astype(np.float64)
    expected_joints = vertices[:, :24] @ basis.T
    np.testing.assert_allclose(motion.joints_smpl, expected_joints, atol=1e-6)
    np.testing.assert_allclose(motion.joints_coco, vertices[:, 24:41] @ basis.T)
    np.testing.assert_allclose(motion.root_position, motion.joints_smpl[:, 0])
    np.testing.assert_allclose(motion.rest_pelvis, np.tile([0.0, 1.2, 0.0], (5, 1)))
    np.testing.assert_allclose(motion.rotation, np.tile(basis[None], (5, 1, 1)))
    assert motion.root_position.dtype == np.float64
    assert motion.joints_smpl.shape == (NUM_FRAMES, 24, 3)
    assert motion.joints_coco.shape == (NUM_FRAMES, 17, 3)
    assert motion.fps == pytest.approx(FPS)
    assert motion.metadata["source_id"] == "unit/test"
    assert reconstructor.calls == [NUM_FRAMES]


def test_load_world_motion_preserves_raw_arrays_and_confidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = make_bundled_assets(tmp_path)
    body_models_dir = make_body_models(tmp_path)
    keypoints: NDArray[np.float32] = np.zeros((NUM_FRAMES, 17, 3), dtype=np.float32)
    keypoints[:, :, 2] = 0.5
    keypoints[0, 0, 2] = 1.4  # raw ViTPose confidence above one
    keypoints[1, 0, 2] = -0.3
    observed = np.array([True, True, False, True, False])
    record = write_record(
        tmp_path / "cam1.gvhmr.npz",
        arrays={"keypoints_2d_px": keypoints, "observed_mask": observed},
    )
    install_fake_reconstructor(monkeypatch, FakeReconstructor(calls=[]))

    motion = load_world_motion(
        record, body_models_dir=body_models_dir, bundled_assets=assets
    )

    np.testing.assert_array_equal(motion.keypoints_2d_px, keypoints)
    np.testing.assert_allclose(motion.confidence, keypoints[:, :, 2].astype(np.float64))
    np.testing.assert_array_equal(motion.observed, observed)
    assert motion.confidence_diagnostics.above_one == 1
    assert motion.confidence_diagnostics.below_zero == 1
    assert motion.confidence_diagnostics.needs_clipping
    np.testing.assert_array_equal(motion.raw_betas, np.full((5, 10), 0.4, np.float32))


def test_load_world_motion_batches_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = make_bundled_assets(tmp_path)
    body_models_dir = make_body_models(tmp_path)
    record = write_record(tmp_path / "cam1.gvhmr.npz")
    reconstructor = FakeReconstructor(calls=[])
    install_fake_reconstructor(monkeypatch, reconstructor)

    load_world_motion(
        record, body_models_dir=body_models_dir, bundled_assets=assets, batch_size=2
    )

    assert reconstructor.calls == [2, 2, 1]


@pytest.mark.parametrize(
    ("metadata_overrides", "arrays", "message"),
    [
        ({"schema_version": "other"}, None, "schema"),
        ({"coordinate_system": "y_up"}, None, "coordinate_system"),
        ({"frame_count": 4}, None, "frame_count"),
        ({"native_fps": 0.0}, None, "native_fps"),
        ({}, {"body_pose": np.zeros((5, 62), dtype=np.float32)}, "width of 63"),
        ({}, {"transl": np.full((5, 3), np.nan, dtype=np.float32)}, "non-finite"),
        ({}, {"observed_mask": np.ones((4,), dtype=np.bool_)}, "observed_mask"),
        ({}, {"observed_mask": np.ones((5,), dtype=np.float32)}, "boolean array"),
        ({}, {"observed_mask": np.full((5,), 2, dtype=np.int64)}, "boolean array"),
        (
            {},
            {"observed_mask": np.full((5,), np.nan, dtype=np.float64)},
            "boolean array",
        ),
        ({}, {"keypoints_2d_px": np.zeros((5, 16, 3), np.float32)}, "keypoints_2d_px"),
        ({}, {"boxes_xys_px": np.zeros((5, 2), np.float32)}, "boxes_xys_px"),
    ],
)
def test_load_world_motion_rejects_invalid_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata_overrides: dict[str, Any],
    arrays: dict[str, NDArray[Any]] | None,
    message: str,
) -> None:
    assets = make_bundled_assets(tmp_path)
    body_models_dir = make_body_models(tmp_path)
    record = write_record(
        tmp_path / "cam1.gvhmr.npz",
        metadata_overrides=metadata_overrides,
        arrays=arrays,
    )
    install_fake_reconstructor(monkeypatch, FakeReconstructor(calls=[]))

    with pytest.raises(ValueError, match=message):
        load_world_motion(record, body_models_dir=body_models_dir, bundled_assets=assets)


def test_load_world_motion_rejects_missing_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = make_bundled_assets(tmp_path)
    body_models_dir = make_body_models(tmp_path)
    record = tmp_path / "cam1.gvhmr.npz"
    np.savez_compressed(
        record,
        metadata_json=np.asarray(json.dumps({"schema_version": RAW_SCHEMA_VERSION})),
    )
    install_fake_reconstructor(monkeypatch, FakeReconstructor(calls=[]))

    with pytest.raises(ValueError, match="missing keys"):
        load_world_motion(record, body_models_dir=body_models_dir, bundled_assets=assets)


def test_load_world_motion_rejects_missing_record(tmp_path: Path) -> None:
    assets = make_bundled_assets(tmp_path)
    with pytest.raises(FileNotFoundError, match="missing"):
        load_world_motion(
            tmp_path / "absent.gvhmr.npz",
            body_models_dir=make_body_models(tmp_path),
            bundled_assets=assets,
        )


@pytest.mark.parametrize("batch_size", [0, -3, 1.5, "32"])
def test_load_world_motion_rejects_invalid_batch_size(
    tmp_path: Path, batch_size: Any
) -> None:
    assets = make_bundled_assets(tmp_path)
    record = write_record(tmp_path / "cam1.gvhmr.npz")
    with pytest.raises(ValueError, match="batch_size"):
        load_world_motion(
            record,
            body_models_dir=make_body_models(tmp_path),
            bundled_assets=assets,
            batch_size=batch_size,
        )


def test_load_world_motion_rejects_non_bundled_assets(tmp_path: Path) -> None:
    record = write_record(tmp_path / "cam1.gvhmr.npz")
    with pytest.raises(TypeError, match="BundledModelAssetPaths"):
        load_world_motion(
            record,
            body_models_dir=make_body_models(tmp_path),
            bundled_assets={"smplx_to_smpl": tmp_path},
        )
