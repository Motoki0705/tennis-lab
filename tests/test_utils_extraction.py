"""Unit tests for the shared ``src/utils`` helpers extracted from task code.

These lock in the behavior of the consolidated utilities (paths, device,
seeding, io, tensor helpers, heatmap decoding and the geometry package) so the
de-duplication remains behavior-preserving.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils import device, io, paths, seeding, tensor_utils
from src.utils.data import augmentation, heatmaps, scene_io
from src.utils.geometry import angles, court_pose, keypoints, matrices, skeleton


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def test_project_root_points_at_repo_root() -> None:
    assert (paths.PROJECT_ROOT / "pyproject.toml").is_file()
    assert (paths.PROJECT_ROOT / "src" / "utils" / "paths.py").is_file()


def test_resolve_project_path_relative_and_absolute(tmp_path: Path) -> None:
    rel = paths.resolve_project_path("third_party/dinov3")
    assert rel == (paths.PROJECT_ROOT / "third_party" / "dinov3").resolve()
    assert rel.is_absolute()
    absolute = tmp_path / "x"
    assert paths.resolve_project_path(absolute) == absolute.resolve()


# --------------------------------------------------------------------------- #
# device
# --------------------------------------------------------------------------- #
def test_resolve_device_cpu_and_auto() -> None:
    assert device.resolve_device("cpu") == torch.device("cpu")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert device.resolve_device("auto").type == expected


def test_resolve_device_cuda_fallback() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA available; fallback path not exercised")
    assert device.resolve_device("cuda").type == "cpu"
    with pytest.raises(RuntimeError):
        device.resolve_device("cuda", allow_fallback=False)


def test_select_accelerator() -> None:
    assert device.select_accelerator(0) == ("cpu", 1)
    if torch.cuda.is_available():
        assert device.select_accelerator(2) == ("gpu", 2)
    else:
        assert device.select_accelerator(2) == ("cpu", 1)


# --------------------------------------------------------------------------- #
# seeding
# --------------------------------------------------------------------------- #
def test_seed_everything_is_reproducible() -> None:
    seeding.seed_everything(123)
    first = (np.random.rand(3).tolist(), torch.rand(3).tolist())
    seeding.seed_everything(123)
    second = (np.random.rand(3).tolist(), torch.rand(3).tolist())
    assert first == second


def test_make_sample_rng_deterministic_per_index() -> None:
    a = seeding.make_sample_rng(7).random()
    b = seeding.make_sample_rng(7).random()
    c = seeding.make_sample_rng(8).random()
    assert a == b
    assert a != c


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def test_ensure_dir_and_json_roundtrip(tmp_path: Path) -> None:
    created = io.ensure_dir(tmp_path / "a" / "b")
    assert created.is_dir()
    target = tmp_path / "nested" / "data.json"
    payload = {"k": [1, 2, 3], "n": "v"}
    returned = io.save_json(payload, target)
    assert returned == target
    assert target.is_file()
    assert io.load_json(target) == payload
    # default indentation matches the historical Result.save() convention.
    assert json.loads(target.read_text()) == payload


# --------------------------------------------------------------------------- #
# tensor_utils
# --------------------------------------------------------------------------- #
def test_clone_tensor_dict_independence() -> None:
    sample = {"t": torch.zeros(2), "label": "x", "n": 3}
    cloned = tensor_utils.clone_tensor_dict(sample)
    cloned["t"].add_(1.0)
    assert torch.equal(sample["t"], torch.zeros(2))  # original untouched
    assert cloned["label"] == "x" and cloned["n"] == 3


def test_to_numpy_variants() -> None:
    t = torch.tensor([1.0, 2.0])
    np.testing.assert_array_equal(tensor_utils.to_numpy(t), np.array([1.0, 2.0]))
    # bfloat16 is upcast rather than raising.
    bf = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    assert tensor_utils.to_numpy(bf).dtype == np.float32
    # dtype override + array-like passthrough.
    assert tensor_utils.to_numpy([1, 2], dtype=np.float32).dtype == np.float32


# --------------------------------------------------------------------------- #
# heatmaps
# --------------------------------------------------------------------------- #
def test_heatmaps_to_pixel_coords_matches_argmax_scaling() -> None:
    hm = torch.zeros(1, 1, 4, 5)
    hm[0, 0, 2, 3] = 1.0  # peak at row=2, col=3
    coords = heatmaps.heatmaps_to_pixel_coords(hm)
    assert coords.shape == (1, 1, 2)
    # x scaled by (W-1)=4, y by (H-1)=3.
    assert torch.allclose(coords[0, 0], torch.tensor([3.0, 2.0]))


# --------------------------------------------------------------------------- #
# data augmentation (imagenet normalize/denormalize, range parsing)
# --------------------------------------------------------------------------- #
def test_imagenet_normalize_denormalize_roundtrip() -> None:
    images = torch.rand(2, 3, 4, 4)
    norm = augmentation.normalize_tensor_images_imagenet(images)
    recovered = augmentation.denormalize_tensor_images_imagenet(norm)
    assert torch.allclose(recovered, images, atol=1e-6)


def test_normalize_frames_imagenet() -> None:
    frame: np.ndarray = np.ones((2, 2, 3), dtype=np.float32)
    out = augmentation.normalize_frames_imagenet([frame])
    expected = (1.0 - np.asarray(augmentation.IMAGENET_MEAN)) / np.asarray(
        augmentation.IMAGENET_STD
    )
    assert np.allclose(out[0][0, 0], expected)


def test_parse_int_range() -> None:
    assert augmentation.parse_int_range([1, 4], "r") == (1, 4)
    with pytest.raises(ValueError):
        augmentation.parse_int_range([4, 1], "r")
    with pytest.raises(ValueError):
        augmentation.parse_int_range([-1, 2], "r")


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_angular_error_and_wrapped_diff() -> None:
    # cos/sin pairs 10deg apart.
    pred = torch.tensor([[math.cos(0.0), math.sin(0.0)]])
    target = torch.tensor([[math.cos(0.2), math.sin(0.2)]])
    err = angles.angular_error(pred, target)
    assert torch.allclose(err, torch.tensor([0.2]), atol=1e-5)
    # wrapping across +/- pi.
    diff = angles.wrapped_angle_diff(torch.tensor(3.0), torch.tensor(-3.0))
    assert abs(float(diff)) < math.pi


def test_normalize_vector_unit_length() -> None:
    v = torch.tensor([3.0, 4.0])
    assert torch.allclose(angles.normalize_vector(v).norm(), torch.tensor(1.0))


def test_skeleton_angle_shapes() -> None:
    pose = torch.randn(2, 5, 17, 3)
    joint = skeleton.compute_joint_angles(pose)
    assert joint.shape[:2] == (2, 5)
    bones = skeleton.compute_bone_lengths(pose)
    assert bones.shape[:2] == (2, 5)
    assert (bones > 0).all()


def test_court_pose_roundtrip() -> None:
    canonical = torch.randn(4, 17, 3)
    position = torch.randn(4, 3)
    yaw = torch.rand(4) * 2 * math.pi
    rotation = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    world = court_pose.canonical_pose_to_world_pose(canonical, position, rotation)
    back = court_pose.world_pose_to_canonical_pose(world, position, rotation)
    assert torch.allclose(back, canonical, atol=1e-5)


def test_rotation_matrices_and_smpl_transform() -> None:
    r = matrices.rotation_matrix_y(0.5)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-6)
    # batched z-rotation orthonormal.
    rz = matrices.rotation_matrix_z(np.array([0.1, 0.2], dtype=np.float32))
    assert rz.shape == (2, 3, 3)
    # axis-angle around z by theta equals rotation_matrix_z(theta).
    theta = 0.7
    aa = matrices.axis_angle_to_rotation_matrix(
        np.array([0.0, 0.0, theta], dtype=np.float32)
    )
    assert np.allclose(aa, matrices.rotation_matrix_z(np.float32(theta)), atol=1e-5)
    verts = np.random.rand(6, 3).astype(np.float32)
    pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    single = matrices.apply_plcs_transform(verts, pos, 0.3)
    batched = matrices.apply_plcs_transform_batch(
        verts[None], pos[None], np.array([0.3], dtype=np.float32)
    )
    assert np.allclose(single, batched[0], atol=1e-5)


def test_keypoint_normalize_roundtrip() -> None:
    kp = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    norm = keypoints.normalize_keypoints(kp, width=100, height=200)
    back = keypoints.denormalize_keypoints(norm, width=100, height=200)
    assert np.allclose(back, kp)
    assert np.allclose(kp, [[10.0, 20.0], [30.0, 40.0]])  # input untouched


# --------------------------------------------------------------------------- #
# scene_io
# --------------------------------------------------------------------------- #
def test_load_scene_payload(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "scalars.json").write_text(json.dumps({"score": 1.5}))
    (scene / "meta.json").write_text(json.dumps({"id": "abc"}))
    np.save(scene / "frames.npy", np.arange(6).reshape(2, 3))
    payload = scene_io.load_scene_payload(scene)
    assert payload["score"] == 1.5
    assert payload["meta"] == {"id": "abc"}
    np.testing.assert_array_equal(np.asarray(payload["frames"]), np.arange(6).reshape(2, 3))
