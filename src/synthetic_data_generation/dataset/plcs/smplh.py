"""Strict SMPL-H loading and bounded CUDA Gaussian linear-blend skinning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from smplx.lbs import (  # type: ignore[import-untyped]
    batch_rigid_transform,
    batch_rodrigues,
    blend_shapes,
    vertices2joints,
)
from torch import Tensor

from src.synthetic_data_generation.dataset.plcs.components.avatar_asset import (
    AvatarGaussianAsset,
    build_surface_gaussian_asset,
)
from src.tasks.plcs.generate_dataset.sampling.motion_sampler import PLCSMotionClip

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

_VERTEX_COUNT = 6890
_JOINT_COUNT = 52
_POSE_BLEND_WIDTH = (_JOINT_COUNT - 1) * 9
_REQUIRED_KEYS = {
    "v_template",
    "f",
    "shapedirs",
    "posedirs",
    "J_regressor",
    "kintree_table",
    "weights",
}


def _finite_float(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return cast(FloatArray, np.ascontiguousarray(array))


def _integer(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> IntArray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must use an integer dtype.")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    return cast(IntArray, np.ascontiguousarray(array, dtype=np.int64))


@dataclass(frozen=True, slots=True)
class SMPLHModelData:
    """Validated official SMPL-H arrays for one explicit gender."""

    model_path: Path
    gender: str
    template_vertices_m: FloatArray
    faces: IntArray
    shape_directions: FloatArray
    pose_directions: FloatArray
    joint_regressor: FloatArray
    parents: IntArray
    vertex_joint_weights: FloatArray

    def __post_init__(self) -> None:
        if self.gender not in {"female", "male", "neutral"}:
            raise ValueError("SMPL-H gender must be female, male, or neutral.")
        if self.model_path.name != "model.npz" or not self.model_path.is_file():
            raise FileNotFoundError(
                f"Expected an explicit licensed SMPL-H model.npz: {self.model_path}"
            )
        template = _finite_float(
            self.template_vertices_m,
            name="template_vertices_m",
            shape=(_VERTEX_COUNT, 3),
        )
        faces = _integer(self.faces, name="faces")
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
            raise ValueError("SMPL-H faces must have non-empty shape [F,3].")
        if np.any(faces < 0) or np.any(faces >= _VERTEX_COUNT):
            raise ValueError("SMPL-H faces contain an out-of-range vertex.")
        shapedirs = _finite_float(self.shape_directions, name="shape_directions")
        if shapedirs.ndim != 3 or shapedirs.shape[:2] != (_VERTEX_COUNT, 3):
            raise ValueError("SMPL-H shapedirs must have shape [6890,3,B].")
        if shapedirs.shape[2] <= 0:
            raise ValueError("SMPL-H shapedirs must contain at least one beta basis.")
        posedirs = _finite_float(
            self.pose_directions,
            name="pose_directions",
            shape=(_POSE_BLEND_WIDTH, _VERTEX_COUNT * 3),
        )
        regressor = _finite_float(
            self.joint_regressor,
            name="joint_regressor",
            shape=(_JOINT_COUNT, _VERTEX_COUNT),
        )
        parents = _integer(self.parents, name="parents", shape=(_JOINT_COUNT,))
        if parents[0] != -1 or any(
            int(parent) < 0 or int(parent) >= index
            for index, parent in enumerate(parents[1:], start=1)
        ):
            raise ValueError("SMPL-H parents must define a forward acyclic tree.")
        weights = _finite_float(
            self.vertex_joint_weights,
            name="vertex_joint_weights",
            shape=(_VERTEX_COUNT, _JOINT_COUNT),
        )
        if np.any(weights < 0.0) or not np.allclose(
            weights.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0
        ):
            raise ValueError("SMPL-H LBS weights must be an explicit simplex.")
        for name, value in (
            ("template_vertices_m", template),
            ("faces", faces),
            ("shape_directions", shapedirs),
            ("pose_directions", posedirs),
            ("joint_regressor", regressor),
            ("parents", parents),
            ("vertex_joint_weights", weights),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def beta_count(self) -> int:
        """Return the exact shape-basis width of this licensed model."""
        return int(self.shape_directions.shape[2])


def load_smplh_model(model_root: str | Path, *, gender: str) -> SMPLHModelData:
    """Load one explicit ``smplh/<gender>/model.npz`` without adaptation."""
    normalized_gender = gender.strip().lower()
    if normalized_gender not in {"female", "male", "neutral"}:
        raise ValueError("SMPL-H gender must be female, male, or neutral.")
    root = Path(model_root).resolve()
    direct = root / normalized_gender / "model.npz"
    nested = root / "smplh" / normalized_gender / "model.npz"
    candidates = [path for path in (direct, nested) if path.is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Exactly one explicit SMPL-H model is required at "
            f"{direct} or {nested}; found={candidates}."
        )
    model_path = candidates[0]
    with np.load(model_path, allow_pickle=False) as archive:
        missing = _REQUIRED_KEYS.difference(archive.files)
        if missing:
            raise ValueError(f"SMPL-H archive is missing arrays: {sorted(missing)}.")
        kintree = _integer(
            archive["kintree_table"],
            name="kintree_table",
            shape=(2, _JOINT_COUNT),
        )
        if not np.array_equal(kintree[1], np.arange(_JOINT_COUNT)):
            raise ValueError("SMPL-H kintree joint IDs must be contiguous 0..51.")
        raw_parents = kintree[0].astype(np.uint64, copy=False)
        parents = raw_parents.astype(np.int64, copy=True)
        parents[0] = -1
        posedirs = np.asarray(archive["posedirs"], dtype=np.float64)
        if posedirs.shape != (_VERTEX_COUNT, 3, _POSE_BLEND_WIDTH):
            raise ValueError(
                "SMPL-H posedirs must have shape "
                f"{(_VERTEX_COUNT, 3, _POSE_BLEND_WIDTH)}, got {posedirs.shape}."
            )
        return SMPLHModelData(
            model_path=model_path,
            gender=normalized_gender,
            template_vertices_m=archive["v_template"],
            faces=archive["f"],
            shape_directions=archive["shapedirs"],
            pose_directions=posedirs.reshape(_VERTEX_COUNT * 3, _POSE_BLEND_WIDTH).T,
            joint_regressor=archive["J_regressor"],
            parents=parents,
            vertex_joint_weights=archive["weights"],
        )


@dataclass(frozen=True, slots=True)
class SMPLHDeviceModel:
    """One gender model uploaded once to the stage CUDA device."""

    gender: str
    template_vertices_m: Tensor
    shape_directions: Tensor
    joint_regressor: Tensor
    parents: Tensor

    def __post_init__(self) -> None:
        tensors = (
            self.template_vertices_m,
            self.shape_directions,
            self.joint_regressor,
            self.parents,
        )
        if any(value.device.type != "cuda" for value in tensors):
            raise ValueError("Production SMPL-H model buffers must be CUDA-resident.")
        if any(value.device != tensors[0].device for value in tensors):
            raise ValueError("SMPL-H model buffers must share one CUDA device.")
        if self.template_vertices_m.dtype != torch.float32:
            raise TypeError("Production SMPL-H model buffers must use float32.")
        if self.shape_directions.dtype != torch.float32:
            raise TypeError("Production SMPL-H shape directions must use float32.")
        if self.joint_regressor.dtype != torch.float32:
            raise TypeError("Production SMPL-H joint regressor must use float32.")
        if self.parents.dtype != torch.int64:
            raise TypeError("Production SMPL-H parents must use int64.")

    @property
    def device(self) -> torch.device:
        """Return the exact resident CUDA device."""
        return self.template_vertices_m.device


@dataclass(frozen=True, slots=True)
class SMPLHDeviceClip:
    """One complete source clip uploaded once for bounded CUDA evaluation."""

    source_path: str
    full_pose_axis_angle: Tensor
    betas: Tensor

    def __post_init__(self) -> None:
        if (
            self.full_pose_axis_angle.device.type != "cuda"
            or self.betas.device.type != "cuda"
        ):
            raise ValueError("Production SMPL-H motion buffers must be CUDA-resident.")
        if self.full_pose_axis_angle.device != self.betas.device:
            raise ValueError("SMPL-H pose and beta buffers must share one device.")
        if (
            self.full_pose_axis_angle.dtype != torch.float32
            or self.betas.dtype != torch.float32
        ):
            raise TypeError("Production SMPL-H motion buffers must use float32.")
        if (
            self.full_pose_axis_angle.ndim != 2
            or self.full_pose_axis_angle.shape[1] != 156
        ):
            raise ValueError("SMPL-H device pose buffer must have shape [T,156].")
        if self.betas.ndim != 1:
            raise ValueError("SMPL-H device beta buffer must be one-dimensional.")

    @property
    def frame_count(self) -> int:
        """Return the complete uploaded source length."""
        return int(self.full_pose_axis_angle.shape[0])


@dataclass(frozen=True, slots=True)
class SMPLHDeviceGaussianAsset:
    """Canonical Gaussian shell and controls resident on the stage device."""

    means_m: Tensor
    quaternions_wxyz: Tensor
    log_scales_m: Tensor
    opacity_logits: Tensor
    point_joint_weights: Tensor

    def __post_init__(self) -> None:
        values = (
            self.means_m,
            self.quaternions_wxyz,
            self.log_scales_m,
            self.opacity_logits,
            self.point_joint_weights,
        )
        if any(value.device.type != "cuda" for value in values):
            raise ValueError("Production Gaussian LBS buffers must be CUDA-resident.")
        if any(value.device != values[0].device for value in values):
            raise ValueError("Gaussian LBS buffers must share one CUDA device.")
        if any(value.dtype != torch.float32 for value in values):
            raise TypeError("Production Gaussian LBS buffers must use float32.")
        count = int(self.means_m.shape[0])
        if self.means_m.shape != (count, 3) or count <= 0:
            raise ValueError("Gaussian means must have non-empty shape [N,3].")
        if self.quaternions_wxyz.shape != (count, 4):
            raise ValueError("Gaussian quaternions must have shape [N,4].")
        if self.log_scales_m.shape != (count, 3):
            raise ValueError("Gaussian log scales must have shape [N,3].")
        if self.opacity_logits.shape != (count,):
            raise ValueError("Gaussian opacity logits must have shape [N].")
        if self.point_joint_weights.shape != (count, _JOINT_COUNT):
            raise ValueError("Gaussian joint weights must have shape [N,52].")

    @property
    def gaussian_count(self) -> int:
        return int(self.means_m.shape[0])


@dataclass(frozen=True, slots=True)
class SMPLHGaussianBatch:
    """Current bounded batch only; no full source geometry is retained."""

    source_frame_indices: tuple[int, ...]
    means_m: Tensor
    quaternions_wxyz: Tensor
    log_scales_m: Tensor
    joints_m: Tensor

    def __post_init__(self) -> None:
        count = len(self.source_frame_indices)
        if count <= 0 or len(set(self.source_frame_indices)) != count:
            raise ValueError("A Gaussian batch needs unique source frame indices.")
        if self.means_m.device.type != "cuda":
            raise ValueError("Production Gaussian batches must remain on CUDA.")
        if self.means_m.shape[:1] != (count,) or self.means_m.shape[-1] != 3:
            raise ValueError("Gaussian batch means must have shape [B,N,3].")
        if self.quaternions_wxyz.shape != (*self.means_m.shape[:-1], 4):
            raise ValueError("Gaussian batch quaternions have the wrong shape.")
        if self.log_scales_m.shape != self.means_m.shape:
            raise ValueError("Gaussian batch log scales have the wrong shape.")
        if self.joints_m.shape != (count, _JOINT_COUNT, 3):
            raise ValueError("Gaussian batch joints must have shape [B,52,3].")
        values = (self.means_m, self.quaternions_wxyz, self.log_scales_m, self.joints_m)
        if any(value.device != self.means_m.device for value in values):
            raise ValueError("Gaussian batch tensors must share one CUDA device.")
        if any(value.dtype != torch.float32 for value in values):
            raise TypeError("Production Gaussian batches must use float32.")


def upload_smplh_model(
    model: SMPLHModelData,
    *,
    device: str | torch.device,
) -> SMPLHDeviceModel:
    """Upload one already-loaded gender model exactly once to CUDA."""
    target = _cuda_device(device)
    return SMPLHDeviceModel(
        gender=model.gender,
        template_vertices_m=torch.as_tensor(
            np.array(model.template_vertices_m, copy=True),
            dtype=torch.float32,
            device=target,
        ),
        shape_directions=torch.as_tensor(
            np.array(model.shape_directions, copy=True),
            dtype=torch.float32,
            device=target,
        ),
        joint_regressor=torch.as_tensor(
            np.array(model.joint_regressor, copy=True),
            dtype=torch.float32,
            device=target,
        ),
        parents=torch.as_tensor(
            np.array(model.parents, copy=True),
            dtype=torch.int64,
            device=target,
        ),
    )


def upload_motion_clip(
    clip: PLCSMotionClip,
    model: SMPLHModelData,
    *,
    device: str | torch.device,
) -> SMPLHDeviceClip:
    """Upload every source pose once without frame selection or dtype widening."""
    target = _cuda_device(device)
    if model.gender != clip.gender:
        raise ValueError(
            f"SMPL-H model gender {model.gender!r} disagrees with clip {clip.gender!r}."
        )
    if clip.betas.shape[0] != model.beta_count:
        raise ValueError(
            "Motion beta width must exactly equal the selected SMPL-H shape basis."
        )
    return SMPLHDeviceClip(
        source_path=clip.source_path,
        full_pose_axis_angle=torch.as_tensor(
            np.array(clip.full_pose_axis_angle(), copy=True),
            dtype=torch.float32,
            device=target,
        ),
        betas=torch.as_tensor(
            np.array(clip.betas, copy=True), dtype=torch.float32, device=target
        ),
    )


def build_smplh_surface_asset(
    model: SMPLHModelData,
    clip: PLCSMotionClip,
    *,
    gaussian_count: int,
    seed: int,
) -> AvatarGaussianAsset:
    """Create one deterministic shaped zero-pose Gaussian shell."""
    if model.gender != clip.gender or clip.betas.shape[0] != model.beta_count:
        raise ValueError("Clip identity and selected SMPL-H model are incompatible.")
    shaped = model.template_vertices_m + np.einsum(
        "vdb,b->vd",
        model.shape_directions,
        clip.betas.astype(np.float64, copy=False),
    )
    return build_surface_gaussian_asset(
        shaped,
        faces=model.faces,
        vertex_joint_weights=model.vertex_joint_weights,
        gaussian_count=gaussian_count,
        seed=seed,
    )


def upload_gaussian_asset(
    asset: AvatarGaussianAsset,
    *,
    device: str | torch.device,
) -> SMPLHDeviceGaussianAsset:
    """Upload one canonical surface shell for all bounded source batches."""
    target = _cuda_device(device)
    return SMPLHDeviceGaussianAsset(
        means_m=torch.tensor(asset.means_m, dtype=torch.float32, device=target),
        quaternions_wxyz=torch.tensor(
            asset.quaternions_wxyz, dtype=torch.float32, device=target
        ),
        log_scales_m=torch.tensor(
            asset.log_scales_m, dtype=torch.float32, device=target
        ),
        opacity_logits=torch.tensor(
            asset.opacity_logits, dtype=torch.float32, device=target
        ),
        point_joint_weights=torch.tensor(
            asset.point_joint_weights, dtype=torch.float32, device=target
        ),
    )


def skin_gaussian_batch(
    model: SMPLHDeviceModel,
    clip: SMPLHDeviceClip,
    asset: SMPLHDeviceGaussianAsset,
    *,
    source_frame_indices: tuple[int, ...],
) -> SMPLHGaussianBatch:
    """Run Gaussian LBS/eigendecomposition for one bounded CUDA frame batch."""
    if not source_frame_indices:
        raise ValueError("Gaussian LBS requires at least one source frame.")
    if len(source_frame_indices) != len(set(source_frame_indices)):
        raise ValueError("Gaussian LBS source frame indices must be unique.")
    if any(index < 0 or index >= clip.frame_count for index in source_frame_indices):
        raise IndexError("Gaussian LBS source frame index is outside the full clip.")
    if (
        model.device != clip.full_pose_axis_angle.device
        or model.device != asset.means_m.device
    ):
        raise ValueError(
            "SMPL-H model, clip, and Gaussian asset must share one CUDA device."
        )
    indices = torch.tensor(source_frame_indices, dtype=torch.int64, device=model.device)
    pose = clip.full_pose_axis_angle.index_select(0, indices)
    betas = clip.betas.unsqueeze(0).expand(len(source_frame_indices), -1)
    with torch.inference_mode():
        joints, transforms = _joint_transforms(
            betas,
            pose,
            template=model.template_vertices_m,
            shapedirs=model.shape_directions,
            joint_regressor=model.joint_regressor,
            parents=model.parents,
        )
        blended = torch.einsum("nj,bjkl->bnkl", asset.point_joint_weights, transforms)
        homogeneous = torch.cat(
            (
                asset.means_m,
                torch.ones(
                    (asset.gaussian_count, 1),
                    dtype=torch.float32,
                    device=model.device,
                ),
            ),
            dim=1,
        )
        means = torch.einsum("bnkl,nl->bnk", blended, homogeneous)[..., :3]
        linear = blended[..., :3, :3]
        canonical_rotations = _quaternion_to_matrix(asset.quaternions_wxyz)
        canonical_variances = torch.exp(2.0 * asset.log_scales_m)
        canonical_covariances = (
            canonical_rotations
            @ torch.diag_embed(canonical_variances)
            @ canonical_rotations.transpose(-1, -2)
        )
        deformed_covariances = (
            linear @ canonical_covariances.unsqueeze(0) @ linear.transpose(-1, -2)
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(deformed_covariances)
        if not bool(torch.isfinite(eigenvalues).all()) or bool(
            (eigenvalues <= 0.0).any()
        ):
            raise ValueError("Gaussian LBS produced a non-positive covariance.")
        determinants = torch.linalg.det(eigenvectors)
        orientation_fix = torch.where(
            determinants < 0.0,
            eigenvectors.new_tensor(-1.0),
            eigenvectors.new_tensor(1.0),
        )
        eigenvectors = eigenvectors.clone()
        eigenvectors[..., :, -1] *= orientation_fix[..., None]
        quaternions = _matrix_to_quaternion(eigenvectors)
        log_scales = 0.5 * torch.log(eigenvalues)
    return SMPLHGaussianBatch(
        source_frame_indices=source_frame_indices,
        means_m=means,
        quaternions_wxyz=quaternions,
        log_scales_m=log_scales,
        joints_m=joints,
    )


def _joint_transforms(
    betas: Tensor,
    pose: Tensor,
    *,
    template: Tensor,
    shapedirs: Tensor,
    joint_regressor: Tensor,
    parents: Tensor,
) -> tuple[Tensor, Tensor]:
    shaped = template + blend_shapes(betas, shapedirs)
    canonical_joints = vertices2joints(joint_regressor, shaped)
    rotations = batch_rodrigues(pose.reshape(-1, 3)).reshape(
        pose.shape[0], _JOINT_COUNT, 3, 3
    )
    joints, transforms = batch_rigid_transform(
        rotations,
        canonical_joints,
        parents,
        dtype=betas.dtype,
    )
    return cast(Tensor, joints), cast(Tensor, transforms)


def _quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    q = F.normalize(quaternions, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternions.shape[:-1], 3, 3)


def _matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Convert proper rotations to normalized wxyz quaternions on-device."""
    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                (
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ),
                dim=-1,
            ),
            min=0.0,
        )
    )
    candidates = torch.stack(
        (
            torch.stack(
                (q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), dim=-1
            ),
            torch.stack(
                (m21 - m12, q_abs[..., 1].square(), m10 + m01, m02 + m20), dim=-1
            ),
            torch.stack(
                (m02 - m20, m10 + m01, q_abs[..., 2].square(), m12 + m21), dim=-1
            ),
            torch.stack(
                (m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3].square()), dim=-1
            ),
        ),
        dim=-2,
    )
    candidates = candidates / (2.0 * q_abs[..., :, None].clamp_min(0.1))
    selected = torch.argmax(q_abs, dim=-1)
    gather_index = selected[..., None, None].expand(*selected.shape, 1, 4)
    quaternion = candidates.gather(-2, gather_index).squeeze(-2)
    quaternion = F.normalize(quaternion, dim=-1)
    return torch.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)


def _cuda_device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda":
        raise ValueError(
            "PLCS production deformation requires an explicit CUDA device."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("PLCS production deformation requires available CUDA.")
    return device


__all__ = [
    "SMPLHDeviceClip",
    "SMPLHDeviceGaussianAsset",
    "SMPLHDeviceModel",
    "SMPLHGaussianBatch",
    "SMPLHModelData",
    "build_smplh_surface_asset",
    "load_smplh_model",
    "skin_gaussian_batch",
    "upload_gaussian_asset",
    "upload_motion_clip",
    "upload_smplh_model",
]
