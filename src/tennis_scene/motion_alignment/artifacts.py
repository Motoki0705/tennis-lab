"""Court-frame world motion exported from raw GVHMR global SMPL predictions.

The GVHMR extractor stores the *raw* network outputs of the gravity-aligned
world: SMPL-X body pose, betas, ``global_orient`` and ``transl`` in a Y-up
meter frame. Those parameters are not scene coordinates yet: ``transl`` is the
regressed translation offset rather than the pelvis, and the body is still
Y-up.

This module is the CPU adapter between that raw record and the aligner. It
reconstructs the SMPL vertices, regresses the SMPL-24 / COCO-17 joints the
renderer uses, converts points into the court Z-up frame, and exposes the raw
parameters that a similarity transform must rewrite.

The exported contract for a rigid court map ``C`` (rotation), ``s`` (body
scale) and ``b`` (translation) is::

    world_vertices = s * SMPL(body_pose, betas, global_orient_court,
                              transl=0).vertices + transl_court_scaled

with ``global_orient_court = C @ R(global_orient)`` and::

    transl_court_scaled = s * C @ (rest_pelvis + transl) + b
                         - s * rest_pelvis

``rest_pelvis`` is the body model's true root pivot, so the map keeps the whole
geometry (mesh and every joint) identical under scaling and rotation. The root
*position* reported here is the renderer's joint-0 anchor, which is a different
quantity from that pivot.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray

from src.submodules.configuration import BundledModelAssetPaths
from src.submodules.vendor.gvhmr.body_model import resolve_smplx_model_file
from src.utils.geometry.matrices import SMPL_Y_UP_TO_COURT_Z_UP, rotation_matrix_z
from src.utils.geometry.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)

Float32Array: TypeAlias = NDArray[np.float32]
Float64Array: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]

RAW_SCHEMA_VERSION = "plcs_gvhmr_global_smpl_v2"
ALIGNED_COORDINATE_SYSTEM = "gvhmr_aligned_y_up_m"

NUM_BETAS = 10
BODY_POSE_WIDTH = 63
NUM_JOINTS_SMPL = 24
NUM_KEYPOINTS_COCO = 17

_MOTION_VERTEX_COUNT = 6890
_SMPLX_VERTEX_COUNT = 10475

_REQUIRED_RAW_ARRAYS = (
    "body_pose",
    "betas",
    "global_orient",
    "transl",
    "K_fullimg",
    "keypoints_2d_px",
    "boxes_xys_px",
    "observed_mask",
)
_REQUIRED_RAW_KEYS = frozenset(_REQUIRED_RAW_ARRAYS) | {"metadata_json"}


class TransformedSmplParameters(TypedDict):
    """SMPL parameters rewritten for one court-frame similarity transform.

    The values are meant to be consumed as::

        vertices = body_scale * SMPL(body_pose, betas, global_orient,
                                     transl=0).vertices + transl

    i.e. ``transl`` is applied after ``body_scale``, and ``body_scale`` must be
    applied to every joint as well as to every mesh vertex.
    """

    body_scale: float
    global_orient: Float32Array
    transl: Float32Array
    body_pose: Float32Array
    betas: Float32Array


@dataclass(frozen=True, slots=True)
class ConfidenceDiagnostics:
    """Explicit 2D confidence audit; values are never silently clipped."""

    total_values: int
    below_zero: int
    above_one: int
    nonfinite: int

    @property
    def needs_clipping(self) -> bool:
        """Whether the upstream extractor produced out-of-range confidences."""
        return bool(self.below_zero or self.above_one or self.nonfinite)


@dataclass(frozen=True, slots=True)
class GvhmrWorldMotion:
    """One GVHMR track exported as court-frame world motion.

    Point arrays are float64 in the court frame (XY is the court plane, +Z is
    up). ``raw_*`` arrays are the untouched Y-up network outputs, and
    ``rest_pelvis`` is the Y-up body-model root pivot consumed by
    :func:`transform_smpl_parameters`.

    Attributes:
        root_position: Renderer joint-0 anchor ``(T, 3)``, court frame.
        joints_smpl: SMPL-24 joints ``(T, 24, 3)``, court frame.
        joints_coco: COCO-17 joints ``(T, 17, 3)``, court frame.
        rotation: Court-frame global rotation ``(T, 3, 3)``; heading is
            ``atan2(rotation[t, 1, 0], rotation[t, 0, 0])``.
        rest_pelvis: Body-model root pivot ``(T, 3)``, Y-up meters.
        raw_body_pose: Axis-angle body pose ``(T, 63)``, Y-up.
        raw_betas: Shape coefficients ``(T, 10)``.
        raw_global_orient: Axis-angle global orientation ``(T, 3)``, Y-up.
        raw_transl: Regressed translation offset ``(T, 3)``, Y-up meters. This
            is *not* the pelvis.
        keypoints_2d_px: Raw COCO-17 keypoints ``(T, 17, 3)`` as ``(x, y,
            confidence)`` pixels, unclipped.
        K_fullimg: Intrinsics used by the extractor ``(T, 3, 3)``.
        boxes_xys_px: Source person boxes ``(T, 3)``.
        confidence: Raw 2D confidence ``(T, 17)``, never clipped.
        confidence_diagnostics: Counts of out-of-range confidences.
        observed: Per-frame observed mask ``(T,)``.
        metadata: Parsed ``metadata_json``.
        fps: Native frame rate recorded by the extractor.
    """

    root_position: Float64Array
    joints_smpl: Float64Array
    joints_coco: Float64Array
    rotation: Float64Array
    rest_pelvis: Float64Array
    raw_body_pose: Float32Array
    raw_betas: Float32Array
    raw_global_orient: Float32Array
    raw_transl: Float32Array
    keypoints_2d_px: Float32Array
    K_fullimg: Float32Array
    boxes_xys_px: Float32Array
    confidence: Float64Array
    confidence_diagnostics: ConfidenceDiagnostics
    observed: BoolArray
    metadata: Mapping[str, object]
    fps: float

    def __post_init__(self) -> None:
        if self.observed.ndim != 1:
            raise ValueError(
                f"observed must have shape (T,), got {self.observed.shape}."
            )
        frames = int(self.observed.shape[0])
        expected: dict[str, tuple[int, ...]] = {
            "root_position": (frames, 3),
            "joints_smpl": (frames, NUM_JOINTS_SMPL, 3),
            "joints_coco": (frames, NUM_KEYPOINTS_COCO, 3),
            "rotation": (frames, 3, 3),
            "rest_pelvis": (frames, 3),
            "raw_body_pose": (frames, BODY_POSE_WIDTH),
            "raw_betas": (frames, NUM_BETAS),
            "raw_global_orient": (frames, 3),
            "raw_transl": (frames, 3),
            "keypoints_2d_px": (frames, NUM_KEYPOINTS_COCO, 3),
            "K_fullimg": (frames, 3, 3),
            "boxes_xys_px": (frames, 3),
            "confidence": (frames, NUM_KEYPOINTS_COCO),
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise ValueError(
                    f"GvhmrWorldMotion.{name} must have shape {shape}, "
                    f"got {actual}."
                )
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError(f"GvhmrWorldMotion.fps must be positive, got {self.fps}.")


def confidence_diagnostics(confidence: NDArray[Any]) -> ConfidenceDiagnostics:
    """Count out-of-range 2D confidences instead of clipping them away."""
    values = np.asarray(confidence, dtype=np.float64)
    return ConfidenceDiagnostics(
        total_values=int(values.size),
        below_zero=int(np.count_nonzero(values < 0.0)),
        above_one=int(np.count_nonzero(values > 1.0)),
        nonfinite=int(np.count_nonzero(~np.isfinite(values))),
    )


def transform_smpl_parameters(
    *,
    body_pose: NDArray[Any],
    betas: NDArray[Any],
    global_orient: NDArray[Any],
    transl: NDArray[Any],
    rest_pelvis: NDArray[Any],
    scale: float,
    yaw: float,
    translation: NDArray[Any],
) -> TransformedSmplParameters:
    """Rewrite raw SMPL parameters for ``x -> scale * (Rz(yaw) @ A) @ x + b``.

    ``A`` is :data:`SMPL_Y_UP_TO_COURT_Z_UP`, so the composed rotation maps the
    Y-up body into the court frame and then rotates about court +Z by ``yaw``.

    Args:
        body_pose: Axis-angle body pose ``(T, 63)``, unchanged in the result.
        betas: Shape coefficients ``(T, 10)``, unchanged in the result.
        global_orient: Axis-angle global orientation ``(T, 3)``, Y-up.
        transl: Raw regressed translation ``(T, 3)``, Y-up meters.
        rest_pelvis: Body-model root pivot ``(T, 3)``, Y-up meters, i.e.
            ``J_regressor[0] @ (v_template + shapedirs[:, :, :10] @ betas)``.
        scale: Body scale, applied to the mesh and to every joint.
        yaw: Rotation about court +Z in radians.
        translation: Court-frame offset ``(3,)``, meters.

    Returns:
        Explicit parameters with the semantics documented on
        :class:`TransformedSmplParameters`. Returned arrays are float32 because
        the body model runs in float32; the algebra is evaluated in float64.
    """
    body_pose_f32 = _as_float32(body_pose, name="body_pose", width=BODY_POSE_WIDTH)
    betas_f32 = _as_float32(betas, name="betas", width=NUM_BETAS)
    orient_f32 = _as_float32(global_orient, name="global_orient", width=3)
    transl_f64 = _as_float64(transl, name="transl", width=3)
    pivot_f64 = _as_float64(rest_pelvis, name="rest_pelvis", width=3)
    translation_f64 = _as_float64(
        np.atleast_2d(translation), name="translation", width=3
    )
    if translation_f64.shape != (1, 3):
        raise ValueError(
            f"translation must have shape (3,), got {np.shape(translation)}."
        )
    frames = body_pose_f32.shape[0]
    for name, value in (
        ("betas", betas_f32),
        ("global_orient", orient_f32),
        ("transl", transl_f64),
        ("rest_pelvis", pivot_f64),
    ):
        if value.shape[0] != frames:
            raise ValueError(
                f"{name} must have the same frame count as body_pose "
                f"({frames}), got {value.shape[0]}."
            )
    scale_value = float(scale)
    yaw_value = float(yaw)
    if not np.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale!r}.")
    if not np.isfinite(yaw_value):
        raise ValueError(f"yaw must be finite, got {yaw!r}.")

    basis = np.asarray(SMPL_Y_UP_TO_COURT_Z_UP, dtype=np.float64)
    yaw_rotation = np.asarray(
        rotation_matrix_z(np.asarray(yaw_value, dtype=np.float32)), dtype=np.float64
    )
    court_map = yaw_rotation @ basis

    orientation = axis_angle_to_matrix(
        torch.from_numpy(orient_f32.astype(np.float64))
    ).numpy()
    rotated_orientation = np.einsum("ij,tjk->tik", court_map, orientation)
    rotated_axis_angle = (
        matrix_to_axis_angle(torch.from_numpy(rotated_orientation))
        .numpy()
        .astype(np.float32)
    )

    scaled_offset = (
        scale_value * ((pivot_f64 + transl_f64) @ court_map.T)
        + translation_f64[0]
        - scale_value * pivot_f64
    )
    return TransformedSmplParameters(
        body_scale=scale_value,
        global_orient=rotated_axis_angle,
        transl=scaled_offset.astype(np.float32),
        body_pose=body_pose_f32.copy(),
        betas=betas_f32.copy(),
    )


def load_world_motion(
    raw_path: str | Path,
    *,
    body_models_dir: str | Path,
    bundled_assets: BundledModelAssetPaths,
    batch_size: int = 32,
) -> GvhmrWorldMotion:
    """Export one raw ``*.gvhmr.npz`` record as court-frame world motion.

    Reconstruction runs on the CPU in batches of ``batch_size`` frames. Every
    contract violation (schema, shapes, frame count, missing keys, non-finite
    values) raises instead of falling back to a default.

    Args:
        raw_path: Raw record written by the PLCS GVHMR extractor.
        body_models_dir: Absolute licensed body-model directory containing
            ``smplx/SMPLX_NEUTRAL.npz``.
        bundled_assets: Repository-owned GVHMR assets.
        batch_size: Frames per SMPL-X forward pass; must be positive.

    Returns:
        The exported motion, including the raw parameters needed to re-export a
        similarity-transformed track.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}.")
    if not isinstance(bundled_assets, BundledModelAssetPaths):
        raise TypeError("bundled_assets must be BundledModelAssetPaths.")

    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw GVHMR motion record is missing: {path}")

    with np.load(path, allow_pickle=False) as record:
        missing = sorted(_REQUIRED_RAW_KEYS - set(record.files))
        if missing:
            raise ValueError(
                f"Raw GVHMR motion record {path} is missing keys: {missing}."
            )
        metadata = _parse_metadata(record["metadata_json"])
        raw: dict[str, Float32Array] = {
            "body_pose": _as_float32(
                record["body_pose"], name="body_pose", width=BODY_POSE_WIDTH
            ),
            "betas": _as_float32(record["betas"], name="betas", width=NUM_BETAS),
            "global_orient": _as_float32(
                record["global_orient"], name="global_orient", width=3
            ),
            "transl": _as_float32(record["transl"], name="transl", width=3),
        }
        keypoints = _as_float32(
            record["keypoints_2d_px"],
            name="keypoints_2d_px",
            width=3,
            channels=NUM_KEYPOINTS_COCO,
        )
        K_fullimg = _as_float32(record["K_fullimg"], name="K_fullimg", width=3)
        boxes = _as_float32(record["boxes_xys_px"], name="boxes_xys_px", width=3)
        observed = _as_bool(record["observed_mask"], name="observed_mask")

    frames = raw["body_pose"].shape[0]
    if observed.shape != (frames,):
        raise ValueError(
            f"observed_mask must have shape ({frames},), got {observed.shape}."
        )
    if keypoints.shape != (frames, NUM_KEYPOINTS_COCO, 3):
        raise ValueError(
            "keypoints_2d_px must have shape "
            f"({frames}, {NUM_KEYPOINTS_COCO}, 3), got {keypoints.shape}."
        )
    if K_fullimg.shape != (frames, 3, 3):
        raise ValueError(
            f"K_fullimg must have shape ({frames}, 3, 3), got {K_fullimg.shape}."
        )
    if boxes.shape != (frames, 3):
        raise ValueError(
            f"boxes_xys_px must have shape ({frames}, 3), got {boxes.shape}."
        )

    fps = _metadata_fps(metadata, frames=frames)
    rest_pelvis = _rest_pelvis(body_models_dir, raw["betas"])
    joints_smpl, joints_coco = _reconstruct_joints(
        raw=raw,
        body_models_dir=body_models_dir,
        bundled_assets=bundled_assets,
        batch_size=batch_size,
    )

    basis = np.asarray(SMPL_Y_UP_TO_COURT_Z_UP, dtype=np.float64)
    orientation = axis_angle_to_matrix(
        torch.from_numpy(raw["global_orient"].astype(np.float64))
    ).numpy()
    rotation = np.einsum("ij,tjk->tik", basis, orientation)
    joints_smpl_court = joints_smpl @ basis.T
    confidence = keypoints[:, :, 2].astype(np.float64)

    return GvhmrWorldMotion(
        root_position=joints_smpl_court[:, 0].copy(),
        joints_smpl=joints_smpl_court,
        joints_coco=joints_coco @ basis.T,
        rotation=rotation,
        rest_pelvis=rest_pelvis,
        raw_body_pose=raw["body_pose"],
        raw_betas=raw["betas"],
        raw_global_orient=raw["global_orient"],
        raw_transl=raw["transl"],
        keypoints_2d_px=keypoints,
        K_fullimg=K_fullimg,
        boxes_xys_px=boxes,
        confidence=confidence,
        confidence_diagnostics=confidence_diagnostics(confidence),
        observed=observed,
        metadata=metadata,
        fps=fps,
    )


def _parse_metadata(value: NDArray[Any]) -> dict[str, object]:
    """Decode ``metadata_json`` and enforce the extractor's schema markers."""
    if value.shape != ():
        raise ValueError(
            f"metadata_json must be a scalar string, got shape {value.shape}."
        )
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"metadata_json is not valid JSON: {error}.") from error
    if not isinstance(parsed, dict):
        raise ValueError(
            f"metadata_json must decode to an object, got {type(parsed).__name__}."
        )
    metadata: dict[str, object] = {str(key): item for key, item in parsed.items()}
    schema_version = metadata.get("schema_version")
    if schema_version != RAW_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported raw GVHMR schema: expected {RAW_SCHEMA_VERSION!r}, "
            f"got {schema_version!r}."
        )
    coordinate_system = metadata.get("coordinate_system")
    if coordinate_system != ALIGNED_COORDINATE_SYSTEM:
        raise ValueError(
            "Raw GVHMR record must use coordinate_system "
            f"{ALIGNED_COORDINATE_SYSTEM!r}, got {coordinate_system!r}."
        )
    return metadata


def _metadata_fps(metadata: Mapping[str, object], *, frames: int) -> float:
    """Cross-check the recorded frame count and return the native frame rate."""
    frame_count = metadata.get("frame_count")
    if type(frame_count) is not int or frame_count != frames:
        raise ValueError(
            "metadata frame_count does not match the stored arrays: "
            f"metadata={frame_count!r}, arrays={frames}."
        )
    native_fps = metadata.get("native_fps")
    if isinstance(native_fps, bool) or not isinstance(native_fps, (int, float)):
        raise ValueError(f"metadata native_fps must be numeric, got {native_fps!r}.")
    fps = float(native_fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"metadata native_fps must be positive, got {native_fps!r}.")
    return fps


def _as_float32(
    value: NDArray[Any],
    *,
    name: str,
    width: int | None,
    channels: int | None = None,
) -> Float32Array:
    """Validate one raw array and return a finite float32 copy."""
    array = np.asarray(value)
    if array.dtype.kind not in ("f", "i", "u"):
        raise ValueError(f"{name} must be numeric, got dtype {array.dtype}.")
    if array.ndim < 2:
        raise ValueError(
            f"{name} must be a batched array with a frame axis, got {array.shape}."
        )
    if width is not None and array.shape[-1] != width:
        raise ValueError(
            f"{name} must have a trailing width of {width}, got {array.shape}."
        )
    if channels is not None and array.shape[-2] != channels:
        raise ValueError(f"{name} must have {channels} channels, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array.astype(np.float32, copy=True)


def _as_bool(value: NDArray[Any], *, name: str) -> BoolArray:
    """Require a genuine boolean mask.

    ``np.asarray(..., dtype=bool)`` would quietly map any non-zero number
    (including ``NaN`` and ``2.0``) to ``True``, so a float or integer mask is
    rejected rather than reinterpreted.
    """
    array = np.asarray(value)
    if array.dtype.kind != "b":
        raise ValueError(
            f"{name} must be a boolean array, got dtype {array.dtype}; "
            "cast the upstream mask explicitly before writing the record."
        )
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (T,), got {array.shape}.")
    return array.astype(np.bool_, copy=True)


def _as_float64(
    value: NDArray[Any],
    *,
    name: str,
    width: int | None,
) -> Float64Array:
    """Validate one parameter array and return a finite float64 copy."""
    array = np.asarray(value)
    if array.dtype.kind not in ("f", "i", "u"):
        raise ValueError(f"{name} must be numeric, got dtype {array.dtype}.")
    if array.ndim < 2:
        raise ValueError(
            f"{name} must be a batched array with a frame axis, got {array.shape}."
        )
    if width is not None and array.shape[-1] != width:
        raise ValueError(
            f"{name} must have a trailing width of {width}, got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array.astype(np.float64, copy=True)


def _rest_pelvis(
    body_models_dir: str | Path,
    betas: Float32Array,
) -> Float64Array:
    """Return the Y-up root pivot ``(T, 3)`` of the licensed SMPL-X model."""
    model_path = resolve_smplx_model_file(Path(body_models_dir) / "smplx", "neutral")
    with np.load(model_path, allow_pickle=False) as data:
        v_template = np.asarray(data["v_template"], dtype=np.float64)
        shapedirs = np.asarray(data["shapedirs"], dtype=np.float64)
        j_regressor = np.asarray(data["J_regressor"], dtype=np.float64)
    if v_template.shape != (_SMPLX_VERTEX_COUNT, 3):
        raise ValueError(
            f"SMPL-X v_template must have shape ({_SMPLX_VERTEX_COUNT}, 3), "
            f"got {v_template.shape}."
        )
    if shapedirs.ndim != 3 or shapedirs.shape[:2] != (_SMPLX_VERTEX_COUNT, 3):
        raise ValueError(
            "SMPL-X shapedirs must have shape "
            f"({_SMPLX_VERTEX_COUNT}, 3, D), got {shapedirs.shape}."
        )
    if shapedirs.shape[-1] < NUM_BETAS:
        raise ValueError(
            f"SMPL-X shapedirs must expose at least {NUM_BETAS} shape "
            f"components, got {shapedirs.shape[-1]}."
        )
    if j_regressor.ndim != 2 or j_regressor.shape[1] != _SMPLX_VERTEX_COUNT:
        raise ValueError(
            "SMPL-X J_regressor must have shape "
            f"(J, {_SMPLX_VERTEX_COUNT}), got {j_regressor.shape}."
        )
    pelvis_regressor = j_regressor[0]
    base = np.einsum("v,vc->c", pelvis_regressor, v_template)
    blend = np.einsum("v,vcd->cd", pelvis_regressor, shapedirs[:, :, :NUM_BETAS])
    pivot: Float64Array = base[None, :] + betas.astype(np.float64) @ blend.T
    return pivot


def _reconstruct_joints(
    *,
    raw: Mapping[str, Float32Array],
    body_models_dir: str | Path,
    bundled_assets: BundledModelAssetPaths,
    batch_size: int,
) -> tuple[Float64Array, Float64Array]:
    """Reconstruct world vertices in CPU batches and regress both joint sets."""
    smpl_regressor = _load_joint_regressor(
        bundled_assets.smpl_neutral_joint_regressor, rows=NUM_JOINTS_SMPL
    )
    coco_regressor = _load_joint_regressor(
        bundled_assets.smpl_coco17_regressor, rows=NUM_KEYPOINTS_COCO
    )
    reconstructor = _build_reconstructor(
        body_models_dir=body_models_dir, bundled_assets=bundled_assets
    )

    frames = raw["body_pose"].shape[0]
    joints_smpl = np.empty((frames, NUM_JOINTS_SMPL, 3), dtype=np.float64)
    joints_coco = np.empty((frames, NUM_KEYPOINTS_COCO, 3), dtype=np.float64)
    for start in range(0, frames, batch_size):
        stop = min(start + batch_size, frames)
        parameters = {name: torch.from_numpy(block[start:stop]) for name, block in raw.items()}
        vertices = reconstructor.reconstruct(parameters).numpy().astype(np.float64)
        joints_smpl[start:stop] = np.einsum(
            "jv,fvc->fjc", smpl_regressor, vertices
        )
        joints_coco[start:stop] = np.einsum(
            "jv,fvc->fjc", coco_regressor, vertices
        )
    return joints_smpl, joints_coco


def _load_joint_regressor(path: Path, *, rows: int) -> Float64Array:
    """Load one dense SMPL joint regressor and validate its shape."""
    if not path.is_file():
        raise FileNotFoundError(f"Joint regressor asset is missing: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, torch.Tensor):
        raise TypeError(f"Joint regressor asset must contain a tensor: {path}")
    if loaded.shape != (rows, _MOTION_VERTEX_COUNT):
        raise ValueError(
            f"Joint regressor {path} must have shape "
            f"({rows}, {_MOTION_VERTEX_COUNT}), got {tuple(loaded.shape)}."
        )
    return loaded.to(dtype=torch.float64).numpy()


def _build_reconstructor(
    *,
    body_models_dir: str | Path,
    bundled_assets: BundledModelAssetPaths,
) -> Any:
    """Build the CPU SMPL vertex reconstructor.

    Kept as a module-level seam so tests can exercise the adapter without the
    licensed body model.
    """
    from src.submodules.models.gvhmr.mesh_recovery import SmplVertexReconstructor

    return SmplVertexReconstructor(
        body_models_dir, device="cpu", bundled_assets=bundled_assets
    )


__all__ = [
    "ALIGNED_COORDINATE_SYSTEM",
    "BODY_POSE_WIDTH",
    "NUM_BETAS",
    "NUM_JOINTS_SMPL",
    "NUM_KEYPOINTS_COCO",
    "RAW_SCHEMA_VERSION",
    "ConfidenceDiagnostics",
    "GvhmrWorldMotion",
    "TransformedSmplParameters",
    "confidence_diagnostics",
    "load_world_motion",
    "transform_smpl_parameters",
]
