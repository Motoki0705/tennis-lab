"""DINOv3 patch-token annotation contract for issue #634 clips.

Tokens are precomputed (see ``src/tasks/slcs/scripts/precompute_dino_tokens.py``)
and stored per clip under ``annotations/dino_v3/``::

    annotations/dino_v3/
        annotation.json      # completion marker: spec + per-camera inventory
        <camera_id>.npz      # tokens (T_d, S, C) float16, frame_idx (T_d,) int64

``frame_idx`` records exactly which clip frames were encoded (every
``frame_stride``-th frame plus the final frame). Consumers must align tokens by
these explicit indices; interpolating between samples is deliberately not part
of this contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.slcs.data.contract import (
    ANNOTATION_MARKER_NAME,
    ANNOTATIONS_DIR_NAME,
    ClipManifest,
    DatasetContractError,
    IncompleteAnnotationError,
    UnsupportedFormatVersionError,
)
from src.utils.io import ensure_dir, load_json, save_json_atomic, utc_now_iso

DINO_DIR_NAME = "dino_v3"
DINO_ANNOTATION_KIND = "dino_v3_patch_tokens"
DINO_ANNOTATION_VERSION = 1


@dataclass(frozen=True)
class DinoTokenSpec:
    """Backbone/sampling parameters a token annotation was generated with."""

    backbone: str
    patch_size: int
    image_height: int
    image_width: int
    embed_dim: int
    frame_stride: int

    def __post_init__(self) -> None:
        if not self.backbone:
            raise DatasetContractError("backbone must not be empty.")
        if self.patch_size <= 0:
            raise DatasetContractError(
                f"patch_size must be positive, got {self.patch_size}."
            )
        if self.image_height <= 0 or self.image_width <= 0:
            raise DatasetContractError(
                "image_height and image_width must be positive, got "
                f"{(self.image_height, self.image_width)}."
            )
        if self.image_height % self.patch_size or self.image_width % self.patch_size:
            raise DatasetContractError(
                f"image size {(self.image_height, self.image_width)} must be divisible "
                f"by patch_size={self.patch_size}."
            )
        if self.embed_dim <= 0:
            raise DatasetContractError(
                f"embed_dim must be positive, got {self.embed_dim}."
            )
        if self.frame_stride <= 0:
            raise DatasetContractError(
                f"frame_stride must be positive, got {self.frame_stride}."
            )

    @property
    def grid_h(self) -> int:
        return self.image_height // self.patch_size

    @property
    def grid_w(self) -> int:
        return self.image_width // self.patch_size

    @property
    def num_tokens(self) -> int:
        return self.grid_h * self.grid_w

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DinoTokenSpec:
        expected = {
            "backbone": str,
            "patch_size": int,
            "image_height": int,
            "image_width": int,
            "embed_dim": int,
            "frame_stride": int,
        }
        unknown = sorted(set(data) - set(expected))
        missing = sorted(set(expected) - set(data))
        if unknown or missing:
            raise DatasetContractError(
                "dino token spec must use the exact current key set; "
                f"missing={missing}, unknown={unknown}."
            )
        wrong_types = {
            key: type(data[key]).__name__
            for key, expected_type in expected.items()
            if type(data[key]) is not expected_type
        }
        if wrong_types:
            raise DatasetContractError(
                f"dino token spec contains wrong exact types: {wrong_types}."
            )
        return cls(
            backbone=data["backbone"],
            patch_size=data["patch_size"],
            image_height=data["image_height"],
            image_width=data["image_width"],
            embed_dim=data["embed_dim"],
            frame_stride=data["frame_stride"],
        )


def sample_frame_indices(num_frames: int, frame_stride: int) -> NDArray[np.int64]:
    """Frame indices to encode: every ``frame_stride``-th frame plus the last."""
    if num_frames <= 0:
        raise DatasetContractError(f"num_frames must be positive, got {num_frames}.")
    if frame_stride <= 0:
        raise DatasetContractError(
            f"frame_stride must be positive, got {frame_stride}."
        )
    indices = list(range(0, num_frames, frame_stride))
    if indices[-1] != num_frames - 1:
        indices.append(num_frames - 1)
    return np.asarray(indices, dtype=np.int64)


def dino_dir(clip_dir: Path) -> Path:
    return clip_dir / ANNOTATIONS_DIR_NAME / DINO_DIR_NAME


def has_dino_tokens(clip_dir: Path) -> bool:
    """Whether a *complete* DINOv3 token annotation exists for the clip."""
    return (dino_dir(clip_dir) / ANNOTATION_MARKER_NAME).is_file()


def write_dino_tokens(
    manifest: ClipManifest,
    tokens_by_camera: dict[str, tuple[NDArray[np.float16], NDArray[np.int64]]],
    spec: DinoTokenSpec,
    *,
    generator: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write per-camera token archives and the completion marker (marker last)."""
    out_dir = dino_dir(manifest.clip_dir)
    marker_path = out_dir / ANNOTATION_MARKER_NAME
    if marker_path.exists() and not overwrite:
        raise DatasetContractError(
            f"{manifest.clip_id}: completed dino_v3 annotation already exists "
            f"({marker_path}). Pass overwrite=True to regenerate."
        )
    if set(tokens_by_camera) != set(manifest.camera_ids):
        raise DatasetContractError(
            f"{manifest.clip_id}: tokens cover cameras {sorted(tokens_by_camera)}, "
            f"manifest declares {sorted(manifest.camera_ids)}."
        )
    ensure_dir(out_dir)

    cameras: dict[str, dict[str, Any]] = {}
    for camera_id, (tokens, frame_idx) in tokens_by_camera.items():
        tokens = np.asarray(tokens, dtype=np.float16)
        frame_idx = np.asarray(frame_idx, dtype=np.int64)
        _validate_token_arrays(
            tokens,
            frame_idx,
            spec=spec,
            num_frames=manifest.num_frames,
            context=f"{manifest.clip_id}/{camera_id}",
        )
        np.savez_compressed(
            out_dir / f"{camera_id}.npz", tokens=tokens, frame_idx=frame_idx
        )
        cameras[camera_id] = {
            "file": f"{camera_id}.npz",
            "num_samples": int(tokens.shape[0]),
        }

    marker: dict[str, Any] = {
        "format_version": DINO_ANNOTATION_VERSION,
        "kind": DINO_ANNOTATION_KIND,
        "created_at": utc_now_iso(),
        "generator": generator or {},
        "spec": asdict(spec),
        "cameras": cameras,
        "input_manifest_digest": manifest.digest(),
    }
    save_json_atomic(marker, marker_path)
    return marker_path


def _validate_token_arrays(
    tokens: NDArray[np.float16],
    frame_idx: NDArray[np.int64],
    *,
    spec: DinoTokenSpec,
    num_frames: int,
    context: str,
) -> None:
    if tokens.ndim != 3:
        raise DatasetContractError(
            f"{context}: tokens must be (T_d, S, C), got shape {tokens.shape}."
        )
    if tokens.shape[1] != spec.num_tokens or tokens.shape[2] != spec.embed_dim:
        raise DatasetContractError(
            f"{context}: tokens shape {tokens.shape} does not match spec "
            f"(S={spec.num_tokens}, C={spec.embed_dim})."
        )
    if frame_idx.ndim != 1 or frame_idx.shape[0] != tokens.shape[0]:
        raise DatasetContractError(
            f"{context}: frame_idx shape {frame_idx.shape} must be (T_d,) with "
            f"T_d={tokens.shape[0]}."
        )
    if frame_idx.shape[0] == 0:
        raise DatasetContractError(f"{context}: token annotation contains no samples.")
    if np.any(np.diff(frame_idx) <= 0):
        raise DatasetContractError(f"{context}: frame_idx must be strictly increasing.")
    if frame_idx[0] < 0 or int(frame_idx[-1]) >= num_frames:
        raise DatasetContractError(
            f"{context}: frame_idx range [{int(frame_idx[0])}, {int(frame_idx[-1])}] "
            f"outside clip frames [0, {num_frames})."
        )
    if not np.isfinite(tokens.astype(np.float32)).all():
        raise DatasetContractError(f"{context}: tokens contain non-finite values.")


def load_dino_spec(clip_dir: Path) -> DinoTokenSpec:
    """Load and validate the spec block from a completed dino annotation."""
    marker_path = dino_dir(clip_dir) / ANNOTATION_MARKER_NAME
    if not marker_path.is_file():
        raise IncompleteAnnotationError(
            f"dino_v3 annotation has no completion marker ({marker_path})."
        )
    marker = load_json(marker_path)
    if not isinstance(marker, dict):
        raise DatasetContractError(f"{marker_path} must contain a JSON object.")
    if marker.get("format_version") != DINO_ANNOTATION_VERSION:
        raise UnsupportedFormatVersionError(
            f"{marker_path} declares format_version={marker.get('format_version')!r}; "
            f"this reader supports {DINO_ANNOTATION_VERSION}."
        )
    if marker.get("kind") != DINO_ANNOTATION_KIND:
        raise DatasetContractError(
            f"{marker_path}: kind={marker.get('kind')!r}, expected {DINO_ANNOTATION_KIND!r}."
        )
    spec_raw = marker.get("spec")
    if not isinstance(spec_raw, dict):
        raise DatasetContractError(
            f"{marker_path}: marker must record a 'spec' mapping."
        )
    return DinoTokenSpec.from_dict(spec_raw)


def load_dino_tokens(
    manifest: ClipManifest,
    camera_id: str,
    *,
    expected_spec: DinoTokenSpec | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.int64], DinoTokenSpec]:
    """Load validated tokens for one camera.

    Returns:
        ``(tokens (T_d, S, C) float32, frame_idx (T_d,) int64, spec)``.
    """
    spec = load_dino_spec(manifest.clip_dir)
    if expected_spec is not None and spec != expected_spec:
        raise DatasetContractError(
            f"{manifest.clip_id}: dino token spec {asdict(spec)} does not match the "
            f"expected spec {asdict(expected_spec)}."
        )
    manifest.camera_index(camera_id)  # validates the camera id
    npz_path = dino_dir(manifest.clip_dir) / f"{camera_id}.npz"
    if not npz_path.is_file():
        raise DatasetContractError(
            f"{manifest.clip_id}: dino token archive missing for camera {camera_id!r}: {npz_path}"
        )
    data = np.load(npz_path, allow_pickle=False)
    if "tokens" not in data or "frame_idx" not in data:
        raise DatasetContractError(
            f"{npz_path}: archive must contain 'tokens' and 'frame_idx' arrays."
        )
    tokens = np.asarray(data["tokens"], dtype=np.float16)
    frame_idx = np.asarray(data["frame_idx"], dtype=np.int64)
    _validate_token_arrays(
        tokens,
        frame_idx,
        spec=spec,
        num_frames=manifest.num_frames,
        context=f"{manifest.clip_id}/{camera_id}",
    )
    return tokens.astype(np.float32), frame_idx, spec


__all__ = [
    "DINO_ANNOTATION_KIND",
    "DINO_ANNOTATION_VERSION",
    "DINO_DIR_NAME",
    "DinoTokenSpec",
    "dino_dir",
    "has_dino_tokens",
    "load_dino_spec",
    "load_dino_tokens",
    "sample_frame_indices",
    "write_dino_tokens",
]
