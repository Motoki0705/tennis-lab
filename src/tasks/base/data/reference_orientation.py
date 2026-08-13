"""Reference-camera orientation and court-space reflection contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor

from src.utils.schema.court import HALF_LENGTH

REFERENCE_BASELINE_AMBIGUITY_MARGIN_M = 1.0


def deterministic_sample_rng(seed: int, sample_key: str) -> np.random.Generator:
    """Create a stable per-sample RNG independent of dataset access order."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if not sample_key or sample_key != sample_key.strip():
        raise ValueError("sample_key must be non-empty and trimmed.")
    digest = hashlib.blake2b(
        f"{seed}:{sample_key}".encode(), digest_size=8
    ).digest()
    return np.random.default_rng(int.from_bytes(digest, byteorder="little"))


def camera_centers_from_scene_payload(
    payload: Mapping[str, object], camera_indices: Sequence[int]
) -> Tensor:
    """Read declared court-coordinate camera centres from scene scalars."""
    centers: list[tuple[float, float, float]] = []
    for camera_index in camera_indices:
        raw = payload.get(f"cam_{camera_index}_params")
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"scene camera {camera_index} has no declared params mapping."
            )
        center = raw.get("C")
        if (
            not isinstance(center, Sequence)
            or isinstance(center, (str, bytes))
            or len(center) != 3
            or any(type(value) not in {int, float} for value in center)
        ):
            raise ValueError(
                f"scene camera {camera_index} params.C must contain three numbers."
            )
        centers.append(
            (float(center[0]), float(center[1]), float(center[2]))
        )
    result = torch.tensor(centers, dtype=torch.float32)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("scene camera centres must be finite.")
    return result


def orientation_signs_from_camera_centers(
    camera_centers: Tensor,
    *,
    ambiguity_margin_m: float = REFERENCE_BASELINE_AMBIGUITY_MARGIN_M,
) -> tuple[Tensor, Tensor]:
    """Return per-view Y reflection signs and unambiguous candidate mask."""
    if camera_centers.ndim != 2 or camera_centers.shape[-1] != 3:
        raise ValueError("camera_centers must have shape (V,3).")
    if not camera_centers.is_floating_point() or not bool(
        torch.isfinite(camera_centers).all()
    ):
        raise ValueError("camera_centers must be finite floating court coordinates.")
    if ambiguity_margin_m <= 0.0:
        raise ValueError("ambiguity_margin_m must be positive.")
    plus = camera_centers.new_tensor((0.0, HALF_LENGTH, 0.0))
    minus = camera_centers.new_tensor((0.0, -HALF_LENGTH, 0.0))
    distance_plus = torch.linalg.vector_norm(camera_centers - plus, dim=-1)
    distance_minus = torch.linalg.vector_norm(camera_centers - minus, dim=-1)
    margin = (distance_plus - distance_minus).abs()
    valid = margin >= ambiguity_margin_m
    signs = torch.where(
        distance_minus < distance_plus,
        torch.ones_like(distance_plus),
        -torch.ones_like(distance_plus),
    )
    return signs, valid


def select_reference_view(
    camera_centers: Tensor,
    *,
    rng: np.random.Generator,
    ambiguity_margin_m: float = REFERENCE_BASELINE_AMBIGUITY_MARGIN_M,
) -> tuple[int, float]:
    """Sample one unambiguous view using the dataset's deterministic RNG."""
    signs, valid = orientation_signs_from_camera_centers(
        camera_centers,
        ambiguity_margin_m=ambiguity_margin_m,
    )
    candidates: NDArray[np.int64] = np.flatnonzero(valid.cpu().numpy()).astype(
        np.int64
    )
    if candidates.size == 0:
        raise ValueError(
            "no reference camera exceeds the baseline-distance ambiguity margin."
        )
    selected = int(rng.choice(candidates))
    return selected, float(signs[selected].item())


def validate_declared_reference_orientation(
    camera_centers: Tensor,
    view_mask: Tensor,
    reference_view_index: Tensor,
    orientation_sign: Tensor,
    *,
    ambiguity_margin_m: float = REFERENCE_BASELINE_AMBIGUITY_MARGIN_M,
) -> None:
    """Validate that each declared sign follows its selected camera geometry."""
    if camera_centers.ndim != 3 or camera_centers.shape[-1] != 3:
        raise ValueError("camera_centers must have shape (B,V,3).")
    batch_size, views = camera_centers.shape[:2]
    if view_mask.shape != (batch_size, views) or view_mask.dtype != torch.bool:
        raise ValueError("view_mask must be boolean (B,V).")
    if reference_view_index.shape != (batch_size,):
        raise ValueError("reference_view_index must have shape (B,).")
    if reference_view_index.dtype == torch.bool or reference_view_index.is_floating_point():
        raise TypeError("reference_view_index must use an integer dtype.")
    if orientation_sign.shape != (batch_size,):
        raise ValueError("orientation_sign must have shape (B,).")
    if not camera_centers.is_floating_point() or not orientation_sign.is_floating_point():
        raise TypeError("camera centers and orientation signs must be floating tensors.")
    if not bool(torch.isfinite(orientation_sign).all()):
        raise ValueError("orientation_sign must be finite.")
    if len(
        {
            camera_centers.device,
            view_mask.device,
            reference_view_index.device,
            orientation_sign.device,
        }
    ) != 1:
        raise ValueError("reference orientation tensors must share one device.")

    for batch_index in range(batch_size):
        current = int(reference_view_index[batch_index].item())
        if current < 0 or current >= views or not bool(view_mask[batch_index, current]):
            raise ValueError("reference_view_index must select a valid unpadded view.")
        signs, unambiguous = orientation_signs_from_camera_centers(
            camera_centers[batch_index],
            ambiguity_margin_m=ambiguity_margin_m,
        )
        if not bool(unambiguous[current]):
            raise ValueError("the selected reference camera is orientation-ambiguous.")
        if not bool(signs[current] == orientation_sign[batch_index]):
            raise ValueError(
                "orientation_sign does not match the selected reference camera."
            )


def select_counterfactual_reference_views(
    camera_centers: Tensor,
    view_mask: Tensor,
    reference_view_index: Tensor,
    orientation_sign: Tensor,
) -> tuple[Tensor, Tensor]:
    """Choose one deterministic valid alternate reference for every sample.

    Opposite-side references are preferred so the paired evaluation exercises
    the Y/heading reflection. If none exists, the lowest-index same-side view is
    selected to measure same-side stability. A sample without either fails.
    """
    validate_declared_reference_orientation(
        camera_centers,
        view_mask,
        reference_view_index,
        orientation_sign,
    )
    batch_size, views = camera_centers.shape[:2]

    alternate_indices: list[int] = []
    alternate_signs: list[Tensor] = []
    for batch_index in range(batch_size):
        signs, unambiguous = orientation_signs_from_camera_centers(
            camera_centers[batch_index]
        )
        current = int(reference_view_index[batch_index].item())
        primary_sign = orientation_sign[batch_index]
        candidates = view_mask[batch_index] & unambiguous
        candidates[current] = False
        opposite = candidates & signs.ne(primary_sign)
        eligible = opposite if bool(opposite.any()) else candidates
        indices = torch.nonzero(eligible, as_tuple=False).flatten()
        if indices.numel() == 0:
            raise ValueError(
                "paired reference consistency requires a valid alternate reference view."
            )
        selected = int(indices.min().item())
        alternate_indices.append(selected)
        alternate_signs.append(signs[selected])
    return (
        torch.tensor(
            alternate_indices,
            dtype=reference_view_index.dtype,
            device=reference_view_index.device,
        ),
        torch.stack(alternate_signs).to(dtype=orientation_sign.dtype),
    )


def reflect_court_vectors(value: Tensor, orientation_sign: Tensor | float) -> Tensor:
    """Reflect the Y component of position or vector tensors."""
    if value.shape[-1] != 3 or not value.is_floating_point():
        raise ValueError("court vectors must be floating tensors ending in XYZ.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("court vectors must be finite.")
    sign = torch.as_tensor(orientation_sign, dtype=value.dtype, device=value.device)
    if not bool(((sign == 1.0) | (sign == -1.0)).all()):
        raise ValueError("orientation_sign must contain only -1 or +1.")
    while sign.ndim < value.ndim - 1:
        sign = sign.unsqueeze(-1)
    result = value.clone()
    result[..., 1] *= sign
    return result


def reflect_heading(value: Tensor, orientation_sign: Tensor | float) -> Tensor:
    """Reflect a court XY heading vector and re-normalize it."""
    if value.shape[-1] != 2 or not value.is_floating_point():
        raise ValueError("heading must be a floating tensor ending in (cos,sin).")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("heading must be finite.")
    sign = torch.as_tensor(orientation_sign, dtype=value.dtype, device=value.device)
    if not bool(((sign == 1.0) | (sign == -1.0)).all()):
        raise ValueError("orientation_sign must contain only -1 or +1.")
    while sign.ndim < value.ndim - 1:
        sign = sign.unsqueeze(-1)
    result = value.clone()
    result[..., 1] *= sign
    return F.normalize(result, dim=-1)


__all__ = [
    "REFERENCE_BASELINE_AMBIGUITY_MARGIN_M",
    "camera_centers_from_scene_payload",
    "deterministic_sample_rng",
    "orientation_signs_from_camera_centers",
    "reflect_court_vectors",
    "reflect_heading",
    "select_counterfactual_reference_views",
    "select_reference_view",
    "validate_declared_reference_orientation",
]
