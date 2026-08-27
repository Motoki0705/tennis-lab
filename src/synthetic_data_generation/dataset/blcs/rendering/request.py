"""Public NHT composition request for one complete BLCS Gaussian timeline."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment import MetricSceneAdapter
from src.synthetic_data_generation.dataset.blcs.ball_asset import (
    build_ball_gaussian_asset,
)
from src.synthetic_data_generation.dataset.blcs.contracts import (
    BLCSCompositionAssets,
)
from src.synthetic_data_generation.dataset.blcs.timeline import BLCSTrajectoryPlan

NHT_COMPOSED_RENDER_REQUEST_SCHEMA = "nht_composed_render_request_v1"


@dataclass(frozen=True, slots=True)
class BLCSNHTCompositionFiles:
    """Three ordinary files that constitute one immutable public request."""

    request_path: Path
    asset_path: Path
    timeline_path: Path


def write_blcs_nht_composition_request(
    directory: Path,
    *,
    plan: BLCSTrajectoryPlan,
    assets: BLCSCompositionAssets,
    metric_adapter: MetricSceneAdapter,
) -> BLCSNHTCompositionFiles:
    """Write asset-local Gaussians and every NHT-space rigid placement once."""
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"NHT composition request directory already exists: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    asset_path = directory / "ball-gaussians.npz"
    timeline_path = directory / "timeline.npz"
    request_path = directory / "composition.json"

    ball = build_ball_gaussian_asset(assets)
    _write_npz_atomic(
        asset_path,
        means_m=ball.means.cpu().numpy(),
        quats_wxyz=ball.quaternions_wxyz.cpu().numpy(),
        log_scales_m=ball.log_scales.cpu().numpy(),
        opacity_logits=ball.opacity_logits.cpu().numpy(),
        colors_linear_rgb=ball.features.cpu().numpy(),
    )

    frame_count = plan.source.frame_count
    object_count = plan.source.object_count
    transforms = np.repeat(
        np.eye(4, dtype=np.float64)[None, None],
        frame_count * object_count,
        axis=0,
    ).reshape(frame_count, object_count, 4, 4)
    object_index = {
        scene_object.object_id: index
        for index, scene_object in enumerate(plan.composition.objects)
    }
    if set(object_index) != {track.object_id for track in plan.source.tracks}:
        raise ValueError("BLCS Gaussian objects differ from the physical trajectory tracks.")
    nht_from_metric = metric_adapter.nht_matrix()
    for frame in plan.composition.frames:
        for instance in frame.instances:
            metric_from_asset = instance.scene_from_asset.rigid.matrix()
            metric_from_asset[:3, :3] *= instance.scene_from_asset.scale
            nht_from_asset = nht_from_metric @ metric_from_asset
            _validate_positive_similarity(nht_from_asset)
            transforms[frame.frame_index, object_index[instance.object_id]] = nht_from_asset

    instance_ids = np.asarray(
        [item.instance_id for item in plan.composition.objects],
        dtype=np.int32,
    )
    if not np.array_equal(instance_ids, np.arange(1, object_count + 1, dtype=np.int32)):
        raise ValueError("BLCS Gaussian instance IDs must exactly equal 1..object_count.")
    _write_npz_atomic(
        timeline_path,
        transforms_nht_from_asset=transforms,
        present=plan.source.present,
        instance_ids=instance_ids,
    )

    settings = assets.settings
    payload = {
        "schema": NHT_COMPOSED_RENDER_REQUEST_SCHEMA,
        "asset": {
            "asset_id": assets.ball.asset_id,
            "coordinate_space": "right_handed_asset_local_metres",
            "appearance_model": "direct_linear_rgb",
            "gaussian_count": assets.ball.gaussian_count,
            "tensors": asset_path.name,
        },
        "timeline": {
            "coordinate_space": "canonical NHT scene space",
            "frame_count": frame_count,
            "object_count": object_count,
            "object_ids": [item.object_id for item in plan.composition.objects],
            "instance_ids": instance_ids.tolist(),
            "tensors": timeline_path.name,
            "chunks": [chunk.to_dict() for chunk in plan.chunks],
        },
        "visibility_threshold": settings.visibility_threshold,
    }
    _write_json_atomic(request_path, payload)
    return BLCSNHTCompositionFiles(
        request_path=request_path,
        asset_path=asset_path,
        timeline_path=timeline_path,
    )


def _validate_positive_similarity(matrix: NDArray[np.float64]) -> None:
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("NHT asset transform must be one finite 4x4 matrix.")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-10, rtol=0.0):
        raise ValueError("NHT asset transform must be homogeneous.")
    linear = matrix[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("NHT asset transform must have positive uniform scale.")
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6, rtol=0.0):
        raise ValueError("NHT asset transform is not a uniform proper similarity.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6, rtol=0.0):
        raise ValueError("NHT asset transform rotation must be proper.")


def _write_npz_atomic(path: Path, **arrays: NDArray[np.generic]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **cast(Any, arrays))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = [
    "BLCSNHTCompositionFiles",
    "NHT_COMPOSED_RENDER_REQUEST_SCHEMA",
    "write_blcs_nht_composition_request",
]
