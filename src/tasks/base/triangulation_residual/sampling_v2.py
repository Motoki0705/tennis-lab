"""Full-rig corruption followed by explicitly audited feasible view selection."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from src.tasks.base.triangulation_residual.cameras import fixed_six_camera_rig
from src.tasks.base.triangulation_residual.configuration import ResidualConfig
from src.tasks.base.triangulation_residual.contracts import CameraRig
from src.tasks.base.triangulation_residual.corruption_v2 import corrupt_candidates
from src.tasks.base.triangulation_residual.geometry import (
    InsufficientGeometryError,
    prepare_geometry,
)


def sample_v2(
    world: np.ndarray,
    frame_valid: np.ndarray,
    fps: float,
    source_rig: CameraRig,
    rng: np.random.Generator,
    config: ResidualConfig,
    *,
    scene_id: str,
    split: str,
) -> dict[str, Any]:
    """Targets/window/split stay fixed across bounded observation-only retries."""
    v2 = config.v2
    if v2 is None:
        raise ValueError("sample_v2 requires an explicit v2 recipe")
    if not np.all(source_rig.image_size == source_rig.image_size[0]):
        raise ValueError(
            "The synthetic six-camera profile requires one common image size"
        )
    base_rig = fixed_six_camera_rig(
        (int(source_rig.image_size[0, 0]), int(source_rig.image_size[0, 1]))
    )
    n_views = (
        int(rng.integers(config.data.min_views, config.data.max_views + 1))
        if split == "train"
        else v2.evaluation_views
    )
    # Keep view ordering independent of the corruption mode for paired probes.
    subset_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
    corruption_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
    all_subsets = [
        subset_rng.permutation(candidate).astype(np.int64)
        for candidate in combinations(range(6), n_views)
    ]
    subset_rng.shuffle(all_subsets)
    draw_count = 0
    failed_cameras: np.ndarray = np.zeros(6, dtype=np.int64)
    failure_reasons: dict[str, int] = {}
    fixed_error: tuple[int, float] | None = None
    for attempt in range(8):
        candidates = corrupt_candidates(
            world,
            base_rig,
            corruption_rng,
            config.corruption,
            v2,
            fps=fps,
            task=config.task,
            fixed_error=fixed_error,
        )
        fixed_error = (int(candidates.family), candidates.severity)
        valid_camera_ids = set(int(i) for i in candidates.valid_indices)
        for index, failure in enumerate(candidates.court_fit.failures):
            if failure is not None:
                failed_cameras[index] += 1
                reason = str(failure.reason)
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        for candidate in all_subsets:
            if not set(candidate).issubset(valid_camera_ids):
                continue
            selected = candidate
            noisy = candidates.subset(selected)
            noisy.scores[:, ~frame_valid] = 0
            noisy.observations_px[:, ~frame_valid] = np.nan
            draw_count += 1
            try:
                geometry = prepare_geometry(
                    noisy.observations_px,
                    noisy.scores,
                    noisy.court_px,
                    noisy.court_scores,
                    noisy.estimated_rig,
                    root_indices=config.root_indices,
                    fps=fps,
                    feature_config=config.features,
                    min_score=config.initializer.min_score,
                    refinement_steps=config.initializer.refinement_steps,
                )
            except InsufficientGeometryError:
                continue
            true_p = noisy.true_rig.matrices.copy()
            true_p[:, 0] /= noisy.true_rig.image_size[:, 0, None]
            true_p[:, 1] /= noisy.true_rig.image_size[:, 1, None]
            event_mask = (
                candidates.persistent_mask[selected] & frame_valid[None, :, None]
            )
            return {
                "features": geometry.features,
                "view_valid": geometry.view_valid & frame_valid[None],
                "time_positions": geometry.time_positions,
                "root_init": geometry.root_init_m,
                "relative_init": geometry.relative_init_m,
                "init_world": geometry.init_world_m,
                "init_valid": geometry.init_valid & frame_valid[:, None],
                "target_world": world,
                "frame_valid": frame_valid,
                "true_projection": true_p.astype(np.float32),
                "clean_uv": noisy.clean_uv,
                "clean_visible": noisy.clean_visible & frame_valid[None, :, None],
                "fps": np.array(fps, np.float32),
                "severity": np.array(noisy.severity, np.float32),
                "geometry_attempts": np.array(draw_count, np.int64),
                "corruption_rounds": np.array(attempt + 1, np.int64),
                "selected_camera_indices": selected,
                "num_views": np.array(n_views, np.int64),
                "corruption_family": np.array(candidates.family, np.int64),
                "calibration_attempts": np.array(6 * (attempt + 1), np.int64),
                "calibration_failed_candidates": failed_cameras,
                "persistent_fraction": np.array(event_mask.mean(), np.float32),
                "persistent_mask": event_mask.any(axis=0),
                "persistent_kind": candidates.persistent_kind[selected],
            }
    raise InsufficientGeometryError(
        f"v2 {scene_id}: no feasible {n_views}-camera subset after 8 full-rig draws; "
        f"geometry_attempts={draw_count}, calibration_failures={failure_reasons}"
    )
