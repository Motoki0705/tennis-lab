"""Refine teacher outputs with observable geometry and preserve label evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np
from omegaconf import DictConfig

from src.tasks.base.configuration import exact_config_mapping
from src.tennis_scene.dataset_pipeline.geometry import (
    TriangulationSettings,
    fill_triangulation_gaps,
    triangulate_ball,
)
from src.tennis_scene.dataset_pipeline.quality import project
from src.tennis_scene.schema import SceneResult


@dataclass(frozen=True)
class RefinementSettings:
    enabled: bool
    ball_max_reprojection_px: float
    ball_max_speed_mps: float
    player_max_reprojection_px: float
    player_max_speed_mps: float
    max_gap_seconds: float
    single_view_ball_weight: float
    single_view_player_weight: float
    min_player_label_fraction: float
    min_ball_label_fraction: float

    @classmethod
    def from_config(cls, cfg: DictConfig) -> RefinementSettings:
        raw = exact_config_mapping(
            cfg, path="refinement", required_keys=set(cls.__dataclass_fields__)
        )
        if type(raw["enabled"]) is not bool:
            raise TypeError("refinement.enabled must be boolean")
        for name, value in raw.items():
            if name == "enabled":
                continue
            if (
                type(value) not in (float, int)
                or not np.isfinite(cast(float, value))
                or cast(float, value) < 0
            ):
                raise ValueError(f"refinement.{name} must be finite and nonnegative")
            if ("weight" in name or "fraction" in name) and cast(float, value) > 1:
                raise ValueError(f"refinement.{name} must be in [0,1]")
        return cls(**cast(dict[str, Any], raw))


def refine_scene(
    scene: SceneResult, homographies: np.ndarray, settings: RefinementSettings
) -> dict[str, Any]:
    """Keep raw model predictions in the caller's archive; modify supported labels.

    Source codes: 0 learned prior, 1 multiview triangulation, 2 short interpolation,
    3 single-view foot-ground anchor. None is independent measured 3D ground truth.
    """
    assert (
        scene.ball_uv is not None
        and scene.ball_vis is not None
        and scene.ball_3d is not None
    )
    assert scene.human_kp_2d is not None and scene.human_kp_vis is not None
    fits = scene.metadata["reference"]["camera_fits"]
    players, _, total = scene.human_kp_2d.shape[:3]
    player_sources = np.zeros((players, total), np.uint8)
    player_weight = np.zeros((players, total), np.float32)
    old_position = scene.player_position.copy()
    size = (scene.width, scene.height)
    gap = round(settings.max_gap_seconds * scene.fps)
    if len(fits) >= 2:
        ball = triangulate_ball(
            scene.ball_uv,
            scene.ball_vis,
            fits,
            size=size,
            fps=scene.fps,
            settings=TriangulationSettings(
                settings.ball_max_reprojection_px,
                settings.ball_max_speed_mps,
                (15.0, 28.0),
                (-0.1, 12.0),
            ),
        )
        scene.ball_3d, ball_sources = fill_triangulation_gaps(
            ball, scene.ball_3d, max_gap_frames=gap
        )
        ball_weight = np.where(
            ball_sources == 1, 1.0, np.where(ball_sources == 2, 0.4, 0.0)
        ).astype(np.float32)
        for player in range(players):
            # Advanced indexing moves the joint axis; explicit take preserves V,T,J.
            hips = np.take(scene.human_kp_2d[player], [11, 12], axis=2).mean(axis=2)
            visible = (
                np.take(scene.human_kp_vis[player], [11, 12], axis=2) >= 0.3
            ).all(axis=2)
            root = triangulate_ball(
                hips,
                visible,
                fits,
                size=size,
                fps=scene.fps,
                settings=TriangulationSettings(
                    settings.player_max_reprojection_px,
                    settings.player_max_speed_mps,
                    (12.0, 25.0),
                    (0.25, 1.8),
                ),
            )
            scene.player_position[player], player_sources[player] = (
                fill_triangulation_gaps(root, old_position[player], max_gap_frames=gap)
            )
            player_weight[player] = np.where(
                player_sources[player] == 1,
                1.0,
                np.where(player_sources[player] == 2, 0.4, 0.0),
            )
    else:
        ball_sources = np.zeros(total, np.uint8)
        projected, front = project(scene.ball_3d, fits[0])
        residual = np.linalg.norm(projected - scene.ball_uv[0] * size, axis=-1)
        supported = (
            front & scene.ball_vis[0] & (residual <= settings.ball_max_reprojection_px)
        )
        supported &= (scene.ball_3d[:, 2] >= -0.1) & (scene.ball_3d[:, 2] <= 12)
        supported &= (np.abs(scene.ball_3d[:, :2]) <= [15, 28]).all(-1)
        ball_weight = supported.astype(np.float32) * settings.single_view_ball_weight
        for player in range(players):
            feet = (
                np.take(scene.human_kp_2d[player, 0], [15, 16], axis=1).mean(axis=1)
                * size
            )
            ground = cv2.perspectiveTransform(
                feet[None].astype(np.float64), np.linalg.inv(homographies[0])
            )[0]
            visible = (
                np.take(scene.human_kp_vis[player, 0], [15, 16], axis=1) >= 0.3
            ).all(axis=1)
            supported = (
                visible
                & np.isfinite(ground).all(-1)
                & (np.abs(ground) <= [12, 25]).all(-1)
            )
            supported &= (old_position[player, :, 2] >= 0.25) & (
                old_position[player, :, 2] <= 1.8
            )
            scene.player_position[player, supported, :2] = ground[supported]
            player_sources[player, supported] = 3
            player_weight[player, supported] = settings.single_view_player_weight
    if scene.player_kp_3d is not None:
        scene.player_kp_3d += (scene.player_position - old_position)[:, :, None]
    # Reject both endpoints of unphysical jumps from label supervision, without
    # concealing their values in the retained model/refined diagnostic archives.
    ball_fast = (
        np.linalg.norm(np.diff(scene.ball_3d, axis=0), axis=-1) * scene.fps
        > settings.ball_max_speed_mps
    )
    ball_weight[np.r_[ball_fast, False] | np.r_[False, ball_fast]] = 0
    for player in range(players):
        fast = (
            np.linalg.norm(np.diff(scene.player_position[player], axis=0), axis=-1)
            * scene.fps
            > settings.player_max_speed_mps
        )
        player_weight[player, np.r_[fast, False] | np.r_[False, fast]] = 0
    quality = {
        "schema_version": 1,
        "player_weight": player_weight.tolist(),
        "ball_weight": ball_weight.tolist(),
        "player_source": player_sources.tolist(),
        "ball_source": ball_sources.tolist(),
        "source_codes": {
            "0": "learned_prior",
            "1": "multiview_triangulation",
            "2": "short_interpolation",
            "3": "single_view_ground_anchor",
        },
        "is_ground_truth": False,
        "calibration": "learned planar court; approximate pinhole, no lens distortion calibration",
    }
    scene.metadata["label_quality"] = quality
    report = {
        "player_label_fraction": (player_weight > 0).mean(axis=1).tolist(),
        "ball_label_fraction": float((ball_weight > 0).mean()),
        "player_source_counts": [
            {
                str(k): int(v)
                for k, v in zip(*np.unique(source, return_counts=True), strict=True)
            }
            for source in player_sources
        ],
        "ball_source_counts": {
            str(k): int(v)
            for k, v in zip(*np.unique(ball_sources, return_counts=True), strict=True)
        },
    }
    scene.metadata["refinement"] = report
    return report


def check_label_coverage(report: dict[str, Any], settings: RefinementSettings) -> None:
    if (
        min(report["player_label_fraction"]) < settings.min_player_label_fraction
        or report["ball_label_fraction"] < settings.min_ball_label_fraction
    ):
        raise ValueError(
            f"Refined label evidence below configured acceptance threshold: {report}"
        )
