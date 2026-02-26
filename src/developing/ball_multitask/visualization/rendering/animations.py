"""Animation composition for ball multitask visualization."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from matplotlib.animation import FuncAnimation

from src.tasks.blcs.visualization.rendering import BLCSSceneRenderer
from src.developing.ball_multitask.visualization.types import RuntimeConfig, SceneInputs
from src.tasks.event_detection.visualization.rendering.traj3d_animation import (
    create_traj3d_event_animation,
)
from src.tasks.event_detection.visualization.rendering.uv_animation import (
    create_uv_event_animation,
)
from src.tasks.event_detection.visualization.types import Traj3DEventInputs, UVEventInputs
from src.tasks.trajectory_completion.visualization.rendering.uv_animation import (
    create_uv_completion_animation,
)
from src.tasks.trajectory_completion.visualization.types import TrajectoryInputs


@dataclass(frozen=True)
class AnimationArtifact:
    """Named animation artifact for display/save workflow."""

    name: str
    animation: FuncAnimation
    default_filename: str


def _as_bool_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values) > 0


def _build_uv_completion(
    *,
    cfg: RuntimeConfig,
    inputs: SceneInputs,
    outputs: dict[str, Any],
) -> AnimationArtifact:
    tc_cfg = SimpleNamespace(fps=cfg.fps, show_court_lines=cfg.show_court_lines)
    tc_inputs = TrajectoryInputs(
        ball_uv_gt=np.asarray(inputs.ball_uv, dtype=np.float32),
        ball_uv_in=np.asarray(inputs.ball_uv, dtype=np.float32),
        ball_gt_visible=_as_bool_array(inputs.ball_vis),
        ball_obs_mask=_as_bool_array(inputs.ball_vis),
        court_kp=np.asarray(inputs.court_kp, dtype=np.float32),
        court_vis=_as_bool_array(inputs.court_vis),
        meta=inputs.meta,
        camera_idx=int(inputs.camera_idx),
        start=0,
    )
    pred_uv = np.asarray(outputs["uv_completed"], dtype=np.float32)
    anim = create_uv_completion_animation(cfg=tc_cfg, inputs=tc_inputs, pred_uv=pred_uv)
    return AnimationArtifact(
        name="uv_completion",
        animation=anim,
        default_filename="uv_completion",
    )


def _build_uv_event(
    *,
    cfg: RuntimeConfig,
    inputs: SceneInputs,
    outputs: dict[str, Any],
) -> AnimationArtifact:
    ed_cfg = SimpleNamespace(
        fps=cfg.fps,
        event_radius_frames=cfg.event_radius_frames,
        event_sigma_frames=cfg.event_sigma_frames,
        show_court_lines=cfg.show_court_lines,
    )
    uv_inputs = UVEventInputs(
        ball_uv=np.asarray(inputs.ball_uv, dtype=np.float32),
        ball_vis=_as_bool_array(inputs.ball_vis),
        court_kp=np.asarray(inputs.court_kp, dtype=np.float32),
        court_vis=_as_bool_array(inputs.court_vis),
        targets=np.zeros((inputs.seq_len, 2), dtype=np.float32),
        shot_indices=[],
        bounce_indices=[],
        meta=inputs.meta,
        camera_idx=int(inputs.camera_idx),
    )
    pred_peaks = outputs.get("event_peaks")
    anim = create_uv_event_animation(cfg=ed_cfg, inputs=uv_inputs, pred_peaks=pred_peaks)
    return AnimationArtifact(name="uv_event", animation=anim, default_filename="uv_event")


def _build_traj3d_event(
    *,
    cfg: RuntimeConfig,
    inputs: SceneInputs,
    outputs: dict[str, Any],
) -> AnimationArtifact:
    if inputs.ball_pos_world is None:
        raise KeyError("Missing 'ball_pos_world' in scene. traj3d_event renderer requires GT 3D trajectory.")

    ed_cfg = SimpleNamespace(
        fps=cfg.fps,
        event_radius_frames=cfg.event_radius_frames,
        event_sigma_frames=cfg.event_sigma_frames,
    )
    traj3d_inputs = Traj3DEventInputs(
        ball_pos_world=np.asarray(inputs.ball_pos_world, dtype=np.float32),
        targets=np.zeros((inputs.seq_len, 2), dtype=np.float32),
        shot_indices=[],
        bounce_indices=[],
        meta=inputs.meta,
    )
    pred_peaks = outputs.get("event_peaks")
    anim = create_traj3d_event_animation(
        cfg=ed_cfg,
        inputs=traj3d_inputs,
        pred_peaks=pred_peaks,
    )
    return AnimationArtifact(
        name="traj3d_event",
        animation=anim,
        default_filename="traj3d_event",
    )


def _build_blcs_traj3d(
    *,
    cfg: RuntimeConfig,
    inputs: SceneInputs,
    outputs: dict[str, Any],
) -> AnimationArtifact:
    _ = outputs
    if inputs.ball_pos_world is None:
        raise KeyError("Missing 'ball_pos_world' in scene. blcs_traj3d renderer requires GT 3D trajectory.")

    scene = {
        "meta": inputs.meta,
        "ball_pos_world": np.asarray(inputs.ball_pos_world, dtype=np.float32),
    }
    renderer = BLCSSceneRenderer()
    anim = renderer.create_animation(
        scene=scene,
        view="3d",
        fps=float(cfg.fps),
    )
    if anim is None:
        raise ValueError("BLCSSceneRenderer failed to create 3D animation.")
    return AnimationArtifact(
        name="blcs_traj3d",
        animation=anim,
        default_filename="blcs_traj3d",
    )


def build_animations(
    *,
    cfg: RuntimeConfig,
    inputs: SceneInputs,
    outputs: dict[str, Any],
) -> list[AnimationArtifact]:
    """Build selected animations from shared prediction outputs."""
    builders = {
        "uv_completion": _build_uv_completion,
        "uv_event": _build_uv_event,
        "traj3d_event": _build_traj3d_event,
        "blcs_traj3d": _build_blcs_traj3d,
    }

    if len(cfg.renderers) == 0:
        raise ValueError("No renderers configured. Set visualization.renderers with at least one renderer.")

    animations: list[AnimationArtifact] = []
    for name in cfg.renderers:
        key = name.strip().lower()
        if key not in builders:
            valid = ", ".join(sorted(builders.keys()))
            raise ValueError(f"Unknown renderer '{name}'. Expected one of: {valid}")
        animations.append(builders[key](cfg=cfg, inputs=inputs, outputs=outputs))
    return animations
