"""Small exact simulated scenes for the BLCS teacher and metric contracts."""

from __future__ import annotations

from typing import Any, cast

import torch
from omegaconf import OmegaConf

from src.tasks.base.generate_dataset import build_physical_court_provenance
from src.tasks.blcs.data.dataset import collate_multiview_trajectories
from src.tasks.blcs.data.types import BLCSMultiViewSample
from src.tasks.blcs.model_io.adapters import AxialTrajectoryModelIOAdapter
from src.tasks.blcs.physics.targets import record_teachers
from src.utils.schema.court_normalization import (
    normalize_court_position,
    normalize_court_velocity,
)
from tests.support.physics.ball_record import simulated_record


def physics_sample(
    frames: int = 12,
    *,
    start: int = 0,
    output_fps: int = 30,
    sim_fps: int = 120,
    surface: str = "hard",
    hits: tuple[int, ...] = (),
    kinds: tuple[str, ...] | None = None,
    name: str = "sample",
) -> dict[str, Any]:
    p, record = simulated_record(
        frames + start,
        output_fps=output_fps,
        sim_fps=sim_fps,
        surface=surface,
        hits=hits,
        hit_kinds=kinds,
    )
    arrays = {
        **record.to_arrays(),
        "ball_vel_norm": normalize_court_velocity(record.velocity),
    }
    return {
        "scene_id": name,
        "ball_uv": torch.full((2, frames, 2), 0.5),
        "ball_vis": torch.ones(2, frames),
        "ball_uv_target": torch.full((2, frames, 2), 0.5),
        "ball_vis_target": torch.ones(2, frames),
        "padding_mask": torch.zeros(2, frames, dtype=torch.bool),
        "court_kp": torch.full((2, frames, 14, 2), 0.5),
        "court_vis": torch.ones(2, frames, 14),
        "position_3d": torch.from_numpy(normalize_court_position(p[start:])),
        "velocity_3d": torch.from_numpy(
            normalize_court_velocity(record.velocity[start:])
        ),
        "seq_len": torch.tensor(frames),
        "camera_R": torch.eye(3).repeat(2, 1, 1),
        "camera_C": torch.tensor([[0.0, 0.0, -30.0]]).repeat(2, 1),
        "camera_f": torch.full((2,), 1000.0),
        "camera_cx": torch.full((2,), 960.0),
        "camera_cy": torch.full((2,), 540.0),
        "camera_w": torch.full((2,), 1920.0),
        "camera_h": torch.full((2,), 1080.0),
        "court_reference_provenance": build_physical_court_provenance(),
        "selected_camera_ids": ("camera_0", "camera_1"),
        **record_teachers(
            arrays, start=start, end=start + frames, full_length=start + frames
        ),
    }


def prepared_samples(samples: list[dict[str, Any]]):
    batch = collate_multiview_trajectories(cast("list[BLCSMultiViewSample]", samples))
    adapter = AxialTrajectoryModelIOAdapter(
        num_court_tokens=14,
        max_seq_len=10240,
        predict_velocity=True,
        input_profile="multiview",
        max_num_cameras=6,
    )
    return batch, adapter.build_training_batch(batch)


def loss_config(**changes: Any) -> dict[str, Any]:
    cfg = dict(OmegaConf.load("src/tasks/blcs/configs/loss/default.yaml"))
    cfg.update(changes)
    return cfg
