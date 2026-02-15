"""Model input adapters for BLCS visualization prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class PredictorInputs:
    """Tensor inputs consumed by ``BLCSPredictor.predict``."""

    ball_uv: Tensor
    court_kp: Tensor
    ball_vis: Tensor
    court_vis: Tensor
    ball_mask: Tensor | None = None


def _build_single_view_input(scene: dict[str, Any], camera_idx: int) -> PredictorInputs:
    cam = scene["cameras"][camera_idx]
    return PredictorInputs(
        ball_uv=torch.from_numpy(cam["ball_uv"]).float(),
        court_kp=torch.from_numpy(cam["court_kp_uv"]).float(),
        ball_vis=torch.from_numpy(cam["ball_visible"].astype(np.float32)),
        court_vis=torch.from_numpy(cam["court_kp_visible"].astype(np.float32)),
        ball_mask=None,
    )


def _build_multiview_input(scene: dict[str, Any], cameras: list[int]) -> PredictorInputs:
    ball_uv_list: list[np.ndarray] = []
    court_kp_list: list[np.ndarray] = []
    ball_vis_list: list[np.ndarray] = []
    court_vis_list: list[np.ndarray] = []

    for camera_idx in cameras:
        cam = scene["cameras"][camera_idx]
        ball_uv_list.append(cam["ball_uv"])
        court_kp_list.append(cam["court_kp_uv"])
        ball_vis_list.append(cam["ball_visible"].astype(np.float32))
        court_vis_list.append(cam["court_kp_visible"].astype(np.float32))

    ball_vis = torch.from_numpy(np.stack(ball_vis_list, axis=0)).float()
    return PredictorInputs(
        ball_uv=torch.from_numpy(np.stack(ball_uv_list, axis=0)).float(),
        court_kp=torch.from_numpy(np.stack(court_kp_list, axis=0)).float(),
        ball_vis=ball_vis,
        court_vis=torch.from_numpy(np.stack(court_vis_list, axis=0)).float(),
        ball_mask=torch.ones_like(ball_vis),
    )


def build_predict_inputs(scene: dict[str, Any], cameras: list[int]) -> PredictorInputs:
    """Build predictor inputs for one or more cameras."""
    if not cameras:
        raise ValueError("No cameras selected for prediction input building.")
    if len(cameras) == 1:
        return _build_single_view_input(scene, cameras[0])
    return _build_multiview_input(scene, cameras)
