"""Reference-frame Dataset tests for BLCS camera-view CourtKP20."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.tasks.base.generate_dataset import (
    CAMERA_VIEW_V2_SELECTOR,
    apply_court_view_record,
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tasks.blcs.generate_dataset.scene_generator import CameraData

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()


def _camera(
    camera_id: str,
    center: tuple[float, float, float],
    physical_court: np.ndarray,
) -> CameraData:
    contract = resolve_court_keypoint_contract(CAMERA_VIEW_V2_SELECTOR)
    view = build_court_view_record(
        camera_id=camera_id,
        camera_center_court_m=center,
        contract=contract,
    )
    disk = apply_court_view_record(physical_court, view, keypoint_axis=0)
    assert isinstance(disk, np.ndarray)
    return CameraData(
        camera_params={
            "R": np.eye(3).tolist(),
            "C": list(center),
            "f": 100.0,
            "cx": 50.0,
            "cy": 40.0,
            "w": 100,
            "h": 80,
        },
        ball_uv=np.full((2, 2), 0.5, dtype=np.float32),
        ball_vis=np.ones(2, dtype=np.bool_),
        ball_visibility_ratio=1.0,
        court_kp_uv=disk,
        court_kp_vis=np.ones(20, dtype=np.bool_),
        court_visibility_count=20.0,
        court_view=view,
    )
