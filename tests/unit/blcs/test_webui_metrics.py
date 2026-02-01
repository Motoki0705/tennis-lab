"""Unit tests for BLCS WebUI derived metrics."""

from __future__ import annotations

import torch

from src.blcs.generate_dataset.api_server.metrics import (
    apex_height_m,
    net_clearance_m,
    time_to_bounce1_s,
)
from src.utils.geometry import NET_HEIGHT_CENTER


def test_apex_height_empty() -> None:
    assert apex_height_m(torch.empty((0, 3))) == 0.0


def test_apex_height_basic() -> None:
    traj = torch.tensor(
        [
            [0.0, 0.0, 0.5],
            [0.0, 1.0, 1.7],
            [0.0, 2.0, 1.2],
        ],
        dtype=torch.float32,
    )
    assert abs(apex_height_m(traj) - 1.7) < 1e-6


def test_time_to_bounce1_s_none_when_missing() -> None:
    assert time_to_bounce1_s(-1, fps_out=30) is None


def test_time_to_bounce1_s_basic() -> None:
    assert time_to_bounce1_s(15, fps_out=30) == 0.5


def test_net_clearance_none_when_no_crossing() -> None:
    traj = torch.tensor(
        [
            [0.0, -2.0, 1.0],
            [0.0, -1.0, 1.1],
            [0.0, -0.5, 1.2],
        ],
        dtype=torch.float32,
    )
    assert net_clearance_m(traj) is None


def test_net_clearance_crossing_center() -> None:
    # Cross y=0 at x=0 with z=1.0; clearance should be z - NET_HEIGHT_CENTER.
    traj = torch.tensor(
        [
            [0.0, -1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    clearance = net_clearance_m(traj)
    assert clearance is not None
    assert abs(clearance - (1.0 - NET_HEIGHT_CENTER)) < 1e-5
