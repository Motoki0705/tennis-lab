"""Construct the BLCS axial model together with its validated I/O adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, cast

from src.tasks.base.model_io import BoundModelIO, bind_model_io
from src.tasks.blcs.configuration import (
    parse_court_keypoint_contract,
    parse_model_config,
)
from src.tasks.blcs.model_io.adapters import (
    AxialTrajectoryModelIOAdapter,
    RawBLCSOutput,
)
from src.tasks.blcs.model_io.contracts import BLCSTrajectoryPrediction
from src.tasks.blcs.models.blcs_multiview_axial_model import BLCSMultiViewAxialModel

TrajectoryBoundModelIO: TypeAlias = BoundModelIO[
    Mapping[str, object], RawBLCSOutput, BLCSTrajectoryPrediction
]
BLCSBoundModelIO: TypeAlias = TrajectoryBoundModelIO


def compose_blcs_model_io(config: object) -> TrajectoryBoundModelIO:
    model_config = parse_model_config(config)
    contract = parse_court_keypoint_contract(config)
    model = BLCSMultiViewAxialModel.from_config(model_config)
    adapter = AxialTrajectoryModelIOAdapter(
        num_court_tokens=model_config.num_court_tokens,
        max_seq_len=model_config.max_seq_len,
        predict_velocity=model_config.predict_velocity,
        input_profile=model_config.input_profile,
        max_num_cameras=model_config.max_num_cameras,
        court_keypoint_contract=contract,
    )
    return cast(TrajectoryBoundModelIO, bind_model_io(model, adapter))


def compose_blcs_trajectory_model_io(config: object) -> TrajectoryBoundModelIO:
    return compose_blcs_model_io(config)


__all__ = [
    "BLCSBoundModelIO",
    "TrajectoryBoundModelIO",
    "compose_blcs_model_io",
    "compose_blcs_trajectory_model_io",
]
