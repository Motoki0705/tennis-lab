"""BLCS composition root that binds each model to exactly one I/O adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, cast

from src.tasks.base.configuration import as_config_mapping, require_config_mapping
from src.tasks.base.model_io import (
    BoundModelIO,
    TrackQueryReferenceContract,
    bind_model_io,
)
from src.tasks.base.models import resolve_reference_selector_mode
from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.blcs.configuration import (
    AxialModelConfig,
    SingleModelConfig,
    TrackQueryModelConfig,
    TrackQueryReferenceModelConfig,
    parse_court_keypoint_contract,
    parse_model_config,
)
from src.tasks.blcs.model_io.adapters import (
    AxialTrajectoryModelIOAdapter,
    RawBLCSOutput,
    SingleTrajectoryModelIOAdapter,
    TrackQueryModelIOAdapter,
    TrackQueryReferenceModelIOAdapter,
    TrajectoryModelIOAdapter,
)
from src.tasks.blcs.model_io.contracts import (
    BLCSTrackQueryPrediction,
    BLCSTrajectoryPrediction,
)
from src.tasks.blcs.models.blcs_model import BLCSModel
from src.tasks.blcs.models.blcs_multiview_axial_model import BLCSMultiViewAxialModel
from src.tasks.blcs.models.blcs_track_query_model import BLCSTrackQueryModel
from src.tasks.blcs.models.blcs_track_query_reference_model import (
    BLCSTrackQueryReferenceModel,
)

TrajectoryBoundModelIO: TypeAlias = BoundModelIO[
    Mapping[str, object], RawBLCSOutput, BLCSTrajectoryPrediction
]
TrackQueryBoundModelIO: TypeAlias = BoundModelIO[
    Mapping[str, object], RawBLCSOutput, BLCSTrackQueryPrediction
]
BLCSBoundModelIO: TypeAlias = TrajectoryBoundModelIO | TrackQueryBoundModelIO


def _tracking_presence_threshold(config: object) -> float:
    root = as_config_mapping(config, path="configuration")
    metrics = TrackingMetricConfig.from_mapping(
        require_config_mapping(root, "tracking_metrics", path="configuration")
    )
    return float(metrics.presence_threshold)


def compose_blcs_model_io(config: object) -> BLCSBoundModelIO:
    """Construct and bind the configured model/adapter pair exactly once."""
    model_config = parse_model_config(config)
    court_keypoint_contract = parse_court_keypoint_contract(config)
    if isinstance(model_config, SingleModelConfig):
        single_model = BLCSModel.from_config(model_config)
        single_adapter = SingleTrajectoryModelIOAdapter(
            num_court_tokens=model_config.num_court_tokens,
            max_seq_len=model_config.max_seq_len,
            predict_velocity=model_config.predict_velocity,
            input_profile=model_config.input_profile,
            max_num_cameras=None,
            court_keypoint_contract=court_keypoint_contract,
        )
        return cast(
            "TrajectoryBoundModelIO", bind_model_io(single_model, single_adapter)
        )
    if isinstance(model_config, AxialModelConfig):
        axial_model = BLCSMultiViewAxialModel.from_config(model_config)
        axial_adapter = AxialTrajectoryModelIOAdapter(
            num_court_tokens=model_config.num_court_tokens,
            max_seq_len=model_config.max_seq_len,
            predict_velocity=model_config.predict_velocity,
            input_profile=model_config.input_profile,
            max_num_cameras=model_config.max_num_cameras,
            court_keypoint_contract=court_keypoint_contract,
        )
        return cast("TrajectoryBoundModelIO", bind_model_io(axial_model, axial_adapter))
    if isinstance(model_config, TrackQueryReferenceModelConfig):
        reference_model = BLCSTrackQueryReferenceModel(model_config)
        reference_contract = TrackQueryReferenceContract.reference_v2(
            resolve_reference_selector_mode(model_config.reference_selector_mode)
        )
        reference_adapter = TrackQueryReferenceModelIOAdapter(
            num_court_tokens=reference_model.num_court_tokens,
            num_queries=model_config.num_queries,
            presence_threshold=_tracking_presence_threshold(config),
            court_keypoint_contract=court_keypoint_contract,
            track_query_reference_contract=reference_contract,
        )
        return cast(
            "TrackQueryBoundModelIO",
            bind_model_io(reference_model, reference_adapter),
        )
    if isinstance(model_config, TrackQueryModelConfig):
        tracking_model = BLCSTrackQueryModel(model_config)
        tracking_adapter = TrackQueryModelIOAdapter(
            num_court_tokens=tracking_model.num_court_tokens,
            num_queries=model_config.num_queries,
            presence_threshold=_tracking_presence_threshold(config),
            court_keypoint_contract=court_keypoint_contract,
        )
        return cast(
            "TrackQueryBoundModelIO",
            bind_model_io(tracking_model, tracking_adapter),
        )
    raise AssertionError("parse_model_config returned an unclassified BLCS contract.")


def compose_blcs_trajectory_model_io(config: object) -> TrajectoryBoundModelIO:
    """Compose a standard trajectory pair and reject tracking configurations."""
    binding = compose_blcs_model_io(config)
    if not isinstance(binding.adapter, TrajectoryModelIOAdapter):
        raise ValueError("A trajectory BLCS model configuration is required.")
    return binding


def compose_blcs_track_query_model_io(config: object) -> TrackQueryBoundModelIO:
    """Compose a track-query pair and reject standard trajectory configurations."""
    binding = compose_blcs_model_io(config)
    if not isinstance(binding.adapter, TrackQueryModelIOAdapter):
        raise ValueError("A track-query BLCS model configuration is required.")
    return binding


__all__ = [
    "BLCSBoundModelIO",
    "TrackQueryBoundModelIO",
    "TrajectoryBoundModelIO",
    "compose_blcs_model_io",
    "compose_blcs_track_query_model_io",
    "compose_blcs_trajectory_model_io",
]
