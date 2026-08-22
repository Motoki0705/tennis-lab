"""PLCS composition factory that binds each model to exactly one I/O adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, cast, overload

from torch import nn

from src.tasks.base.model_io import (
    BoundModelIO,
    ModelAdapterMismatchError,
    bind_model_io,
)
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.model_io.adapters import (
    PLCSAdapter,
    PLCSModelIOAdapter,
    PLCSTrackQueryIOAdapter,
)
from src.tasks.plcs.model_io.contracts import (
    PLCSDecodedPrediction,
    PLCSInputProfile,
    PLCSTrackingDecodedPrediction,
)
from src.tasks.plcs.models.plcs_model import PLCSModel
from src.tasks.plcs.models.plcs_multiview_axial_camtoken_model import (
    PLCSMultiViewAxialCamTokenModel,
)
from src.tasks.plcs.models.plcs_multiview_axial_model import PLCSMultiViewAxialModel
from src.tasks.plcs.models.plcs_multiview_axial_split_model import (
    PLCSMultiViewAxialSplitModel,
)
from src.tasks.plcs.models.plcs_multiview_model import PLCSMultiViewModel
from src.tasks.plcs.models.plcs_track_query_model import PLCSTrackQueryModel

PLCSRawOutput = Mapping[str, object]
PLCSStandardBoundModelIO: TypeAlias = BoundModelIO[
    Mapping[str, object], PLCSRawOutput, PLCSDecodedPrediction
]
PLCSTrackingBoundModelIO: TypeAlias = BoundModelIO[
    Mapping[str, object], PLCSRawOutput, PLCSTrackingDecodedPrediction
]
PLCSBoundModelIO: TypeAlias = PLCSStandardBoundModelIO | PLCSTrackingBoundModelIO


@overload
def bind_plcs_model_io(
    model: nn.Module, adapter: PLCSModelIOAdapter
) -> PLCSStandardBoundModelIO: ...


@overload
def bind_plcs_model_io(
    model: nn.Module, adapter: PLCSTrackQueryIOAdapter
) -> PLCSTrackingBoundModelIO: ...


def bind_plcs_model_io(model: nn.Module, adapter: PLCSAdapter) -> PLCSBoundModelIO:
    """Bind an exact PLCS model/adapter pair and reject subclass mismatches."""
    if type(model) is not adapter.model_type:
        expected = adapter.model_type
        raise ModelAdapterMismatchError(
            f"{type(adapter).__name__} requires exact model type "
            f"{expected.__module__}.{expected.__qualname__}, got "
            f"{type(model).__module__}.{type(model).__qualname__}."
        )
    return cast(PLCSBoundModelIO, bind_model_io(model, adapter))


def _standard_adapter(
    runtime: PLCSTrainingConfig,
    *,
    model_type: type[nn.Module],
    profile: PLCSInputProfile,
    output_rank: int,
    min_views: int = 1,
) -> PLCSModelIOAdapter:
    num_court_tokens = runtime.data.num_court_tokens
    if num_court_tokens is None:
        raise ValueError("Standard PLCS models require data.num_court_kp.")
    values = runtime.model.values
    return PLCSModelIOAdapter(
        model_type=model_type,
        profile=profile,
        num_court_tokens=num_court_tokens,
        camera_index=runtime.data.adapter_camera_index,
        output_rank=output_rank,
        predict_canonical_pose=runtime.model.boolean("predict_canonical_pose"),
        predict_auxiliary_position=(
            runtime.model.boolean("aux_position_on_rotation_branch")
            if "aux_position_on_rotation_branch" in values
            else False
        ),
        max_views=(
            runtime.model.integer("max_views") if "max_views" in values else None
        ),
        max_sequence_length=(
            runtime.model.integer("max_seq_len")
            if "max_seq_len" in values
            else None
        ),
        min_views=min_views,
    )


def build_plcs_model_io(runtime: PLCSTrainingConfig) -> PLCSBoundModelIO:
    """Construct and bind the configured PLCS model and adapter exactly once."""
    model_cfg = runtime.model
    model_name = model_cfg.name
    num_court_tokens = runtime.data.num_court_tokens
    model: nn.Module
    adapter: PLCSAdapter

    if model_name == "plcs":
        if num_court_tokens is None:
            raise ValueError("PLCS frame/sequence models require data.num_court_kp.")
        model = PLCSModel.from_config(model_cfg, num_court_tokens=num_court_tokens)
        data_mode = str(runtime.data.values["mode"])
        profile = (
            PLCSInputProfile.SEQUENCE
            if data_mode == "sequence"
            else PLCSInputProfile.FRAME
        )
        adapter = _standard_adapter(
            runtime,
            model_type=PLCSModel,
            profile=profile,
            output_rank=2,
        )
    elif model_name == "plcs_multiview":
        if num_court_tokens is None:
            raise ValueError("PLCS multiview models require data.num_court_kp.")
        model = PLCSMultiViewModel.from_config(
            model_cfg, num_court_tokens=num_court_tokens
        )
        adapter = _standard_adapter(
            runtime,
            model_type=PLCSMultiViewModel,
            profile=PLCSInputProfile.MULTIVIEW,
            output_rank=3,
        )
    elif model_name == "plcs_multiview_axial":
        if num_court_tokens is None:
            raise ValueError("PLCS axial models require data.num_court_kp.")
        model = PLCSMultiViewAxialModel.from_config(
            model_cfg, num_court_tokens=num_court_tokens
        )
        adapter = _standard_adapter(
            runtime,
            model_type=PLCSMultiViewAxialModel,
            profile=PLCSInputProfile.MULTIVIEW,
            output_rank=3,
        )
    elif model_name == "plcs_multiview_axial_split":
        if num_court_tokens is None:
            raise ValueError("PLCS split models require data.num_court_kp.")
        model = PLCSMultiViewAxialSplitModel.from_config(
            model_cfg, num_court_tokens=num_court_tokens
        )
        adapter = _standard_adapter(
            runtime,
            model_type=PLCSMultiViewAxialSplitModel,
            profile=PLCSInputProfile.MULTIVIEW,
            output_rank=3,
        )
    elif model_name == "plcs_multiview_axial_camtoken":
        if num_court_tokens is None:
            raise ValueError("PLCS camera-token models require data.num_court_kp.")
        model = PLCSMultiViewAxialCamTokenModel.from_config(
            model_cfg, num_court_tokens=num_court_tokens
        )
        adapter = _standard_adapter(
            runtime,
            model_type=PLCSMultiViewAxialCamTokenModel,
            profile=PLCSInputProfile.MULTIVIEW,
            output_rank=3,
            min_views=2,
        )
    elif model_name == "plcs_track_query":
        model = PLCSTrackQueryModel(model_cfg)
        adapter = PLCSTrackQueryIOAdapter(
            model_type=PLCSTrackQueryModel,
            num_queries=model_cfg.integer("num_queries"),
            num_court_tokens=14,
            num_joints=model_cfg.integer("num_joints"),
        )
    else:
        raise ValueError(f"Unsupported validated PLCS model {model_name!r}.")

    return bind_plcs_model_io(model, adapter)


__all__ = [
    "PLCSBoundModelIO",
    "PLCSRawOutput",
    "PLCSStandardBoundModelIO",
    "PLCSTrackingBoundModelIO",
    "bind_plcs_model_io",
    "build_plcs_model_io",
]
