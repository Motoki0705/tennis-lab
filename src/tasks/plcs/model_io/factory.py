"""PLCS composition factory that binds each model to exactly one I/O adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypeAlias, TypeVar

from torch import nn

from src.tasks.base.generate_dataset import CourtKeypointContract
from src.tasks.base.model_io import (
    BoundModelIO,
    ModelAdapterMismatchError,
    ModelCall,
    bind_model_io,
)
from src.tasks.plcs.configuration import PLCSModelConfig
from src.tasks.plcs.model_io.adapters import PLCSModelIOAdapter
from src.tasks.plcs.model_io.contracts import PLCSDecodedPrediction, PLCSInputProfile
from src.tasks.plcs.models.plcs_multiview_axial_model import PLCSMultiViewAxialModel

PLCSRawOutput = Mapping[str, object]
PLCSStandardBoundModelIO: TypeAlias = BoundModelIO[
    Mapping[str, object], PLCSRawOutput, PLCSDecodedPrediction
]
PLCSBoundModelIO: TypeAlias = PLCSStandardBoundModelIO
DecodedPredictionT_co = TypeVar("DecodedPredictionT_co", covariant=True)


class PLCSModelIODataConfig(Protocol):
    @property
    def num_court_tokens(self) -> int | None: ...

    @property
    def adapter_camera_index(self) -> int: ...

    @property
    def values(self) -> Mapping[str, object]: ...


class PLCSModelIOConfig(Protocol):
    """Read-only configuration slice the PLCS model/adapter factory consumes.

    Only the model variant, the resolved data contract, and the CourtKP20
    contract shape model construction. ``PLCSTrainingConfig`` satisfies this
    structurally; inference-only boundaries may pass a lighter object without a
    training/run section instead of fabricating training configuration.
    """

    @property
    def model(self) -> PLCSModelConfig: ...

    @property
    def data(self) -> PLCSModelIODataConfig: ...

    @property
    def court_keypoint_contract(self) -> CourtKeypointContract: ...


class _PLCSBindingAdapter(Protocol[DecodedPredictionT_co]):
    """Structural adapter contract preserving the bound decoded type."""

    @property
    def model_type(self) -> type[nn.Module]: ...

    def build_call(self, batch: Mapping[str, object]) -> ModelCall: ...

    def decode_output(
        self,
        output: PLCSRawOutput,
    ) -> DecodedPredictionT_co: ...


def bind_plcs_model_io(
    model: nn.Module,
    adapter: _PLCSBindingAdapter[DecodedPredictionT_co],
) -> BoundModelIO[Mapping[str, object], PLCSRawOutput, DecodedPredictionT_co]:
    """Bind an exact PLCS model/adapter pair and reject subclass mismatches."""
    if type(model) is not adapter.model_type:
        expected = adapter.model_type
        raise ModelAdapterMismatchError(
            f"{type(adapter).__name__} requires exact model type "
            f"{expected.__module__}.{expected.__qualname__}, got "
            f"{type(model).__module__}.{type(model).__qualname__}."
        )
    return bind_model_io(model, adapter)


def _standard_adapter(runtime: PLCSModelIOConfig) -> PLCSModelIOAdapter:
    num_court_tokens = runtime.data.num_court_tokens
    if num_court_tokens is None:
        raise ValueError("PLCS axial requires data.num_court_kp.")
    return PLCSModelIOAdapter(
        model_type=PLCSMultiViewAxialModel,
        profile=PLCSInputProfile.MULTIVIEW,
        num_court_tokens=num_court_tokens,
        camera_index=runtime.data.adapter_camera_index,
        output_rank=3,
        predict_canonical_pose=runtime.model.boolean("predict_canonical_pose"),
        predict_auxiliary_position=False,
        max_views=runtime.model.integer("max_views"),
        max_sequence_length=runtime.model.integer("max_seq_len"),
        min_views=1,
        court_keypoint_contract=runtime.court_keypoint_contract,
    )


def build_plcs_model_io(runtime: PLCSModelIOConfig) -> PLCSStandardBoundModelIO:
    if runtime.model.name != "plcs_multiview_axial":
        raise ValueError(f"Unsupported PLCS model {runtime.model.name!r}.")
    if runtime.court_keypoint_contract.selector != "physical_v1":
        raise ValueError("PLCS axial requires physical_v1 court keypoints.")
    num_court_tokens = runtime.data.num_court_tokens
    if num_court_tokens is None:
        raise ValueError("PLCS axial requires data.num_court_kp.")
    model = PLCSMultiViewAxialModel.from_config(
        runtime.model, num_court_tokens=num_court_tokens
    )
    return bind_plcs_model_io(model, _standard_adapter(runtime))


__all__ = [
    "PLCSBoundModelIO",
    "PLCSModelIOConfig",
    "PLCSRawOutput",
    "PLCSStandardBoundModelIO",
    "bind_plcs_model_io",
    "build_plcs_model_io",
]
