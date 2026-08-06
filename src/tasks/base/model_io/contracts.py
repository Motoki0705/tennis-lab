"""Typed lifecycle contracts shared by task-specific model I/O adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, Protocol, TypeAlias, TypeVar, cast

from torch import Tensor, nn

BatchT_contra = TypeVar("BatchT_contra", contravariant=True)
RawOutputT_contra = TypeVar("RawOutputT_contra", contravariant=True)
DecodedOutputT_co = TypeVar("DecodedOutputT_co", covariant=True)
BatchT = TypeVar("BatchT")
RawOutputT = TypeVar("RawOutputT")
DecodedOutputT = TypeVar("DecodedOutputT")

ModelArgument: TypeAlias = Tensor | None


class ModelIOContractError(ValueError):
    """Base error raised when a model I/O boundary contract is violated."""


class ModelAdapterMismatchError(ModelIOContractError):
    """Raised when an adapter is bound to an unsupported model class."""


class ModelInputContractError(ModelIOContractError):
    """Raised when a batch cannot produce a valid model invocation."""


class ModelOutputContractError(ModelIOContractError):
    """Raised when a model result cannot be decoded under the active contract."""


@dataclass(frozen=True, slots=True)
class ModelCall:
    """Immutable tensor arguments prepared before entering ``nn.Module.forward``.

    Task adapters own tensor names and semantics. This shared value only carries
    the already validated positional and keyword arguments to the model.
    """

    args: tuple[ModelArgument, ...] = ()
    kwargs: Mapping[str, ModelArgument] = field(default_factory=dict)

    def __post_init__(self) -> None:
        args = tuple(self.args)
        kwargs = dict(self.kwargs)
        for index, value in enumerate(args):
            if value is not None and not isinstance(value, Tensor):
                raise ModelInputContractError(
                    f"ModelCall args[{index}] must be a tensor or None, got "
                    f"{type(value).__name__}."
                )
        for name, value in kwargs.items():
            if not isinstance(name, str) or not name:
                raise ModelInputContractError(
                    "ModelCall keyword names must be non-empty strings."
                )
            if value is not None and not isinstance(value, Tensor):
                raise ModelInputContractError(
                    f"ModelCall keyword {name!r} must be a tensor or None, got "
                    f"{type(value).__name__}."
                )
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "kwargs", MappingProxyType(kwargs))


class ModelIOAdapter(
    Protocol[BatchT_contra, RawOutputT_contra, DecodedOutputT_co]
):
    """Task-local adapter contract consumed by model-agnostic lifecycles."""

    @property
    def model_type(self) -> type[nn.Module]:
        """Return the model class accepted by this adapter."""
        ...

    def build_call(self, batch: BatchT_contra) -> ModelCall:
        """Validate one external batch and build its immutable model call."""
        ...

    def decode_output(self, output: RawOutputT_contra) -> DecodedOutputT_co:
        """Validate and decode one raw model result."""
        ...


@dataclass(frozen=True, slots=True)
class BoundModelIO(Generic[BatchT, RawOutputT, DecodedOutputT]):
    """A model and its once-selected, construction-validated I/O adapter."""

    model: nn.Module
    adapter: ModelIOAdapter[BatchT, RawOutputT, DecodedOutputT]

    def build_call(self, batch: BatchT) -> ModelCall:
        """Validate a batch at the I/O boundary without invoking the model."""
        return self.adapter.build_call(batch)

    def execute_call(self, call: ModelCall) -> RawOutputT:
        """Run an already validated call through the bound model."""
        return cast(
            RawOutputT,
            self.model(*call.args, **dict(call.kwargs)),
        )

    def decode_output(self, output: RawOutputT) -> DecodedOutputT:
        """Decode a raw model output through the selected adapter."""
        return self.adapter.decode_output(output)

    def run(self, batch: BatchT) -> DecodedOutputT:
        """Build, execute, and decode one batch under the bound contract."""
        call = self.build_call(batch)
        return self.decode_output(self.execute_call(call))


def bind_model_io(
    model: nn.Module,
    adapter: ModelIOAdapter[BatchT, RawOutputT, DecodedOutputT],
) -> BoundModelIO[BatchT, RawOutputT, DecodedOutputT]:
    """Validate a model-adapter pair once at its composition boundary."""
    expected_type = adapter.model_type
    if not isinstance(model, expected_type):
        raise ModelAdapterMismatchError(
            f"{type(adapter).__name__} requires {expected_type.__module__}."
            f"{expected_type.__qualname__}, got "
            f"{type(model).__module__}.{type(model).__qualname__}."
        )
    return BoundModelIO(model=model, adapter=adapter)


__all__ = [
    "BoundModelIO",
    "ModelAdapterMismatchError",
    "ModelArgument",
    "ModelCall",
    "ModelIOAdapter",
    "ModelIOContractError",
    "ModelInputContractError",
    "ModelOutputContractError",
    "bind_model_io",
]
