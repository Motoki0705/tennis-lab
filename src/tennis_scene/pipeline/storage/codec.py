"""Schema-directed JSON/NumPy encoding without pickle or dynamic class imports."""

from __future__ import annotations

import math
import sys
import types
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Generic,
    Literal,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np
import torch
from numpy.typing import NDArray

from src.utils.checksum import dual_sha256

ValueT = TypeVar("ValueT")


def encode_value(value: Any, directory: Path, arrays: dict[str, Any]) -> Any:
    if isinstance(value, torch.Tensor):
        return {"tensor": encode_value(value.detach().cpu().numpy(), directory, arrays)}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays cannot be component artifacts")
        name = f"array_{len(arrays):04d}.npy"
        path = directory / name
        np.save(path, value, allow_pickle=False)
        arrays[name] = {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": dual_sha256(path)}
        return {"array": name}
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: encode_value(getattr(value, f.name), directory, arrays) for f in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return encode_value(value.item(), directory, arrays)
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise TypeError("Artifact mapping keys must be strings")
        return {k: encode_value(v, directory, arrays) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode_value(v, directory, arrays) for v in value]
    if value is None or type(value) in (str, int, bool):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"Unsupported artifact value: {type(value).__name__}")


def unpack_value(value: Any, directory: Path, arrays: Mapping[str, Any]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"array"}:
            name = value["array"]
            if not isinstance(name, str) or Path(name).name != name or name not in arrays:
                raise ValueError("Invalid artifact array reference")
            path = directory / name
            declaration = arrays[name]
            if not path.is_file() or dual_sha256(path) != declaration["sha256"]:
                raise ValueError(f"Artifact array checksum mismatch: {path}")
            array = np.load(path, allow_pickle=False, mmap_mode="r")
            if list(array.shape) != declaration["shape"] or str(array.dtype) != declaration["dtype"]:
                raise ValueError("Artifact array schema mismatch")
            return array
        if set(value) == {"tensor"}:
            array = unpack_value(value["tensor"], directory, arrays)
            return torch.from_numpy(np.array(array, copy=True))
        return {k: unpack_value(v, directory, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack_value(v, directory, arrays) for v in value]
    return value


def restore_type(value: Any, annotation: Any) -> Any:
    """Construct only the output type that the selected component declares."""
    if annotation is Any:
        return value
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Literal:
        if value not in args:
            raise ValueError("Artifact literal disagrees with declared schema")
        return value
    if origin in (types.UnionType, Union):
        if value is None and type(None) in args:
            return None
        candidates = [item for item in args if item is not type(None)]
        if len(candidates) != 1:
            raise TypeError("Artifact schemas require unambiguous union members")
        return restore_type(value, candidates[0])
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, dict) or set(value) != {f.name for f in fields(annotation)}:
            raise ValueError(f"Fields disagree with artifact schema {annotation.__name__}")
        hints = get_type_hints(annotation, globalns={**vars(sys.modules[annotation.__module__]), "NDArray": NDArray})
        return annotation(**{name: restore_type(v, hints[name]) for name, v in value.items()})
    if origin is tuple:
        if not isinstance(value, list):
            raise TypeError("Expected artifact sequence")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(restore_type(v, args[0]) for v in value)
        if len(value) != len(args):
            raise ValueError("Artifact tuple length mismatch")
        return tuple(restore_type(v, a) for v, a in zip(value, args, strict=True))
    if origin is list:
        return [restore_type(v, args[0]) for v in value]
    if origin in (dict, Mapping):
        return {k: restore_type(v, args[1]) for k, v in value.items()}
    if annotation is Path:
        return Path(value)
    if annotation in (str, int, bool, float):
        if type(value) is not annotation:
            raise TypeError(f"Artifact requires {annotation.__name__}, got {type(value).__name__}")
        return value
    if origin is np.ndarray or annotation is np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError("Expected NumPy artifact array")
        return value
    if annotation is torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError("Expected tensor artifact")
        return value
    if annotation is type(None) and value is None:
        return None
    raise TypeError(f"Undeclared artifact field type: {annotation}")


class ArtifactCodec(Generic[ValueT]):
    """A component owns its output schema; the store owns file publication."""

    def __init__(self, output_type: type[ValueT]) -> None:
        self.output_type = output_type

    def dump(self, value: ValueT, directory: Path) -> tuple[Any, dict[str, Any]]:
        if not isinstance(value, self.output_type):
            raise TypeError(f"Expected output {self.output_type.__name__}")
        arrays: dict[str, Any] = {}
        return encode_value(value, directory, arrays), arrays

    def load(self, payload: Any, directory: Path, arrays: Mapping[str, Any]) -> ValueT:
        return cast(ValueT, restore_type(unpack_value(payload, directory, arrays), self.output_type))
