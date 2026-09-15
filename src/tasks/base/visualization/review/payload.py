"""Scene payload container and the entity binary codec.

The entity binary keeps a fixed little-endian layout so the browser can decode
it with aligned typed-array views:

* ``float32`` ``(slots, frames, joint_count, 3)`` in C order,
* ``float32`` ``(slots, frames, 2)`` orientation when ``entity.orientation``, then
* ``uint8`` ``(slots, frames)`` presence when ``entity.presence``.

Element ``(s, t, j, k)`` lives at ``((s * frames + t) * joint_count + j) * 3 + k``.
The two additions to the values the review spec listed are the orientation
block (PLCS yaw is not derivable from the joints) and ordering every ``float32``
section before the trailing ``uint8`` presence, so both typed-array views stay
4-byte aligned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float32]
BoolArray: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ScenePayload:
    """One scene's JSON document plus its undecoded entity binary."""

    document: dict[str, Any]
    binary: bytes

    def json_bytes(self) -> bytes:
        return json.dumps(
            self.document, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")


def pack_entity_frames(
    frames: FloatArray,
    *,
    presence: BoolArray | None = None,
    orientation: FloatArray | None = None,
) -> bytes:
    """Serialize ``(slots, frames, joint_count, 3)`` entity data to bytes."""
    if frames.dtype != np.float32:
        raise ValueError(f"Entity frames must be float32, got {frames.dtype}.")
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(
            "Entity frames must have shape (slots, frames, joint_count, 3), "
            f"got {frames.shape}."
        )
    if not np.isfinite(frames).all():
        raise ValueError("Entity frames contain NaN or infinity.")
    parts = [np.ascontiguousarray(frames, dtype="<f4").tobytes()]
    if orientation is not None:
        expected_orientation = (frames.shape[0], frames.shape[1], 2)
        if orientation.dtype != np.float32 or orientation.shape != expected_orientation:
            raise ValueError(
                f"Orientation must be float32 with shape {expected_orientation}, "
                f"got {orientation.dtype} {orientation.shape}."
            )
        if not np.isfinite(orientation).all():
            raise ValueError("Orientation contains NaN or infinity.")
        parts.append(np.ascontiguousarray(orientation, dtype="<f4").tobytes())
    if presence is not None:
        expected = (frames.shape[0], frames.shape[1])
        if presence.shape != expected:
            raise ValueError(
                f"Presence must have shape {expected}, got {presence.shape}."
            )
        parts.append(np.ascontiguousarray(presence.astype(np.uint8)).tobytes())
    return b"".join(parts)


__all__ = ["BoolArray", "FloatArray", "ScenePayload", "pack_entity_frames"]
