"""Match stage-required checkpoints against explicitly pinned SHA-256 values."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from src.tasks.base.configuration import exact_config_mapping
from src.utils.checksum import dual_sha256

_CHECKPOINT_ROLES = frozenset({"court", "dino", "vitpose", "plcs", "blcs", "dinov3"})


def validate_checkpoint_sha256(value: object) -> dict[str, str]:
    """Require all six roles and lowercase SHA-256 hex; null is not omission."""
    raw = exact_config_mapping(
        value, path="checkpoint_sha256", required_keys=_CHECKPOINT_ROLES
    )
    result: dict[str, str] = {}
    for role, digest in raw.items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(
                f"checkpoint_sha256.{role} must be 64 lowercase hex digits"
            )
        result[role] = digest
    return result


def verify_checkpoint_integrity(
    assets: Mapping[str, Path], expected: Mapping[str, str] | None
) -> dict[str, str]:
    """Verify only supplied stage assets; never derive or repair expected values.

    None preserves profiles without pins. Provider/stability failures from
    dual_sha256 propagate, and a trusted-digest mismatch stops generation.
    """
    if expected is None:
        return {}
    pins = validate_checkpoint_sha256(expected)
    unknown = set(assets) - _CHECKPOINT_ROLES
    if unknown:
        raise ValueError(f"Unknown checkpoint asset roles: {sorted(unknown)}")
    verified: dict[str, str] = {}
    for role, path in assets.items():
        actual = dual_sha256(path)
        if actual != pins[role]:
            raise RuntimeError(
                f"Checkpoint SHA-256 mismatch for {role} ({path}): "
                f"expected {pins[role]}, actual {actual}"
            )
        verified[role] = actual
    return verified
