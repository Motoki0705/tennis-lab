"""Explicit run-scoped exceptions to checkpoint digest equality, never to hashing."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

from src.utils.checksum import FileIntegrityError

_FIELDS = {
    "checkpoint_sha256": "dino",
    "detector_sha256": "dino",
    "pose_sha256": "vitpose",
}


def validate_warning_roles(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError("checkpoint_warning_roles must be a list")
    roles = tuple(value)
    if any(role not in {"dino", "vitpose"} for role in roles) or len(set(roles)) != len(
        roles
    ):
        raise ValueError(
            "checkpoint_warning_roles permits unique dino/vitpose roles only"
        )
    return roles


@dataclass(frozen=True)
class WarningPolicy:
    pins: Mapping[str, str]
    roles: tuple[str, ...]
    path: Path | None
    records: list[dict[str, Any]] = dataclass_field(default_factory=list)


_POLICY: ContextVar[WarningPolicy | None] = ContextVar(
    "checkpoint_warning_policy", default=None
)


@contextmanager
def checkpoint_warning_policy(
    pins: Mapping[str, str] | None, roles: Sequence[str], warning_path: Path | None
) -> Iterator[list[dict[str, Any]]]:
    allowed = validate_warning_roles(roles)
    if allowed and pins is None:
        raise ValueError("Checkpoint warning policy requires declared checkpoint pins")
    from .checkpoint_integrity import validate_checkpoint_sha256

    checked = validate_checkpoint_sha256(pins) if pins is not None else {}
    policy = WarningPolicy(checked, allowed, warning_path)
    token = _POLICY.set(policy)
    try:
        yield policy.records
    finally:
        _POLICY.reset(token)


def warn_checkpoint_difference(
    actual: object, expected: str, *, role: str, path: Path, context: str
) -> bool:
    """Return whether equality was waived; hash/provider exceptions never enter here."""
    policy = _POLICY.get()
    if policy is None or role not in policy.roles:
        return False
    if not isinstance(actual, str) or re.fullmatch(r"[0-9a-f]{64}", actual) is None:
        raise FileIntegrityError(
            "Invalid observed checkpoint digest",
            details={"path": str(path), "actual": actual},
        )
    record = {
        "role": role,
        "path": str(path),
        "context": context,
        "observed_sha256": actual,
        "expected_sha256": expected,
        "declared_sha256": policy.pins[role],
        "decision": "continue_using_declared_model_identity",
        "limitation": "checkpoint bytes are not authenticated against the declared pin",
    }
    policy.records.append(record)
    if policy.path is not None:
        policy.path.parent.mkdir(parents=True, exist_ok=True)
        with policy.path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    logging.getLogger(__name__).warning(
        "Checkpoint integrity exception: %s", json.dumps(record, sort_keys=True)
    )
    return True


def declared_checkpoint_identity() -> dict[str, str]:
    policy = _POLICY.get()
    return {} if policy is None else {role: policy.pins[role] for role in policy.roles}


def normalize_receipt(
    saved: Mapping[str, Any], *, path: Path, context: str
) -> dict[str, Any]:
    """Comparison-only copy: preserve saved bytes and reject different declared pins."""
    result = dict(saved)
    policy = _POLICY.get()
    if policy is None or not policy.roles:
        return result
    declared = result.pop("declared_checkpoint_sha256", {})
    if not isinstance(declared, dict):
        raise ValueError(f"Invalid declared checkpoint identity: {path}")
    if set(declared) - {"dino", "vitpose"}:
        raise ValueError(f"Unknown declared checkpoint roles: {path}")
    for role, pin in declared.items():
        if role not in policy.pins or pin != policy.pins[role]:
            raise ValueError(f"Different declared checkpoint model for {role}: {path}")
    for field, role in _FIELDS.items():
        if role not in policy.roles or field not in result:
            continue
        pin = policy.pins[role]
        if role in declared and declared[role] != pin:
            raise ValueError(f"Different declared checkpoint model for {role}: {path}")
        observed = result[field]
        if observed != pin or role not in declared:
            warn_checkpoint_difference(
                observed,
                pin,
                role=role,
                path=path,
                context=context
                + (
                    "; legacy receipt declared pin assumed"
                    if role not in declared
                    else ""
                ),
            )
        result[field] = pin
    return result


def receipt_digest_role(field: str) -> str:
    return _FIELDS.get(field, field)
