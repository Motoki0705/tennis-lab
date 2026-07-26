"""Explicit acceptance decisions layered over immutable alignment evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from src.synthetic_data_generation.scene_contract import ArtifactRef

ALIGNMENT_ACCEPTANCE_DECISION_SCHEMA = "court_alignment_acceptance_decision_v1"
USER_OVERRIDE_DECISION = "accepted_by_user_override"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class AlignmentAcceptanceDecision:
    """A user-authorized acceptance that preserves failed machine evidence."""

    schema: str
    decision_id: str
    created_at_utc: str
    decision: str
    authority: str
    reason: str
    provider_bundle_fingerprint: str
    selected_court_cluster: str
    selected_symmetry: str
    machine_validation_status: str
    failed_gates: tuple[str, ...]
    decision_source: ArtifactRef
    calibration: ArtifactRef
    holdout_validation: ArtifactRef
    git_revision: str
    git_dirty: bool
    command: str
    code_sha256: str

    def __post_init__(self) -> None:
        if self.schema != ALIGNMENT_ACCEPTANCE_DECISION_SCHEMA:
            raise ValueError(
                f"Unsupported acceptance decision schema: {self.schema!r}."
            )
        _path_safe_id(self.decision_id, name="decision_id")
        if self.decision != USER_OVERRIDE_DECISION:
            raise ValueError(
                "Only an explicit accepted_by_user_override decision is supported."
            )
        if self.authority != "user":
            raise ValueError("A manual alignment override requires user authority.")
        if not self.created_at_utc.strip():
            raise ValueError("created_at_utc must not be empty.")
        if not self.reason.strip():
            raise ValueError("Override reason must not be empty.")
        _sha256(
            self.provider_bundle_fingerprint,
            name="provider_bundle_fingerprint",
        )
        _path_safe_id(
            self.selected_court_cluster,
            name="selected_court_cluster",
        )
        _path_safe_id(self.selected_symmetry, name="selected_symmetry")
        if self.machine_validation_status != "rejected":
            raise ValueError(
                "A user override must preserve machine_validation_status='rejected'."
            )
        failed_gates = tuple(self.failed_gates)
        if not failed_gates or len(set(failed_gates)) != len(failed_gates):
            raise ValueError("failed_gates must be non-empty and unique.")
        for gate in failed_gates:
            _path_safe_id(gate, name="failed gate")
        if not self.git_revision.strip() or not self.command.strip():
            raise ValueError("Decision git revision and command must not be empty.")
        _sha256(self.code_sha256, name="code_sha256")
        object.__setattr__(self, "failed_gates", failed_gates)

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-compatible payload."""
        return {
            "schema": self.schema,
            "decision_id": self.decision_id,
            "created_at_utc": self.created_at_utc,
            "decision": self.decision,
            "authority": self.authority,
            "reason": self.reason,
            "provider_bundle_fingerprint": self.provider_bundle_fingerprint,
            "selected_court_cluster": self.selected_court_cluster,
            "selected_symmetry": self.selected_symmetry,
            "machine_validation_status": self.machine_validation_status,
            "failed_gates": list(self.failed_gates),
            "decision_source": self.decision_source.to_dict(),
            "calibration": self.calibration.to_dict(),
            "holdout_validation": self.holdout_validation.to_dict(),
            "git_revision": self.git_revision,
            "git_dirty": self.git_dirty,
            "command": self.command,
            "code_sha256": self.code_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse one strict acceptance-decision payload."""
        raw = _strict_mapping(
            value,
            keys={
                "schema",
                "decision_id",
                "created_at_utc",
                "decision",
                "authority",
                "reason",
                "provider_bundle_fingerprint",
                "selected_court_cluster",
                "selected_symmetry",
                "machine_validation_status",
                "failed_gates",
                "decision_source",
                "calibration",
                "holdout_validation",
                "git_revision",
                "git_dirty",
                "command",
                "code_sha256",
            },
        )
        failed_gates = _string_sequence(raw["failed_gates"], name="failed_gates")
        return cls(
            schema=_string(raw["schema"], name="schema"),
            decision_id=_string(raw["decision_id"], name="decision_id"),
            created_at_utc=_string(raw["created_at_utc"], name="created_at_utc"),
            decision=_string(raw["decision"], name="decision"),
            authority=_string(raw["authority"], name="authority"),
            reason=_string(raw["reason"], name="reason"),
            provider_bundle_fingerprint=_string(
                raw["provider_bundle_fingerprint"],
                name="provider_bundle_fingerprint",
            ),
            selected_court_cluster=_string(
                raw["selected_court_cluster"],
                name="selected_court_cluster",
            ),
            selected_symmetry=_string(
                raw["selected_symmetry"],
                name="selected_symmetry",
            ),
            machine_validation_status=_string(
                raw["machine_validation_status"],
                name="machine_validation_status",
            ),
            failed_gates=failed_gates,
            decision_source=ArtifactRef.from_dict(raw["decision_source"]),
            calibration=ArtifactRef.from_dict(raw["calibration"]),
            holdout_validation=ArtifactRef.from_dict(raw["holdout_validation"]),
            git_revision=_string(raw["git_revision"], name="git_revision"),
            git_dirty=_boolean(raw["git_dirty"], name="git_dirty"),
            command=_string(raw["command"], name="command"),
            code_sha256=_string(raw["code_sha256"], name="code_sha256"),
        )


def verify_machine_evidence(
    decision: AlignmentAcceptanceDecision,
    *,
    calibration: Mapping[str, object],
    holdout_validation: Mapping[str, object],
) -> None:
    """Require the decision to describe the immutable machine result exactly."""
    if calibration.get("status") != "fit_calibration_passed":
        raise ValueError("Override requires a passed fit calibration.")
    if holdout_validation.get("status") != "rejected":
        raise ValueError("Override requires a rejected holdout validation.")
    provider = holdout_validation.get("provider")
    if not isinstance(provider, Mapping):
        raise ValueError("Holdout validation provider record is missing.")
    if provider.get("bundle_fingerprint") != decision.provider_bundle_fingerprint:
        raise ValueError("Decision provider fingerprint differs from holdout evidence.")
    geometry = holdout_validation.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("Holdout validation geometry record is missing.")
    if geometry.get("selected_candidate_id") != decision.selected_court_cluster:
        raise ValueError("Decision court cluster differs from holdout evidence.")
    if geometry.get("selected_symmetry") != decision.selected_symmetry:
        raise ValueError("Decision symmetry differs from holdout evidence.")
    gate_results = holdout_validation.get("gate_results")
    if not isinstance(gate_results, Mapping):
        raise ValueError("Holdout validation gate results are missing.")
    actual_failed = tuple(
        sorted(str(name) for name, passed in gate_results.items() if passed is False)
    )
    if tuple(sorted(decision.failed_gates)) != actual_failed:
        raise ValueError(
            "Decision failed_gates differ from machine evidence: "
            f"declared={sorted(decision.failed_gates)}, actual={list(actual_failed)}."
        )


def publish_alignment_acceptance_decision(
    decision: AlignmentAcceptanceDecision,
    *,
    output_dir: Path,
) -> Path:
    """Atomically publish a fingerprinted decision without overwriting."""
    payload = decision.to_dict()
    fingerprint = _fingerprint(payload)
    manifest = {**payload, "artifact_fingerprint": fingerprint}
    destination = output_dir / f"{decision.decision_id}-{fingerprint[:16]}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=f".{decision.decision_id}-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                manifest,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to overwrite acceptance decision: {destination}"
            ) from exc
        temporary_path.unlink()
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def load_alignment_acceptance_decision(
    path: Path,
) -> tuple[AlignmentAcceptanceDecision, str]:
    """Load and fingerprint-verify one acceptance decision."""
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("Acceptance decision must be a JSON object.")
    declared = raw.get("artifact_fingerprint")
    payload = {
        str(key): item for key, item in raw.items() if key != "artifact_fingerprint"
    }
    decision = AlignmentAcceptanceDecision.from_dict(payload)
    expected = _fingerprint(decision.to_dict())
    if declared != expected:
        raise ValueError(
            "Acceptance decision fingerprint mismatch: "
            f"declared {declared}, computed {expected}."
        )
    return decision, expected


def _fingerprint(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _strict_mapping(value: object, *, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("Acceptance decision must be a mapping.")
    raw = {str(key): item for key, item in value.items()}
    missing = keys.difference(raw)
    extra = set(raw).difference(keys)
    if missing or extra:
        raise ValueError(
            "Acceptance decision fields mismatch; "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )
    return raw


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _string_sequence(value: object, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence.")
    result = tuple(_string(item, name=f"{name} item") for item in value)
    return result


def _path_safe_id(value: str, *, name: str) -> None:
    if _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be path-safe, got {value!r}.")


def _sha256(value: str, *, name: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
