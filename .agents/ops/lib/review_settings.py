"""Review-only policy and immutable choices shared by producers and consumers.

Historical verification uses the policy frozen in the request. An actual launch
checks the current policy or the explicit continuation pin for a saved session.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

FIELDS = {"model", "effort", "service_tier"}
LEGACY_FIXED = {"model": "gpt-6-astra", "effort": "xhigh", "service_tier": "default"}
SOURCES = {"settings", "parent_choice", "human_instruction"}
CONTINUATION_PAIRS_PIN = "REVIEW_CODEX_CONTINUATION_PAIRS"


def obj(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("review settings must be an object")
    return value


def exact(value: Any, keys: set[str]) -> dict[str, Any]:
    data = obj(value)
    if set(data) != keys:
        raise ValueError("invalid review settings fields")
    return data


def text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"review {name} must be nonempty")
    return value


def hex_digest(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid source review digest")
    return value


def run_id(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"parent-[1-9][0-9]*-[0-9a-f]{32}", value
    ):
        raise ValueError("invalid source review run ID")
    return value


def read_object(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        if len(pairs) != len({k for k, _ in pairs}):
            raise ValueError("duplicate review settings keys")
        return dict(pairs)

    def invalid(value: str) -> Any:
        raise ValueError(f"invalid JSON constant: {value}")

    return obj(
        json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique,
            parse_constant=invalid,
        )
    )


def validate_values(value: Any) -> dict[str, str]:
    data = exact(value, FIELDS)
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._:/-]*", text(data["model"], "model")):
        raise ValueError("invalid review model")
    if not isinstance(data["effort"], str) or data["effort"] not in {
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    }:
        raise ValueError("invalid review effort")
    if not isinstance(data["service_tier"], str) or data["service_tier"] not in {
        "default",
        "fast",
        "flex",
        "priority",
    }:
        raise ValueError("invalid review service tier")
    return data


def validate_policy(value: Any) -> dict[str, Any]:
    policy = exact(value, {"version", "mode", "fixed", "allowed_pairs"})
    if type(policy["version"]) is not int or policy["version"] != 1:
        raise ValueError("unknown review selection policy version")
    if not isinstance(policy["mode"], str) or policy["mode"] not in {"fixed", "select"}:
        raise ValueError("invalid review selection mode")
    fixed = validate_values(policy["fixed"])
    pairs = policy["allowed_pairs"]
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("review allowed pairs must be nonempty")
    seen: list[tuple[str, str]] = []
    for pair in pairs:
        pair = exact(pair, {"model", "effort"})
        validate_values({**pair, "service_tier": fixed["service_tier"]})
        seen.append((pair["model"], pair["effort"]))
    if seen != sorted(set(seen)):
        raise ValueError("review allowed pairs must be unique and sorted")
    if (fixed["model"], fixed["effort"]) not in seen:
        raise ValueError("fixed review settings are not allowed")
    return policy


def _parse_pairs(value: str) -> list[dict[str, str]]:
    pairs = []
    for raw in value.split(","):
        if ":" not in raw:
            raise ValueError("invalid review allowed pair")
        model, effort = raw.rsplit(":", 1)
        pairs.append({"model": model, "effort": effort})
    pairs.sort(key=lambda pair: (pair["model"], pair["effort"]))
    return pairs


def policy_from_pins(pins: dict[str, str]) -> dict[str, Any]:
    def pin(key: str) -> str:
        return text(pins.get(key), key)

    if pin("REVIEW_SELECTION_POLICY_VERSION") != "1":
        raise ValueError("unknown review selection policy version")
    return validate_policy(
        {
            "version": 1,
            "mode": pin("REVIEW_SELECTION_MODE"),
            "fixed": {k: pin("REVIEW_CODEX_" + k.upper()) for k in FIELDS},
            "allowed_pairs": _parse_pairs(pin("REVIEW_CODEX_ALLOWED_PAIRS")),
        }
    )


def load_policy(config: Path | None = None) -> dict[str, Any]:
    from .agent_settings import DEFAULT_CONFIG, read_pins

    return policy_from_pins(read_pins(config or DEFAULT_CONFIG))


def validate_candidates(value: Any) -> None:
    candidates = value
    if not isinstance(candidates, list):
        raise ValueError("handoff candidates must be a list")
    ids = []
    for raw in candidates:
        candidate = exact(
            raw, {"status", "source_run", "validation", "cross_findings", "prs"}
        )
        if candidate["status"] != "candidate_only":
            raise ValueError("handoff issues are candidates, not formal coverage")
        ids.append(run_id(candidate["source_run"]))
        validation = exact(
            candidate["validation"],
            {
                "source_request_sha256",
                "source_execution_sha256",
                "source_result_sha256",
                "runtime_sha256",
                "events_sha256",
                "final_sha256",
                "output_sha256",
                "current_prs",
            },
        )
        for key in validation.keys() - {"current_prs"}:
            hex_digest(validation[key])
        for key in ("cross_findings", "prs"):
            if not isinstance(candidate[key], list):
                raise ValueError("handoff candidate issues must be lists")
        for pr in candidate["prs"]:
            exact(pr, {"number", "findings", "evidence_waits"})
    if ids != sorted(set(ids)):
        raise ValueError("handoff candidate runs must be unique and sorted")


def candidate_references(selection: dict[str, Any]) -> list[dict[str, Any]]:
    """Read issue references from an already validated run or recovery choice."""
    reference = (
        selection
        if selection.get("kind") == "recover"
        else (selection.get("handoff") or {})
    )
    records: list[dict[str, Any]] = reference.get("candidates", [])
    return records


def validate_selection(value: Any) -> dict[str, Any]:
    selection = obj(value)
    if selection.get("kind") == "recover":
        exact(selection, {"kind", "source_run", "source_request_sha256", "candidates"})
        run_id(selection["source_run"])
        hex_digest(selection["source_request_sha256"])
        validate_candidates(selection["candidates"])
        return selection
    selection = exact(
        selection,
        {"kind", "policy", "selected", "specified_by", "reason", "human", "handoff"},
    )
    if selection["kind"] != "run":
        raise ValueError("unknown review selection kind")
    policy = validate_policy(selection["policy"])
    selected = validate_values(selection["selected"])
    if {k: selected[k] for k in ("model", "effort")} not in policy["allowed_pairs"]:
        raise ValueError("review selection is not an allowed pair")
    if selected["service_tier"] != policy["fixed"]["service_tier"]:
        raise ValueError("review service tier differs from policy")
    if policy["mode"] == "fixed" and selected != policy["fixed"]:
        raise ValueError("review selection differs from fixed settings")
    if (
        not isinstance(selection["specified_by"], str)
        or selection["specified_by"] not in SOURCES
    ):
        raise ValueError("unknown review selection source")
    if policy["mode"] == "select" and selection["specified_by"] == "settings":
        raise ValueError("select requires an explicit review model")
    text(selection["reason"], "model reason")
    human = selection["human"]
    if human is not None:
        exact(human, {"model", "effort", "reference"})
        text(human["reference"], "human reference")
        if human["model"] != selected["model"] or (
            human["effort"] is not None and human["effort"] != selected["effort"]
        ):
            raise ValueError("review selection contradicts human instruction")
    if (human is not None) != (selection["specified_by"] == "human_instruction"):
        raise ValueError("review human instruction/source mismatch")
    if selection["handoff"] is not None:
        handoff = exact(
            selection["handoff"],
            {
                "source_run",
                "source_request_sha256",
                "source_result_sha256",
                "scope",
                "candidates",
            },
        )
        run_id(handoff["source_run"])
        hex_digest(handoff["source_request_sha256"])
        hex_digest(handoff["source_result_sha256"])
        text(handoff["scope"], "handoff scope")
        validate_candidates(handoff["candidates"])
    return selection


def choose(
    policy: dict[str, Any],
    *,
    model: str | None = None,
    effort: str | None = None,
    reason: str | None = None,
    specified_by: str | None = None,
    human_model: str | None = None,
    human_effort: str | None = None,
    human_reference: str | None = None,
    handoff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_policy(policy)
    explicit = model is not None or effort is not None or human_model is not None
    if specified_by == "human_instruction" and human_model is None:
        human_model = model
        if human_effort is None:
            human_effort = effort
    if human_model is not None:
        if model is not None and model != human_model:
            raise ValueError("parent model contradicts human instruction")
        if effort is not None and human_effort is not None and effort != human_effort:
            raise ValueError("parent effort contradicts human instruction")
        model, effort = human_model, human_effort or effort
        explicit = True
    elif human_effort is not None or human_reference is not None:
        raise ValueError("human review model is required")
    if policy["mode"] == "select" and model is None:
        allowed = ", ".join(
            f"{pair['model']}/{pair['effort']}" for pair in policy["allowed_pairs"]
        )
        raise ValueError(
            "select requires --review-model and --model-reason-file "
            f"(allowed pairs: {allowed})"
        )
    selected_model = model if model is not None else policy["fixed"]["model"]
    efforts = [
        p["effort"] for p in policy["allowed_pairs"] if p["model"] == selected_model
    ]
    if effort is None:
        if policy["mode"] == "fixed":
            effort = policy["fixed"]["effort"]
        elif len(efforts) != 1:
            raise ValueError("review effort must name an unambiguous allowed pair")
        else:
            effort = efforts[0]
    source = (
        "human_instruction"
        if human_model is not None
        else "parent_choice"
        if explicit
        else "settings"
    )
    if specified_by is not None and specified_by != source:
        raise ValueError("review selection source contradicts supplied choice")
    if explicit and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError(
            "explicit review model requires --model-reason-file with a nonempty reason"
        )
    return validate_selection(
        {
            "kind": "run",
            "policy": policy,
            "selected": {
                "model": selected_model,
                "effort": effort,
                "service_tier": policy["fixed"]["service_tier"],
            },
            "specified_by": source,
            "reason": text(reason, "model reason")
            if explicit
            else reason or "agents.conf の fixed 設定を使用。",
            "human": {
                "model": human_model,
                "effort": human_effort,
                "reference": human_reference,
            }
            if human_model is not None
            else None,
            "handoff": handoff,
        }
    )


def _allows_continuation(pins: dict[str, str], saved: dict[str, str]) -> bool:
    raw = pins.get(CONTINUATION_PAIRS_PIN)
    if raw is None:
        return False
    pairs = _parse_pairs(raw)
    for pair in pairs:
        validate_values({**pair, "service_tier": saved["service_tier"]})
    if len(pairs) != len({(p["model"], p["effort"]) for p in pairs}):
        raise ValueError("review continuation pairs must be unique")
    return {k: saved[k] for k in ("model", "effort")} in pairs


def runner_selection(
    pins: dict[str, str], selection: Any, saved: dict[str, str] | None
) -> tuple[dict[str, str], bool]:
    """Resolve new launches and the exact selection from a previous run."""
    policy = policy_from_pins(pins)
    data = validate_selection(selection) if selection is not None else None
    if data is not None and data["kind"] != "run":
        raise ValueError("recovery request cannot launch an agent")
    if saved is not None:
        validate_values(saved)
        if data is not None and data["selected"] != saved:
            raise ValueError("resume review selection differs from the saved contract")
        pair = {k: saved[k] for k in ("model", "effort")}
        if pair not in policy["allowed_pairs"] and _allows_continuation(pins, saved):
            return dict(saved), True
    if data is None:
        if policy["mode"] != "fixed":
            raise ValueError("select review profile requires --review-selection-file")
        return dict(policy["fixed"]), False
    if data["policy"] != policy:
        raise ValueError("review selection policy changed since preparation")
    return dict(data["selected"]), False


def runtime_view(runtime: dict[str, Any]) -> dict[str, Any]:
    """Display observations even when they disagree, excluding private sources."""
    settings = obj(obj(runtime.get("contract", {})).get("settings", {}))
    selected = obj(settings.get("selected", {}))
    effective = obj(obj(runtime.get("runtime", {})).get("effective", {}))
    return {
        "selected": {key: selected.get(key) for key in sorted(FIELDS)},
        "effective": {
            key: {
                k: v
                for k, v in obj(effective.get(key, {})).items()
                if k in {"status", "value", "reason"}
            }
            for key in sorted(FIELDS)
        },
    }
