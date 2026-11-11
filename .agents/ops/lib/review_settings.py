"""Review-only policy and immutable choices shared by producers and consumers.

Historical verification uses the policy frozen in the request. Only an actual
launch compares it with the current configuration. No runtime is invoked here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

FIELDS = {"model", "effort", "service_tier"}
LEGACY_FIXED = {"model": "gpt-6-astra", "effort": "xhigh", "service_tier": "default"}
SOURCES = {"settings", "parent_choice", "human_instruction"}


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


def policy_from_pins(pins: dict[str, str]) -> dict[str, Any]:
    def pin(key: str) -> str:
        return text(pins.get(key), key)

    if pin("REVIEW_SELECTION_POLICY_VERSION") != "1":
        raise ValueError("unknown review selection policy version")
    pairs = []
    for raw in pin("REVIEW_CODEX_ALLOWED_PAIRS").split(","):
        if ":" not in raw:
            raise ValueError("invalid review allowed pair")
        model, effort = raw.rsplit(":", 1)
        pairs.append({"model": model, "effort": effort})
    pairs.sort(key=lambda pair: (pair["model"], pair["effort"]))
    return validate_policy(
        {
            "version": 1,
            "mode": pin("REVIEW_SELECTION_MODE"),
            "fixed": {k: pin("REVIEW_CODEX_" + k.upper()) for k in FIELDS},
            "allowed_pairs": pairs,
        }
    )


def load_policy(config: Path | None = None) -> dict[str, Any]:
    from .agent_settings import DEFAULT_CONFIG, read_pins

    return policy_from_pins(read_pins(config or DEFAULT_CONFIG))


def validate_selection(value: Any) -> dict[str, Any]:
    selection = obj(value)
    if selection.get("kind") == "recover":
        exact(selection, {"kind", "source_run", "source_request_sha256"})
        run_id(selection["source_run"])
        hex_digest(selection["source_request_sha256"])
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
            {"source_run", "source_request_sha256", "source_result_sha256", "scope"},
        )
        run_id(handoff["source_run"])
        hex_digest(handoff["source_request_sha256"])
        hex_digest(handoff["source_result_sha256"])
        text(handoff["scope"], "handoff scope")
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
        raise ValueError("select requires an explicit review model")
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


def launch_selection(selection: Any, pins: dict[str, str]) -> dict[str, Any]:
    data = validate_selection(selection)
    if data["kind"] != "run":
        raise ValueError("recovery request cannot launch an agent")
    if data["policy"] != policy_from_pins(pins):
        raise ValueError("review selection policy changed since preparation")
    return data


def verify_execution(request: dict[str, Any], execution: dict[str, Any]) -> None:
    if request["schema_version"] == 3:
        selection = validate_selection(request["review_selection"])
        if execution.get("review_selection") != selection:
            raise ValueError("review execution selection differs from request")


def runtime_summary(runtime: dict[str, Any]) -> dict[str, Any]:
    selected = validate_values(
        obj(obj(runtime.get("contract")).get("settings")).get("selected")
    )
    effective = obj(obj(runtime.get("runtime")).get("effective"))
    observed = {}
    for key in sorted(FIELDS):
        value = obj(effective.get(key))
        if value.get("status") == "observed":
            text(value.get("value"), "effective " + key)
            observed[key] = {"status": "observed", "value": value["value"]}
        elif (
            key == "service_tier"
            and value.get("status") == "unavailable"
            and value.get("value") is None
        ):
            observed[key] = {
                "status": "unavailable",
                "value": None,
                "reason": text(value.get("reason"), "effective tier reason"),
            }
        else:
            raise ValueError(f"review effective {key} is not confirmed")
        if (
            observed[key]["status"] == "observed"
            and observed[key]["value"] != selected[key]
        ):
            raise ValueError(f"review effective {key} differs from selection")
    return {"selected": selected, "effective": observed}


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


def validate_summary(value: Any) -> dict[str, Any]:
    summary = exact(value, {"selected", "effective"})
    normalized = runtime_summary(
        {
            "contract": {"settings": {"selected": summary["selected"]}},
            "runtime": {"effective": summary["effective"]},
        }
    )
    if summary != normalized:
        raise ValueError("invalid public review runtime settings")
    return summary


def verify_runtime(request: dict[str, Any], runtime: dict[str, Any]) -> None:
    if request["schema_version"] != 3:
        return
    selection = validate_selection(request["review_selection"])
    if selection["kind"] != "run":
        raise ValueError("recovery request has no local agent runtime")
    contract = obj(runtime.get("contract"))
    settings = obj(contract.get("settings"))
    if (
        contract.get("mode") != "read-only"
        or settings.get("agent") != "codex"
        or settings.get("profile") != "review"
    ):
        raise ValueError("review runtime permissions/profile differ from request")
    if (
        contract.get("review_selection") != selection
        or settings.get("selected") != selection["selected"]
    ):
        raise ValueError("review runtime selection differs from request")
    runtime_summary(runtime)
