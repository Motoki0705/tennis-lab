"""Public task names and worker settings shown beside each child's status."""

from __future__ import annotations

import re
from typing import Any

AGENTS = {"codex", "claude"}
# Same literal shape the runner accepts for model pins/overrides (agent_settings).
SETTING = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,119}")


def title(task: dict[str, Any], guard: Any) -> dict[str, Any]:
    """The task name the parent CLI recorded locally, as a guarded text object."""
    value = task.get("title")
    if value is not None and (not isinstance(value, str) or "\n" in value):
        raise ValueError("invalid task title")
    shown: dict[str, Any] = guard.text(value, limit=200)
    return {**shown, "source": "parent_registry.task" if value is not None else None}


def remote_title(value: str, guard: Any) -> dict[str, Any]:
    """The name read from GitHub during an explicit refresh."""
    if not isinstance(value, str) or "\n" in value:
        raise ValueError("invalid issue title")
    shown: dict[str, Any] = guard.text(value, limit=200)
    return shown


def prefer_local_title(child: dict[str, Any]) -> None:
    """Use the GitHub name only when no local name was recorded.

    The local name is updated by parent sync/create/delegate. Preferring it keeps
    10-second local observations and explicit GitHub refreshes consistent.
    """
    github = child["github"]
    if child["title"]["status"] != "unavailable" or github.get("status") != "observed":
        return
    child["title"] = {**github["issue"]["title"], "source": "github.issue"}


def _setting(value: Any) -> str:
    if not isinstance(value, str) or SETTING.fullmatch(value) is None:
        raise ValueError("invalid worker setting")
    return value


def run_settings(record: dict[str, Any], source: str) -> dict[str, Any]:
    """Agent/model/effort of a run, preferring observed effective values.

    Effective values exist only after the CLI reported them (never inferred from
    defaults). Until then, the selected values of the run contract are shown and
    marked as such. Private effective sources (transcript paths) are not copied.
    """
    settings = record["contract"].get("settings")
    if not isinstance(settings, dict) or settings.get("agent") not in AGENTS:
        raise ValueError("invalid worker settings")
    selected = settings.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("invalid selected worker settings")
    runtime = record.get("runtime")
    effective = runtime.get("effective") if isinstance(runtime, dict) else None
    if effective is not None and not isinstance(effective, dict):
        raise ValueError("invalid effective worker settings")
    result: dict[str, Any] = {"agent": settings["agent"], "source": source}
    for field in ("model", "effort"):
        observed = (effective or {}).get(field)
        if observed is not None and (
            not isinstance(observed, dict)
            or observed.get("status") not in {"observed", "unavailable"}
        ):
            raise ValueError("invalid effective worker setting")
        if observed is not None and observed["status"] == "observed":
            result[field] = {"value": _setting(observed["value"]), "basis": "effective"}
        else:
            result[field] = {
                "value": _setting(selected.get(field)),
                "basis": "selected",
            }
    return result


def unstarted(task: dict[str, Any], *, basis: str) -> dict[str, Any]:
    """No run settings: model/effort are never guessed from configured defaults."""
    agent = task.get("agent") if task.get("delegated") else None
    if agent is not None and agent not in AGENTS:
        raise ValueError("invalid delegated agent")
    return {
        "agent": agent,
        "source": "parent_registry.task" if agent is not None else None,
        "model": {"value": None, "basis": basis},
        "effort": {"value": None, "basis": basis},
    }
