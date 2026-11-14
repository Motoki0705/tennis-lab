"""Declarative unit, start, stop and verification contracts for operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Step:
    tool: Literal["systemctl", "dispatcher"]
    args: tuple[str, ...]


@dataclass(frozen=True)
class Target:
    name: str
    kind: Literal["service", "timer", "group"]
    units: tuple[str, ...] = ()
    enable_units: tuple[str, ...] = ()
    before_on: tuple[Step, ...] = ()
    off_steps: tuple[Step, ...] = ()
    verify_dispatcher: bool = False
    run_unit: str | None = None
    members: tuple[str, ...] = ()
    status_file: str | None = None


def service_target(name: str, stem: str) -> Target:
    unit = f"{stem}.service"
    return Target(
        name=name,
        kind="service",
        units=(unit,),
        enable_units=(unit,),
        off_steps=(Step("systemctl", ("disable", "--now", unit)),),
    )


def timer_target(name: str, stem: str, *, status_file: str | None = None) -> Target:
    timer, service = f"{stem}.timer", f"{stem}.service"
    return Target(
        name=name,
        kind="timer",
        units=(timer, service),
        enable_units=(timer,),
        off_steps=(
            Step("systemctl", ("disable", "--now", timer)),
            Step("systemctl", ("stop", service)),
        ),
        run_unit=service,
        status_file=status_file,
    )


TARGETS = {
    "dispatcher": Target(
        name="dispatcher",
        kind="service",
        units=(
            "tennis-lab-dispatch-supervisor.service",
            "tennis-lab-dispatch-judgement.timer",
            "tennis-lab-dispatch-judgement.service",
        ),
        enable_units=(
            "tennis-lab-dispatch-supervisor.service",
            "tennis-lab-dispatch-judgement.timer",
        ),
        before_on=(Step("dispatcher", ("enable",)),),
        off_steps=(
            Step(
                "systemctl", ("disable", "--now", "tennis-lab-dispatch-judgement.timer")
            ),
            Step("dispatcher", ("shutdown",)),
            Step("systemctl", ("stop", "tennis-lab-dispatch-judgement.service")),
            Step(
                "systemctl",
                ("disable", "--now", "tennis-lab-dispatch-supervisor.service"),
            ),
        ),
        verify_dispatcher=True,
    ),
    "report": timer_target("report", "tennis-lab-ops-report"),
    "triage": timer_target("triage", "tennis-lab-ops-triage"),
    "proposal-expiry": timer_target("proposal-expiry", "tennis-lab-improvement-expire"),
    # Existing merged-branch pruning. This is separate from worktree expiration.
    "cleanup": timer_target("cleanup", "tennis-lab-cleanup"),
    "agents": Target(
        name="agents", kind="group", members=("dispatcher", "report", "triage")
    ),
    "git-sync": timer_target(
        "git-sync", "tennis-lab-git-sync", status_file="git-sync/status.json"
    ),
    "worktree-cleanup": timer_target("worktree-cleanup", "tennis-lab-worktree-cleanup"),
}


def select(name: str | None) -> tuple[Target, ...]:
    """Expand groups without duplicating units or including maintenance in agents."""
    selected: dict[str, Target] = {}

    def add(key: str, ancestors: tuple[str, ...] = ()) -> None:
        if key in ancestors:
            raise ValueError(f"cyclic ops target group: {key}")
        target = TARGETS[key]
        if target.kind == "group":
            for member in target.members:
                add(member, (*ancestors, key))
        else:
            selected[key] = target

    for key in (name,) if name else TARGETS:
        add(key)
    return tuple(selected.values())
