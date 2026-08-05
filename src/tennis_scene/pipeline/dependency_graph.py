"""Dependency graph and resolution for tennis scene pipeline stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class Stage(StrEnum):
    """Pipeline stages."""

    COURT_KP = "court_kp"
    GVHMR = "gvhmr"
    BALL_DETECTION = "ball_detection"
    PLCS = "plcs"
    BLCS = "blcs"


class ResolutionPolicy(StrEnum):
    """Policy for handling unmet dependencies."""

    STRICT = "strict"
    LENIENT = "lenient"


@dataclass(frozen=True)
class StageSpec:
    """Stage metadata in dependency graph."""

    stage: Stage
    config_key: str
    depends_on: tuple[Stage, ...] = ()
    default_enabled: bool = True
    required: bool = False


@dataclass(frozen=True)
class ResolutionResult:
    """Result of dependency resolution."""

    enabled_order: tuple[Stage, ...]
    enabled_set: frozenset[Stage]
    requested_set: frozenset[Stage]
    disabled_reasons: dict[Stage, str]


class PipelineDependencyGraph:
    """Holds, validates, and resolves pipeline stage dependencies."""

    def __init__(
        self,
        specs: dict[Stage, StageSpec],
        policy: ResolutionPolicy = ResolutionPolicy.LENIENT,
    ) -> None:
        self.specs = specs
        self.policy = policy
        self.validate_graph()

    def validate_graph(self) -> None:
        """Validate graph structure."""
        for stage, spec in self.specs.items():
            if spec.stage != stage:
                raise ValueError(f"Stage spec key mismatch for {stage}")
            for dep in spec.depends_on:
                if dep not in self.specs:
                    raise ValueError(f"Unknown dependency: {stage} depends on {dep}")

        # Cycle detection by DFS.
        temp: set[Stage] = set()
        perm: set[Stage] = set()

        def visit(node: Stage) -> None:
            if node in perm:
                return
            if node in temp:
                raise ValueError(f"Cyclic dependency detected at stage: {node.value}")
            temp.add(node)
            for dep in self.specs[node].depends_on:
                visit(dep)
            temp.remove(node)
            perm.add(node)

        for stage in self.specs:
            visit(stage)

    def resolve_from_enabled(self, configured: Mapping[str, bool]) -> ResolutionResult:
        """Resolve explicitly enabled stages with dependency handling."""
        expected = {spec.config_key for spec in self.specs.values()}
        if set(configured) != expected:
            raise ValueError(
                "Enabled-stage keys must exactly match the dependency graph: "
                f"expected {sorted(expected)}, got {sorted(configured)}"
            )
        requested = {
            stage for stage, spec in self.specs.items() if configured[spec.config_key]
        }
        enabled = set(requested)
        disabled_reasons: dict[Stage, str] = {}

        changed = True
        while changed:
            changed = False
            for stage in list(enabled):
                missing = [
                    dep for dep in self.specs[stage].depends_on if dep not in enabled
                ]
                if not missing:
                    continue

                dep_names = ", ".join(dep.value for dep in missing)
                if self.policy == ResolutionPolicy.STRICT:
                    raise ValueError(
                        f"Stage '{stage.value}' requires missing dependency: {dep_names}"
                    )

                enabled.remove(stage)
                disabled_reasons[stage] = f"Missing dependency: {dep_names}"
                changed = True

        for stage, spec in self.specs.items():
            if spec.required and stage not in enabled:
                reason = disabled_reasons.get(stage, "Disabled by configuration")
                raise ValueError(
                    f"Required stage '{stage.value}' is not enabled. Reason: {reason}"
                )

        order = tuple(self._topological_order(enabled))
        self.validate_resolution(order, enabled)
        return ResolutionResult(
            enabled_order=order,
            enabled_set=frozenset(enabled),
            requested_set=frozenset(requested),
            disabled_reasons=disabled_reasons,
        )

    def validate_resolution(
        self, order: tuple[Stage, ...], enabled: set[Stage]
    ) -> None:
        """Validate resolved execution plan consistency."""
        if set(order) != enabled:
            raise ValueError("Resolved order and enabled set are inconsistent")

        seen: set[Stage] = set()
        for stage in order:
            missing = [dep for dep in self.specs[stage].depends_on if dep not in seen]
            if missing:
                dep_names = ", ".join(dep.value for dep in missing)
                raise ValueError(
                    f"Invalid execution order: {stage.value} before dependency {dep_names}"
                )
            seen.add(stage)

    def format_resolution_messages(self, result: ResolutionResult) -> list[str]:
        """Build human-readable messages for logs."""
        messages = [
            "Enabled stages: "
            + ", ".join(stage.value for stage in result.enabled_order)
        ]
        for stage, reason in sorted(
            result.disabled_reasons.items(), key=lambda item: item[0].value
        ):
            messages.append(f"Disabled stage '{stage.value}': {reason}")
        return messages

    def _topological_order(self, enabled: set[Stage]) -> list[Stage]:
        visited: set[Stage] = set()
        order: list[Stage] = []

        def dfs(stage: Stage) -> None:
            if stage in visited or stage not in enabled:
                return
            for dep in self.specs[stage].depends_on:
                dfs(dep)
            visited.add(stage)
            order.append(stage)

        for stage in self.specs:
            dfs(stage)
        return order


def build_default_dependency_graph(
    policy: ResolutionPolicy = ResolutionPolicy.LENIENT,
) -> PipelineDependencyGraph:
    """Construct the default dependency graph for tennis scene pipeline."""
    specs = {
        Stage.COURT_KP: StageSpec(
            stage=Stage.COURT_KP,
            config_key="court_kp",
            required=True,
        ),
        Stage.GVHMR: StageSpec(
            stage=Stage.GVHMR,
            config_key="gvhmr",
            default_enabled=True,
        ),
        Stage.BALL_DETECTION: StageSpec(
            stage=Stage.BALL_DETECTION,
            config_key="ball_detection",
            default_enabled=True,
        ),
        Stage.PLCS: StageSpec(
            stage=Stage.PLCS,
            config_key="plcs",
            depends_on=(Stage.COURT_KP, Stage.GVHMR),
            required=True,
        ),
        Stage.BLCS: StageSpec(
            stage=Stage.BLCS,
            config_key="blcs",
            depends_on=(Stage.COURT_KP, Stage.BALL_DETECTION),
            default_enabled=True,
        ),
    }
    return PipelineDependencyGraph(specs=specs, policy=policy)
