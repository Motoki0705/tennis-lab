"""Machine-readable artifact contracts for issue-subagent-workflow."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArtifactContract:
    """One formal workflow artifact contract."""

    path: str
    headings: tuple[str, ...]
    nonempty_headings: tuple[str, ...]
    allow_none_headings: tuple[str, ...] = ()
    requires_cycle: bool = False
    requires_issue_hash: bool = False
    requires_checklist_hash: bool = False
    requires_candidate: bool = False


ARTIFACT_CONTRACTS = {
    "feasibility": ArtifactContract(
        path="00-feasibility/feasibility.md",
        headings=(
            "## Allowed and prohibited changes",
            "## Required checks and baseline",
            "## Breaking-change and compatibility impact",
            "## Acceptance checklist feasibility",
            "## Constraint conflicts",
            "## Final feasibility verdict",
            "## Blocker resolution required",
        ),
        nonempty_headings=(
            "## Allowed and prohibited changes",
            "## Required checks and baseline",
            "## Breaking-change and compatibility impact",
            "## Acceptance checklist feasibility",
            "## Constraint conflicts",
            "## Final feasibility verdict",
            "## Blocker resolution required",
        ),
        allow_none_headings=(
            "## Constraint conflicts",
            "## Blocker resolution required",
        ),
        requires_issue_hash=True,
        requires_checklist_hash=True,
    ),
    "exploration": ArtifactContract(
        path="01-exploration/exploration.md",
        headings=(
            "## Scope and Issue interpretation",
            "## Relevant files and symbols",
            "## Entry points and execution paths",
            "## Data, configuration, and interface contracts",
            "## Existing tests and fixtures",
            "## Invariants and compatibility constraints",
            "## Risks and likely impact radius",
            "## Unresolved questions",
            "## Evidence table",
        ),
        nonempty_headings=(
            "## Scope and Issue interpretation",
            "## Relevant files and symbols",
            "## Entry points and execution paths",
            "## Data, configuration, and interface contracts",
            "## Existing tests and fixtures",
            "## Invariants and compatibility constraints",
            "## Risks and likely impact radius",
            "## Unresolved questions",
            "## Evidence table",
        ),
        allow_none_headings=("## Unresolved questions",),
    ),
    "plan": ArtifactContract(
        path="02-planning/plan.md",
        headings=(
            "## Acceptance checklist mapping",
            "## Planned files and symbols",
            "## Implementation topology and ownership",
            "## Independent test work unit",
            "## Canonical verification commands",
            "## Ordered execution plan",
            "## Validation strategy",
            "## Non-goals and prohibited changes",
            "## Risks, rollback, and open decisions",
        ),
        nonempty_headings=(
            "## Acceptance checklist mapping",
            "## Planned files and symbols",
            "## Implementation topology and ownership",
            "## Independent test work unit",
            "## Canonical verification commands",
            "## Ordered execution plan",
            "## Validation strategy",
            "## Non-goals and prohibited changes",
            "## Risks, rollback, and open decisions",
        ),
        allow_none_headings=("## Risks, rollback, and open decisions",),
        requires_issue_hash=True,
        requires_checklist_hash=True,
    ),
    "implementation": ArtifactContract(
        path="03-implementation/implementation.md",
        headings=(
            "## Assigned ownership",
            "## Files and symbols changed",
            "## Behavior implemented",
            "## Plan deviations and rationale",
            "## Commands and results",
            "## Known limitations and remaining risks",
            "## Handoff",
        ),
        nonempty_headings=(
            "## Assigned ownership",
            "## Files and symbols changed",
            "## Behavior implemented",
            "## Plan deviations and rationale",
            "## Commands and results",
            "## Known limitations and remaining risks",
            "## Handoff",
        ),
        allow_none_headings=(
            "## Plan deviations and rationale",
            "## Known limitations and remaining risks",
        ),
        requires_cycle=True,
    ),
    "preflight": ArtifactContract(
        path="03-implementation/preflight.md",
        headings=(
            "## Candidate identity",
            "## Changed scope",
            "## Deterministic policy checks",
            "## Focused checks",
            "## Canonical command results",
            "## Baseline comparison",
            "## Commands and exact outcomes",
            "## Final production preflight verdict",
            "## RETURN implementation findings",
        ),
        nonempty_headings=(
            "## Candidate identity",
            "## Changed scope",
            "## Deterministic policy checks",
            "## Focused checks",
            "## Canonical command results",
            "## Baseline comparison",
            "## Commands and exact outcomes",
            "## Final production preflight verdict",
            "## RETURN implementation findings",
        ),
        allow_none_headings=("## RETURN implementation findings",),
        requires_cycle=True,
        requires_candidate=True,
    ),
    "tests": ArtifactContract(
        path="03-implementation/tests.md",
        headings=(
            "## Candidate identity",
            "## Acceptance-checklist-to-test mapping",
            "## Tests added or changed",
            "## Normal, boundary, invalid, and regression cases",
            "## Canonical command results",
            "## Commands and exact outcomes",
            "## Failures encountered",
            "## Untested risks and reasons",
            "## Final test verdict",
            "## RETURN implementation findings",
        ),
        nonempty_headings=(
            "## Candidate identity",
            "## Acceptance-checklist-to-test mapping",
            "## Tests added or changed",
            "## Normal, boundary, invalid, and regression cases",
            "## Canonical command results",
            "## Commands and exact outcomes",
            "## Failures encountered",
            "## Untested risks and reasons",
            "## Final test verdict",
            "## RETURN implementation findings",
        ),
        allow_none_headings=(
            "## Tests added or changed",
            "## Failures encountered",
            "## Untested risks and reasons",
            "## RETURN implementation findings",
        ),
        requires_cycle=True,
        requires_checklist_hash=True,
        requires_candidate=True,
    ),
    "seal": ArtifactContract(
        path="03-implementation/seal.md",
        headings=(
            "## Candidate identity",
            "## Changed-since-test inspection",
            "## Canonical command results",
            "## Complete scope inspection",
            "## Commands and exact outcomes",
            "## Final candidate seal verdict",
            "## RETURN implementation findings",
        ),
        nonempty_headings=(
            "## Candidate identity",
            "## Changed-since-test inspection",
            "## Canonical command results",
            "## Complete scope inspection",
            "## Commands and exact outcomes",
            "## Final candidate seal verdict",
            "## RETURN implementation findings",
        ),
        allow_none_headings=("## RETURN implementation findings",),
        requires_cycle=True,
        requires_candidate=True,
    ),
    "validation": ArtifactContract(
        path="04-validation/validation.md",
        headings=(
            "## Inspection scope and revision",
            "## Acceptance checklist verification",
            "## Code evidence",
            "## Runtime and test evidence",
            "## Regression and repository-rule checks",
            "## Final verdict",
            "## RETURN exploration questions",
        ),
        nonempty_headings=(
            "## Inspection scope and revision",
            "## Acceptance checklist verification",
            "## Code evidence",
            "## Runtime and test evidence",
            "## Regression and repository-rule checks",
            "## Final verdict",
            "## RETURN exploration questions",
        ),
        allow_none_headings=("## RETURN exploration questions",),
        requires_issue_hash=True,
        requires_checklist_hash=True,
        requires_candidate=True,
    ),
    "packaging": ArtifactContract(
        path="05-packaging/packaging.md",
        headings=(
            "## Final candidate binding",
            "## Pull request identity",
            "## Complete paginated diff scope",
            "## Remote required checks",
            "## Packaging evidence",
            "## Final packaging verdict",
        ),
        nonempty_headings=(
            "## Final candidate binding",
            "## Pull request identity",
            "## Complete paginated diff scope",
            "## Remote required checks",
            "## Packaging evidence",
            "## Final packaging verdict",
        ),
        requires_candidate=True,
    ),
}

ARTIFACT_PATHS = {
    name: contract.path for name, contract in ARTIFACT_CONTRACTS.items()
}
REQUIRED_HEADINGS = {
    contract.path: contract.headings for contract in ARTIFACT_CONTRACTS.values()
}
