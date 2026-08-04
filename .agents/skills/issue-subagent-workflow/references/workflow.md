# Workflow contract

## Preconditions

The GitHub Issue body must contain a concrete Markdown task-list acceptance checklist. The initializer normalizes each item to an ordered AC ID and fails closed when the checklist is absent, empty, or duplicated.

## State machine

```text
optional scouting -> exploration -> planning -> implementation -> validation
                                                   validation PASS -> complete
                                                   validation RETURN -> exploration
```

Scouting is advisory and does not change task state. `RETURN` always restarts formal exploration. It does not jump directly to planning or implementation because the failure may expose a mistaken codebase model.

An optional active `/goal` surrounds this state machine and keeps the parent working across turns. It does not replace or mutate phase transitions. See [goal integration](goal-integration.md).

## Artifact tree

```text
.codex/tasks/issue-<number>/
├── issue.md
├── state.toml
├── 01-exploration/exploration.md
├── 02-planning/plan.md
├── 03-implementation/implementation.md
├── 03-implementation/tests.md
└── 04-validation/validation.md
```

One logical artifact has one path. Replace files in place. Do not create attempt-, date-, final-, revised-, or v2-suffixed copies. Git history is the audit trail. Scout output is returned to the parent and is not a separate authoritative artifact.

## Acceptance checklist

- `issue.md` contains the frozen, normalized AC list and checklist hash.
- `state.toml` records the same checklist hash and item count.
- The parent maps every AC ID in `plan.md`.
- The test writer accounts for every AC ID in `tests.md`, even when proof must come from non-test evidence.
- The validator independently emits exactly one ordered verdict row for every AC ID.
- Source GitHub checkbox state is never accepted as implementation evidence.
- The state helper accepts PASS only when all AC verdicts are PASS and the final standalone verdict is PASS.

## Exploration routing

Use `codebase_scout` only for bounded, high-volume lookup work such as locating named symbols, direct references, nearby tests, configuration keys, or candidate entry points. Multiple scouts may run in parallel when their questions are independent.

Use `codebase_explorer` directly when any of the following applies:

- the Issue spans multiple packages or execution stages;
- control flow depends on registries, plugins, dynamic dispatch, Hydra or other configuration resolution;
- schemas, tensor shapes, coordinate systems, persistence formats, or public interfaces may change;
- code is being deleted, moved, or broadly refactored;
- the validator returned the task;
- a scout reports ambiguity, competing entry points, or an incomplete call path.

Formal exploration is mandatory before planning. The explorer must independently verify scout leads and write `exploration.md`; the parent must independently verify high-impact explorer claims.

## Responsibilities

### Parent orchestrator

- Fetch and freeze the Issue and reject it when it lacks a usable checklist.
- When `/goal` is active, preserve the full Issue objective and keep it active through validator RETURN cycles.
- Decide whether bounded scout work is useful and formulate independent questions.
- Select task boundaries and spawn agents.
- Verify the explorer's high-impact claims by reading relevant code itself.
- Own `plan.md`, including every AC mapping and file ownership.
- Resolve conflicts between implementer and test writer.
- Enforce the validator context firewall.
- Apply PASS/RETURN through the state helper and create the PR only after accepted PASS.
- Mark the outer goal complete only after the accepted workflow state and all goal deliverables exist.

The parent should inspect at least the principal entry point, one complete call path, relevant configuration, and existing tests. It may inspect more whenever the explorer evidence is incomplete or surprising.

### Scout

- Answer one narrow repository lookup question quickly.
- Locate candidate files, symbols, references, tests, and configuration with direct evidence.
- Distinguish verified matches from likely candidates and report what was not inspected.
- Return results to the parent only; do not write workflow artifacts or modify the repository.
- Recommend promotion to `codebase_explorer` when scope or uncertainty expands.

### Explorer

- Collect evidence without modifying source code, tests, configuration, or GitHub.
- Map code paths, contracts, tests, constraints, and unknowns.
- Treat scout results as unverified leads and independently confirm material claims.
- Write only `exploration.md`.
- Never decide the final plan.

### Implementer

- Modify only explicitly owned production files.
- Replace `implementation.md`.
- Avoid test ownership unless the parent deliberately assigns it.

### Test writer

- Independently derive tests from the Issue checklist and plan.
- Account for every AC ID.
- May inspect code and diff, but must not read `implementation.md`.
- Modify tests only unless a production seam is explicitly authorized.
- Replace `tests.md`.

### Validator

- Treat `issue.md` as the sole requirements source.
- Use the normalized AC checklist as the authoritative acceptance matrix.
- Ignore source checkbox state as evidence.
- Do not read other workflow artifacts or Issue comments.
- Inspect code, diff, behavior, configuration, artifacts, and tests directly.
- Emit exactly one ordered PASS/FAIL/NOT VERIFIED row per AC ID.
- May delegate narrow inspections to built-in explorer children, passing only issue.md and explicit AC IDs.
- Write only `validation.md` with PASS or RETURN.

## Context firewall

The validator cannot literally run without system instructions, repository guidance, tool definitions, and an operational output path. "Issue only" therefore means the frozen Issue snapshot is the only task specification and workflow narrative supplied to it.

Spawn it without parent history. Do not summarize the plan or implementation in its message. Do not tell it what should pass. Child validators receive the same Issue snapshot plus explicit AC IDs and one narrow inspection question.

## RETURN discipline

On RETURN:

1. Run the state helper with `verdict ... RETURN`.
2. Preserve the frozen Issue unless the upstream Issue changed.
3. Keep an active `/goal` active; RETURN is ordinary iteration, not a blocker.
4. Re-run `codebase_explorer`, explicitly including the validator's unresolved questions in the explorer task message. Scouts may assist with independent narrow questions, but they do not replace formal exploration.
5. The parent rechecks evidence and rewrites the existing plan for every AC ID.
6. Re-run implementation, independent tests, and validation.

If the GitHub Issue changed, refresh `issue.md`, update the Issue and checklist hashes, and treat the change as a new exploration attempt.
