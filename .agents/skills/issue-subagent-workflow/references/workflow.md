# Workflow contract

## State machine

```text
optional scouting -> exploration -> planning -> implementation -> validation
                                                   validation PASS -> complete
                                                   validation RETURN -> exploration
```

Scouting is advisory and does not change task state. `RETURN` always restarts formal exploration. It does not jump directly to planning or implementation because the failure may expose a mistaken codebase model.

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

## Exploration routing

Use `codebase_scout` only for bounded, high-volume lookup work such as locating named symbols, direct references, nearby tests, configuration keys, or candidate entry points. Multiple scouts may run in parallel when their questions are independent.

Use `codebase_explorer` directly when any of the following applies:

- the issue spans multiple packages or execution stages;
- control flow depends on registries, plugins, dynamic dispatch, Hydra or other configuration resolution;
- schemas, tensor shapes, coordinate systems, persistence formats, or public interfaces may change;
- code is being deleted, moved, or broadly refactored;
- the validator returned the task;
- a scout reports ambiguity, competing entry points, or an incomplete call path.

Formal exploration is mandatory before planning. The explorer must independently verify scout leads and write `exploration.md`; the parent must independently verify high-impact explorer claims.

## Responsibilities

### Parent orchestrator

- Fetch and freeze the issue.
- Decide whether bounded scout work is useful and formulate independent questions.
- Select task boundaries and spawn agents.
- Verify the explorer's high-impact claims by reading relevant code itself.
- Own `plan.md`, including acceptance mapping and file ownership.
- Resolve conflicts between implementer and test writer.
- Enforce the validator context firewall.
- Apply PASS/RETURN and create the PR only after PASS.

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

- Independently derive tests from the issue and plan.
- May inspect code and diff, but must not read `implementation.md`.
- Modify tests only unless a production seam is explicitly authorized.
- Replace `tests.md`.

### Validator

- Treat `issue.md` as the sole requirements source.
- Do not read other workflow artifacts or issue comments.
- Inspect code, diff, behavior, and tests directly.
- May delegate narrow inspections to built-in explorer children.
- Write only `validation.md` with PASS or RETURN.

## Context firewall

The validator cannot literally run without system instructions, repository guidance, tool definitions, and an operational output path. "Issue only" therefore means the issue snapshot is the only task specification and workflow narrative supplied to it.

Spawn it without parent history. Do not summarize the plan or implementation in its message. Do not tell it what should pass. Child validators receive the same issue snapshot plus one narrow inspection question.

## RETURN discipline

On RETURN:

1. Run the state helper with `verdict ... RETURN`.
2. Preserve the frozen issue unless the upstream issue changed.
3. Re-run `codebase_explorer`, explicitly including the validator's unresolved questions in the explorer task message. Scouts may assist with independent narrow questions, but they do not replace this formal exploration.
4. The parent rechecks evidence and rewrites the existing plan.
5. Re-run implementation, independent tests, and validation.

If the GitHub issue changed, refresh `issue.md`, update its hash, and treat the change as a new exploration attempt.
