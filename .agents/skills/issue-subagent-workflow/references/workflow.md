# Workflow contract

## State machine

```text
exploration -> planning -> implementation -> validation
                                      validation PASS -> complete
                                      validation RETURN -> exploration
```

`RETURN` always restarts exploration. It does not jump directly to planning or implementation because the failure may expose a mistaken codebase model.

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

One logical artifact has one path. Replace files in place. Do not create attempt-, date-, final-, revised-, or v2-suffixed copies. Git history is the audit trail.

## Responsibilities

### Parent orchestrator

- Fetch and freeze the issue.
- Select task boundaries and spawn agents.
- Verify the explorer's high-impact claims by reading relevant code itself.
- Own `plan.md`, including acceptance mapping and file ownership.
- Resolve conflicts between implementer and test writer.
- Enforce the validator context firewall.
- Apply PASS/RETURN and create the PR only after PASS.

The parent should inspect at least the principal entry point, one complete call path, relevant configuration, and existing tests. It may inspect more whenever the explorer evidence is incomplete or surprising.

### Explorer

- Collect evidence without modifying source code, tests, configuration, or GitHub.
- Map code paths, contracts, tests, constraints, and unknowns.
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
3. Re-run the explorer, explicitly including the validator's unresolved questions in the explorer task message.
4. The parent rechecks evidence and rewrites the existing plan.
5. Re-run implementation, independent tests, and validation.

If the GitHub issue changed, refresh `issue.md`, update its hash, and treat the change as a new exploration attempt.
