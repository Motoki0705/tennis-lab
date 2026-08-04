# Spawn contracts

Use `fork_turns = "none"` whenever selecting a custom `agent_type`. Always provide a unique lowercase `task_name`.

## Scout

Use `codebase_scout` only for one bounded lookup question. It returns an advisory report to the parent and does not write a workflow artifact.

```json
{
  "task_name": "issue_<n>_scout_<question>",
  "message": "Read .codex/tasks/issue-<n>/issue.md and answer only this bounded repository question: <question>. Locate candidate files, symbols, references, tests, and configuration with direct evidence. Do not modify files or GitHub. State ambiguities and whether formal exploration must broaden the scope.",
  "agent_type": "codebase_scout",
  "fork_turns": "none"
}
```

Run multiple scouts in parallel only when the questions are independent. Do not ask a scout to produce `exploration.md`, reconstruct a cross-module architecture, decide the implementation plan, or resolve a validator RETURN by itself.

## Explorer

```json
{
  "task_name": "issue_<n>_exploration",
  "message": "Read .codex/tasks/issue-<n>/issue.md, including the normalized AC checklist. Investigate the repository for this Issue and replace .codex/tasks/issue-<n>/01-exploration/exploration.md according to your role contract. This is attempt <attempt>. Independently verify these optional scout leads before using them: <leads-or-none>. Do not modify code or GitHub.",
  "agent_type": "codebase_explorer",
  "fork_turns": "none"
}
```

For a RETURN attempt, append only the validator's concrete exploration questions and affected AC IDs. Do not pass the old plan as authority. A formal explorer run remains mandatory even when scouts were used.

## Implementer

```json
{
  "task_name": "issue_<n>_implementation_<unit>",
  "message": "Implement work unit <unit>. Read issue.md, exploration.md, and plan.md under .codex/tasks/issue-<n>/. The plan maps every AC item. You own only: <files-or-modules>. Replace 03-implementation/implementation.md with the combined implementation record. Do not edit tests unless explicitly assigned.",
  "agent_type": "issue_implementer",
  "fork_turns": "none"
}
```

When multiple implementers run, assign disjoint files and either designate one artifact integrator or give each a named section in the same artifact. Never allow concurrent writes to the same artifact file.

## Test writer

```json
{
  "task_name": "issue_<n>_tests",
  "message": "Read issue.md and plan.md under .codex/tasks/issue-<n>/. Account for every normalized AC ID. Inspect the current code and diff, but do not read implementation.md. Own only these test paths: <paths>. Add meaningful independent tests and replace 03-implementation/tests.md. For any AC item not provable by an automated test, identify the authoritative evidence the validator must collect.",
  "agent_type": "test_writer",
  "fork_turns": "none"
}
```

## Validator

The validator message is intentionally sparse. It references the normalized checklist but does not expose any workflow conclusions:

```json
{
  "task_name": "issue_<n>_validation",
  "message": "Use .codex/tasks/issue-<n>/issue.md as the sole task specification. Treat its normalized AC checklist as the authoritative acceptance matrix. Independently inspect the current repository revision and replace .codex/tasks/issue-<n>/04-validation/validation.md. Emit exactly one ordered PASS, FAIL, or NOT VERIFIED row for every AC ID and a final PASS or RETURN. Source checkbox state is not evidence. Do not read any other file under .codex/tasks/issue-<n>/.",
  "agent_type": "issue_validator",
  "fork_turns": "none"
}
```

Do not include plan summaries, implementation claims, test summaries, expected verdicts, or prior validator conclusions.

### Validator child

A validator may spawn a built-in explorer with a narrow checklist question:

```json
{
  "task_name": "issue_<n>_validation_<question>",
  "message": "Use .codex/tasks/issue-<n>/issue.md as the sole task specification. Inspect only checklist item(s) <AC-IDs> and this evidence question: <question>. Do not read other workflow artifacts. Return concrete file, symbol, command, artifact, and behavior evidence to the validator. Do not decide the overall verdict.",
  "agent_type": "explorer",
  "fork_turns": "none"
}
```
