# Spawn contracts

Use `fork_turns = "none"` whenever selecting a custom `agent_type`. Always provide a unique lowercase `task_name`.

## Explorer

```json
{
  "task_name": "issue_<n>_exploration",
  "message": "Read .codex/tasks/issue-<n>/issue.md. Investigate the repository for this issue and replace .codex/tasks/issue-<n>/01-exploration/exploration.md according to your role contract. This is attempt <attempt>. Do not modify code or GitHub.",
  "agent_type": "codebase_explorer",
  "fork_turns": "none"
}
```

For a RETURN attempt, append only the validator's concrete exploration questions. Do not pass the old plan as authority.

## Implementer

```json
{
  "task_name": "issue_<n>_implementation_<unit>",
  "message": "Implement work unit <unit>. Read issue.md, exploration.md, and plan.md under .codex/tasks/issue-<n>/. You own only: <files-or-modules>. Replace 03-implementation/implementation.md with the combined implementation record. Do not edit tests unless explicitly assigned.",
  "agent_type": "issue_implementer",
  "fork_turns": "none"
}
```

When multiple implementers run, assign disjoint files and either designate one artifact integrator or give each a named section in the same artifact. Never allow concurrent writes to the same artifact file.

## Test writer

```json
{
  "task_name": "issue_<n>_tests",
  "message": "Read issue.md and plan.md under .codex/tasks/issue-<n>/. Inspect the current code and diff, but do not read implementation.md. Own only these test paths: <paths>. Add meaningful independent tests and replace 03-implementation/tests.md.",
  "agent_type": "test_writer",
  "fork_turns": "none"
}
```

## Validator

The validator message is intentionally sparse:

```json
{
  "task_name": "issue_<n>_validation",
  "message": "Use .codex/tasks/issue-<n>/issue.md as the sole task specification. Independently inspect the current repository revision and replace .codex/tasks/issue-<n>/04-validation/validation.md with PASS or RETURN and evidence. Do not read any other file under .codex/tasks/issue-<n>/.",
  "agent_type": "issue_validator",
  "fork_turns": "none"
}
```

Do not include plan summaries, implementation claims, test summaries, expected verdicts, or prior validator conclusions.

### Validator child

A validator may spawn a child with a narrow question:

```json
{
  "task_name": "issue_<n>_validation_<question>",
  "message": "Use .codex/tasks/issue-<n>/issue.md as the sole task specification. Inspect only this question: <question>. Do not read other workflow artifacts. Return concrete file, symbol, command, and behavior evidence to the validator.",
  "agent_type": "codebase_explorer",
  "fork_turns": "none"
}
```
