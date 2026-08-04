# Spawn contracts

Use `fork_turns = "none"` whenever selecting a custom `agent_type`. Always provide a unique lowercase `task_name`.

## Scout

```json
{
  "task_name": "issue_<n>_scout_<question>",
  "message": "Read .codex/tasks/issue-<n>/issue.md and answer only this bounded repository question: <question>. Locate candidate files, symbols, references, tests, and configuration with direct evidence. Do not modify files or GitHub. State ambiguities and whether formal exploration must broaden the scope.",
  "agent_type": "codebase_scout",
  "fork_turns": "none"
}
```

Run multiple Scouts in parallel only for independent questions. Scout output is advisory and never replaces formal exploration.

## Explorer

```json
{
  "task_name": "issue_<n>_exploration",
  "message": "Read .codex/tasks/issue-<n>/issue.md, including the normalized AC checklist. Investigate the repository and replace 01-exploration/exploration.md. This is attempt <attempt>. Independently verify optional Scout leads: <leads-or-none>. Do not modify code or GitHub.",
  "agent_type": "codebase_explorer",
  "fork_turns": "none"
}
```

For validator RETURN, append only the Validator’s concrete exploration questions and affected AC IDs. Do not pass the old plan as authority.

## Implementer: initial cycle

```json
{
  "task_name": "issue_<n>_implementation_<unit>",
  "message": "Implement work unit <unit> for attempt <attempt>, test cycle <cycle>. Read issue.md, exploration.md, and plan.md. Own only: <files-or-modules>. Do not edit tests. Replace 03-implementation/implementation.md and record the supplied test-cycle number.",
  "agent_type": "issue_implementer",
  "fork_turns": "none"
}
```

Multiple Implementers may run in parallel only with disjoint production ownership. Join their work before starting the Test Writer, and designate one artifact integrator to avoid concurrent writes to `implementation.md`.

## Implementer: tester RETURN

```json
{
  "task_name": "issue_<n>_implementation_retry_<cycle>",
  "message": "Address the independent tester RETURN for attempt <attempt>, next test cycle <cycle>. Read issue.md, exploration.md, plan.md, and the current 03-implementation/tests.md RETURN findings. Affected AC IDs: <ids>. Failing commands and behavior: <failures>. Own only: <production-files>. Do not edit or weaken tests. Replace implementation.md for the new test cycle.",
  "agent_type": "issue_implementer",
  "fork_turns": "none"
}
```

Tester RETURN does not trigger Explorer or planning unless the failure reveals that the plan or codebase model is invalid rather than merely incomplete implementation. In that exceptional case, the parent must explicitly restart exploration rather than silently broaden implementation scope.

## Test writer

Run only after all implementation work is integrated.

```json
{
  "task_name": "issue_<n>_tests_<cycle>",
  "message": "Independently test attempt <attempt>, test cycle <cycle>. Read issue.md and plan.md. Inspect current code and diff, but do not read implementation.md. Do not modify production code. Own only these test paths: <paths>. Replace tests.md with one exact ordered row per AC item and a standalone PASS or RETURN. On RETURN, include exact failing commands, observed behavior, affected AC IDs, and actionable implementation findings.",
  "agent_type": "test_writer",
  "fork_turns": "none"
}
```

After the Test Writer completes, the parent must run:

```bash
python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py test-verdict .codex/tasks/issue-<n> <PASS|RETURN>
```

Do not enter validation until tester PASS is recorded.

## Validator

The Validator message is intentionally sparse:

```json
{
  "task_name": "issue_<n>_validation",
  "message": "Use .codex/tasks/issue-<n>/issue.md as the sole task specification. Treat its normalized AC checklist as authoritative. Independently inspect the current repository revision and replace validation.md. Emit exactly one ordered PASS, FAIL, or NOT VERIFIED row for every AC ID and a final PASS or RETURN. Do not read any other file under .codex/tasks/issue-<n>/.",
  "agent_type": "issue_validator",
  "fork_turns": "none"
}
```

Do not include plan, implementation, test-artifact summaries, expected verdicts, or prior Validator conclusions.

### Validator child

```json
{
  "task_name": "issue_<n>_validation_<question>",
  "message": "Use issue.md as the sole task specification. Inspect only checklist item(s) <AC-IDs> and this evidence question: <question>. Do not read other workflow artifacts. Return concrete file, symbol, command, artifact, and behavior evidence. Do not decide the overall verdict.",
  "agent_type": "explorer",
  "fork_turns": "none"
}
```
