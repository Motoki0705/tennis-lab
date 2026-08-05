# Spawn contracts

Use `fork_turns = "none"` whenever selecting a custom `agent_type`. Always provide a unique lowercase `task_name`.

## Default delegation policy

The parent must actively search for safe parallelism before each exploration or implementation wave.

- Partition broad work into the smallest useful independent questions or non-overlapping production ownership units.
- When at least two independent units exist, spawn at least two agents concurrently by default.
- Do not default to one broad assignment when several narrow assignments would produce independent evidence or shorten the critical path.
- Do not create redundant agents that answer the same question or own overlapping files.
- Record the delegation map before spawning: task name, question or ownership, expected output, and dependency wave.
- Wait for and join every agent in a wave before starting work that depends on the joined result.
- A configured concurrency limit is capacity, not a target. Use only the parallelism justified by independent work.
- If decomposable work is handled by a single Scout or Implementer, record the concrete reason: unavoidable ownership overlap, sequential dependency, artifact contention, or no additional independent evidence.

Keep one authoritative Explorer, Test Writer, and Validator per cycle so formal artifacts and final verdicts have a single owner. These agents may use bounded child agents where explicitly allowed below.

## Scout

```json
{
  "task_name": "issue_<n>_scout_<question>",
  "message": "Read .codex/tasks/issue-<n>/issue.md and answer only this bounded repository question: <question>. Locate candidate files, symbols, references, tests, and configuration with direct evidence. Do not modify files or GitHub. State ambiguities and whether formal exploration must broaden the scope.",
  "agent_type": "codebase_scout",
  "fork_turns": "none"
}
```

For cross-cutting Issues, prefer a wave of several narrow Scouts partitioned by subsystem, execution stage, configuration domain, or evidence question. Run multiple Scouts concurrently whenever their questions are independent. Scout output is advisory and never replaces formal exploration.

Examples of useful Scout partitions:

- one subsystem or package per Scout;
- configuration definition versus runtime consumption;
- entry points versus tests and fixtures;
- current implementation versus stale aliases and references;
- one independent AC evidence question per Scout.

## Explorer

```json
{
  "task_name": "issue_<n>_exploration",
  "message": "Read .codex/tasks/issue-<n>/issue.md, including the normalized AC checklist. Investigate the repository and replace 01-exploration/exploration.md. This is attempt <attempt>. Independently verify the joined Scout leads: <leads-or-none>. Do not modify code or GitHub.",
  "agent_type": "codebase_explorer",
  "fork_turns": "none"
}
```

Spawn one authoritative Explorer after all Scout tasks in the current wave have joined. The Explorer must independently verify Scout evidence rather than concatenate it. For validator RETURN, append only the Validator’s concrete exploration questions and affected AC IDs. Do not pass the old plan as authority.

## Implementer: initial cycle

```json
{
  "task_name": "issue_<n>_implementation_<unit>",
  "message": "Implement work unit <unit> for attempt <attempt>, test cycle <cycle>. Read issue.md, exploration.md, and plan.md. Own only: <files-or-modules>. Do not edit tests. Report code changes and handoff evidence to the designated artifact integrator.",
  "agent_type": "issue_implementer",
  "fork_turns": "none"
}
```

Before spawning Implementers, the parent must create an ownership matrix. When the plan contains two or more disjoint production units, spawn multiple Implementers concurrently by default. Assign each production file to exactly one owner for that wave.

Join every Implementer before starting the Test Writer. Designate one parent or Implementer as the artifact integrator; only that integrator replaces `implementation.md`. Parallel Implementers must not concurrently write the shared artifact.

## Implementer: tester RETURN

```json
{
  "task_name": "issue_<n>_implementation_retry_<cycle>_<unit>",
  "message": "Address work unit <unit> from the independent tester RETURN for attempt <attempt>, next test cycle <cycle>. Read issue.md, exploration.md, plan.md, and the current 03-implementation/tests.md RETURN findings. Affected AC IDs: <ids>. Failing commands and behavior: <failures>. Own only: <production-files>. Do not edit or weaken tests. Report the repair and evidence to the designated artifact integrator.",
  "agent_type": "issue_implementer",
  "fork_turns": "none"
}
```

Partition independent tester failures into non-overlapping repair units and spawn multiple retry Implementers concurrently when safe. Tester RETURN does not trigger Explorer or planning unless the failure reveals that the plan or codebase model is invalid rather than merely incomplete implementation. In that exceptional case, the parent must explicitly restart exploration rather than silently broaden implementation scope.

## Test writer

Run one authoritative Test Writer only after all implementation work is integrated.

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

Run one authoritative Validator. Its message is intentionally sparse:

```json
{
  "task_name": "issue_<n>_validation",
  "message": "Use .codex/tasks/issue-<n>/issue.md as the sole task specification. Treat its normalized AC checklist as authoritative. Independently inspect the current repository revision and replace validation.md. Emit exactly one ordered PASS, FAIL, or NOT VERIFIED row for every AC ID and a final PASS or RETURN. Do not read any other file under .codex/tasks/issue-<n>/.",
  "agent_type": "issue_validator",
  "fork_turns": "none"
}
```

Do not include plan, implementation, test-artifact summaries, expected verdicts, or prior Validator conclusions.

### Validator children

The Validator should actively partition independent evidence questions and spawn multiple bounded child Explorers concurrently when this improves coverage or latency. Suitable partitions include separate AC groups, subsystems, runtime commands, or static searches. Child agents collect evidence only; they do not decide the overall verdict.

```json
{
  "task_name": "issue_<n>_validation_<question>",
  "message": "Use issue.md as the sole task specification. Inspect only checklist item(s) <AC-IDs> and this evidence question: <question>. Do not read other workflow artifacts. Return concrete file, symbol, command, artifact, and behavior evidence. Do not decide the overall verdict.",
  "agent_type": "explorer",
  "fork_turns": "none"
}
```

The authoritative Validator joins all child evidence, independently checks load-bearing claims, writes the single `validation.md`, and owns the final PASS or RETURN.
