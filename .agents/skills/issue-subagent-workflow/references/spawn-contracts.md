# Spawn contracts

Use `fork_turns = "none"` whenever selecting a custom `agent_type`. Always provide a unique lowercase `task_name`.

## Default delegation policy

The parent actively searches for safe parallelism, but subagent count is not a quality metric.

- Partition independent semantic questions or non-overlapping production ownership.
- Use at least two agents by default when at least two useful independent units exist.
- Do not create duplicate questions, overlapping file ownership, or shared artifact writers.
- Do not spawn an agent for a deterministic complete inventory that a repository command, AST script, or policy check can produce.
- Record the delegation map before spawning and join the complete wave before dependent work.
- Use one long or event-driven wait where available. Do not repeatedly issue short waits when agent state has not changed.
- Send a running agent another message only when constraints, ownership, or evidence changed.

Keep one authoritative Explorer, Test Writer, and Validator per cycle. Child responses are compact: status, direct evidence or changed files, commands and outcomes, unresolved risks, and artifact/log paths. Raw logs stay in the child thread or `.codex/tasks/issue-<n>/logs/`.

## Feasibility lookup

The parent owns the feasibility verdict. It may use deterministic commands or bounded Scouts before formal exploration to answer questions such as:

- Which tests encode behavior the Issue removes?
- Can every required check pass within the allowed write scope?
- Does an AC require editing a prohibited directory?
- Is a required external authority or environment unavailable?

These Scouts collect evidence only. They do not decide PASS or BLOCKED and do not modify formal artifacts.

## Scout

```json
{
  "task_name": "issue_<n>_scout_<question>",
  "message": "Read issue.md and answer only this bounded repository question: <question>. Return candidate files, direct evidence, relevant tests/configuration, ambiguity, and whether formal exploration must broaden the scope. Do not modify files, artifacts, or GitHub. Keep raw search output out of the parent summary.",
  "agent_type": "codebase_scout",
  "fork_turns": "none"
}
```

Use Scouts for bounded semantic lookups, not for mechanical repository-wide counts. Scout output is advisory and never replaces formal exploration.

## Explorer

```json
{
  "task_name": "issue_<n>_exploration",
  "message": "Read issue.md and replace 01-exploration/exploration.md for attempt <attempt>. Independently verify these joined Scout leads: <compact-leads-or-none>. Trace real entry points, contracts, tests, risks, and unresolved questions. Do not modify code or GitHub. Return only a compact handoff; the artifact is authoritative.",
  "agent_type": "codebase_explorer",
  "fork_turns": "none"
}
```

Spawn one authoritative Explorer after the Scout wave. For Validator RETURN, pass only the unresolved AC IDs and concrete questions, not the old plan or expected conclusion.

## Implementer: initial cycle

```json
{
  "task_name": "issue_<n>_implementation_<unit>",
  "message": "Implement work unit <unit> for attempt <attempt>, test cycle <cycle>. Read issue.md, exploration.md, and plan.md. Own only: <files-or-modules>. Respect the plan's allowed test ownership. Run focused checks and return changed files, commands/outcomes, and unresolved risks to the artifact integrator. Do not write shared artifacts or GitHub.",
  "agent_type": "issue_implementer",
  "fork_turns": "none"
}
```

Before spawning, create an ownership matrix and assign every production file to exactly one owner in that wave. Join every Implementer before preflight.

## Implementer: preflight or Tester RETURN

```json
{
  "task_name": "issue_<n>_implementation_retry_<cycle>_<unit>",
  "message": "Repair work unit <unit> for attempt <attempt>, next test cycle <cycle>. Read the authoritative artifacts. New failure bundle only: affected AC IDs <ids>; commands and observed behavior <failures>; source <preflight|tester>. Own only <files>. Do not weaken tests or broaden scope. Return the repair, focused evidence, and remaining risk to the integrator.",
  "agent_type": "issue_implementer",
  "fork_turns": "none"
}
```

Do not repeat the full Issue narrative in retry messages. Partition independent failures and use multiple repair agents only for non-overlapping ownership.

## Artifact integrator and preflight

The parent or one designated Integrator is the sole writer of `implementation.md` and `preflight.md`.

- Join all Implementers.
- Reduce child handoffs; do not paste every raw log into the formal artifact.
- Run deterministic policy checks, focused checks, then canonical required checks.
- On preflight RETURN, route concrete failures directly to Implementers and do not spawn the Test Writer.
- Record PASS or RETURN with `preflight-verdict`.

## Test writer

Spawn only after a matching preflight PASS.

```json
{
  "task_name": "issue_<n>_tests_<cycle>",
  "message": "Independently test attempt <attempt>, test cycle <cycle>. Read issue.md and plan.md. Inspect current code and diff; do not read implementation.md or preflight.md. Respect the Issue and plan's allowed test ownership. Replace tests.md with one exact ordered row per AC item and a standalone PASS or RETURN. On RETURN include exact commands, observed behavior, affected AC IDs, and actionable production findings. Return only a compact verdict summary.",
  "agent_type": "test_writer",
  "fork_turns": "none"
}
```

After completion, apply `test-verdict`. A second Tester RETURN triggers mandatory parent `return-review` before another preflight.

## Validator

```json
{
  "task_name": "issue_<n>_validation",
  "message": "Use issue.md as the sole task specification. Independently inspect the current revision and replace validation.md. Emit exactly one ordered PASS, FAIL, or NOT VERIFIED row per AC ID and a final PASS or RETURN. Do not read any other workflow artifact.",
  "agent_type": "issue_validator",
  "fork_turns": "none"
}
```

Do not include plan, feasibility, implementation, preflight, Tester summaries, expected verdicts, or prior Validator conclusions.

### Validator children

The Validator may spawn bounded child Explorers for independent AC evidence questions. Children receive only issue.md, explicit AC IDs, and one evidence question. They collect evidence but do not decide the final verdict. The authoritative Validator joins their compact summaries, verifies load-bearing claims, and owns `validation.md`.
