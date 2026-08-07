# Spawn contracts

Use `fork_turns = "none"` with custom agent types and a unique lowercase `task_name`. Parallelism is optional when the user explicitly requests one Implementer or sequential execution.

## Mandatory assignment footer

Append this exact footer to every `spawn_agent` assignment, including retries and bounded Validator child exploration:

```text
Communication mode: terminal-only.
Work silently in the child thread. Do not send commentary, milestone, percentage, command-in-progress, or routine progress messages to the parent.
Before the final response, contact the parent only for missing authority, an ownership collision, or a blocker that cannot be resolved inside the assigned scope.
Return exactly one compact final handoff when the assignment is complete.
```

Do not weaken or paraphrase the footer. Custom agent `developer_instructions` repeat the same contract so the policy is versioned and testable; the assignment footer remains mandatory because it sets the cadence for the concrete child turn.

An allowed pre-terminal escalation must identify the exact authority needed, conflicting owner/path, or unresolved blocker and stop. Newly discovered evidence, completed substeps, test output, elapsed time, and estimated completion are not escalation reasons; keep them in the child thread, task logs, or authoritative artifact.

## Scout

One bounded repository question, read-only. Return candidate files/symbols, direct evidence, nearby tests/configuration, ambiguity, and whether formal exploration must broaden. Do not perform mechanical whole-repository inventories that a deterministic script can produce.

## Explorer

One authoritative Explorer per attempt. Read the frozen Issue, independently verify Scout leads, replace only `exploration.md`, run `artifact-check ... exploration`, and return a compact handoff. After Validator RETURN, use a fresh bounded session by default.

## Implementer

Default Implementers modify only assigned production or explicitly allowed test ownership and return a compact handoff. They do **not** write `implementation.md`, `preflight.md`, `seal.md`, plans, or validation.

Before spawning, record ownership and topology in `plan.md`. In a one-Implementer topology, or when one worker is explicitly named integrator, the spawn message may grant `artifact_integrator = true`. Only that explicit integrator may join handoffs, write shared implementation artifacts, and run preflight/seal commands.

Retry messages contain only affected AC IDs, concrete failing command IDs and observations, ownership, and artifact paths. Do not restate the full Issue.

## Test Writer

Spawn only after production preflight PASS. It reads the frozen Issue, plan, `checks.json`, current code/diff, and public behavior; it does not read implementation or preflight narratives. It may change only allowed tests. It executes authorized test-stage checks through `run-check`, writes the post-test candidate fingerprint into `tests.md`, runs `artifact-check ... tests`, and returns PASS or RETURN. Any production edit is a contract violation and requires a fresh preflight.

## Validator

Spawn only after final-seal PASS. Give it the frozen Issue, sealed candidate fingerprint, and validation artifact path—no plan, implementation, test narrative, prior validation, or expected verdict. It independently inspects the current revision and complete diff, writes one exact AC matrix, runs `artifact-check ... validation`, and owns the final PASS/RETURN.

Bounded child Explorers may collect direct evidence for explicit AC IDs, but they do not decide the verdict.

## Waiting and context

Do independent parent work before waiting. Then join the active wave with one event-driven `wait_agent` call using `timeout_ms = 3_600_000` when supported, otherwise the maximum accepted timeout. Do not use shorter waits as polling intervals and do not call `list_agents` merely to check unchanged state.

`wait_agent` can return when any child communicates, so a one-hour timeout alone does not guarantee a one-hour sleep. Treat only `FINAL_ANSWER` as completion. For any nonterminal message that is not an allowed escalation, do not reply or summarize it; resume the same long wait immediately.

Join each completed child once. Raw logs and intermediate evidence stay in child threads or task logs. Parent summaries contain only terminal status, changed files or evidence, exact command IDs and outcomes, unresolved risks, and the next state transition.
