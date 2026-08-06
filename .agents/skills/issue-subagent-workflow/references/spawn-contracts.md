# Spawn contracts

Use `fork_turns = "none"` with custom agent types and a unique lowercase `task_name`. Parallelism is optional when the user explicitly requests one Implementer or sequential execution.

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

Join a complete wave once. Use a long/event-driven wait where available. Do not send progress messages unless scope, ownership, or evidence changed. Raw logs stay in child threads or task logs; parent summaries stay compact.
