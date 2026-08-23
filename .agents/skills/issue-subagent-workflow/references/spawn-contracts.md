# Spawn contracts

Use `fork_turns = "none"` exactly, a custom agent type, and unique lowercase `task_name`. Numeric values, inherited turn windows, `all`, and omission are noncompliant for initial/retry/post-compaction work, packaging repairs, all reviewers, Test Writers, Validators, and bounded Validator children. Context comes from frozen artifacts, paths, AC IDs, ownership, and focused failures. Parallelism is optional when the user requests one Implementer/sequential execution.

For any other `fork_turns`, do not accept its handoff or verdict: stop/discard the child and respawn the same bounded assignment fresh. This is mandatory for Reviewer, Test Writer, and Validator independence.

## Mandatory footer

Append verbatim to every assignment, including retries and bounded Validator children:

```text
Communication mode: terminal-only.
Work silently in the child thread. Do not send commentary, milestone, percentage, command-in-progress, or routine progress messages to the parent.
Before the final response, contact the parent only for missing authority, an ownership collision, or a blocker that cannot be resolved inside the assigned scope.
Return exactly one compact final handoff when the assignment is complete.
```

Do not weaken or paraphrase it. Custom-agent instructions repeat it for versioned, auditable testing; the assignment copy overrides generic cadence for the concrete turn. A permitted pre-terminal escalation names the exact missing authority, conflicting owner/path, or unresolved blocker, then stops. Evidence, milestones, test output, elapsed time, percentages, and estimates stay in the child thread, logs, or artifact.

## Roles

**Scout.** Read-only; answer one bounded repository question with candidate files/symbols, direct evidence, nearby tests/config, ambiguity, and whether exploration must broaden. Deterministic scripts handle mechanical whole-repository inventories.

**Explorer.** One authoritative session per attempt: read the frozen Issue, independently verify Scout leads, replace only `exploration.md`, run `artifact-check ... exploration`, return a compact handoff. After Validator RETURN, default fresh.

**Implementer.** Modify only assigned production or explicitly allowed tests; return a compact handoff. Do not write `implementation.md`, `preflight.md`, `seal.md`, plans, or validation. Record topology/ownership in `plan.md` first. Only a sole/named worker explicitly granted `artifact_integrator = true` may join handoffs and write `implementation.md`; no Implementer owns reviewer artifacts, stage checks, or verdicts. Retry prompts contain only affected ACs, failing command IDs/observations, ownership, and paths—not the full Issue.

**Preflight Reviewer.** Spawn after integration and complete `implementation.md`. Read frozen Issue, exploration, plan, `checks.json`, implementation, repository guidance, state, current code, and full diff; write only `preflight.md` plus canonical `run-check` results/logs; edit neither production nor tests. When the current state has no Preflight RETURN for the pending test cycle, run one discovery review: inspect full scope and use only bounded diagnostics derived from frozen ACs and planned risks. When the current state already records Preflight RETURN for that cycle, run closure review: read the existing `preflight.md`, freeze its findings, and verify only those findings, canonical checks, and direct repair regressions. Closure review must not invent a new mutation category or restart full exploratory review. Run `artifact-check ... preflight`, then PASS/RETURN with actionable findings. Parent alone calls `preflight-verdict`; first RETURN permits bounded repair and closure review, while second consecutive RETURN requires `return-review`.

**Test Writer.** Spawn after Preflight PASS. It reads the frozen Issue, repository guidance, public behavior, state, current code/full diff, plan, and `checks.json`—not implementation/preflight narratives. Treat the plan/checks as mandatory minimums, independently derive a bounded adversarial risk model, and inspect the changed impact radius for Issue-contract, public-contract, repository-invariant, and baseline-regression defects. Change only impacted tests/fixtures, never production, the plan, or `checks.json`. Record each executed independent perspective as contiguous `AT-*` evidence through `run-test-probe`; ambiguous expected behavior is missing authority, not a defect. PASS requires all ACs, baseline checks, and probes to pass. RETURN may cite a reproducible failed `AT-*` even when every AC passes. After a prior Tester RETURN, first close frozen findings, then limit new discovery to repair-local adversarial coverage. Record the post-test fingerprint in `tests.md`, run `artifact-check ... tests`, return PASS/RETURN. Any production edit requires fresh Preflight.

**Seal Reviewer.** Spawn exactly one `seal_reviewer` after Tester PASS. It reads frozen Issue, plan, `checks.json`, state, `tests.md`, repository guidance, current code/full diff—not implementation/preflight narratives—and first proves equality with the Tester PASS fingerprint. Write only `seal.md` plus canonical results/logs; edit neither source nor tests. Verify changed-since-test identity, changed/untracked scope, repository rules, evidence completeness, and every seal-stage canonical check. Do not conduct new semantic mutation testing, fuzzing, architecture exploration, or another open-ended Preflight. Run `artifact-check ... seal`, then PASS/RETURN. Parent alone calls `seal-verdict`; every RETURN requires `return-review`, and any content repair requires fresh Preflight/Test before resealing.

**Validator.** Spawn only after Seal PASS with frozen Issue and sealed identity, never prior narratives. The assignment may name validation artifact/helper paths, but omits plan, implementation/test narratives, prior validation, and expected verdict. Independently inspect current revision/full diff, write one exact AC matrix, run `artifact-check ... validation`, and own PASS/RETURN. Bounded child Explorers gather evidence for explicit ACs only; they never decide.

## Waiting

Do parent work before one event-driven `wait_agent` call with `timeout_ms = 3_600_000` when supported, otherwise the maximum. Do not use shorter waits as polling intervals or call `list_agents` merely to check unchanged state.

Any child message may wake the wait. Treat only `FINAL_ANSWER` as completion. For nonterminal non-escalations, neither reply nor summarize—resume the same wait. Join each child once. Raw evidence stays in child threads/task logs; parent summaries contain terminal status, changed files/evidence, exact command IDs/outcomes, risks, and next transition.
