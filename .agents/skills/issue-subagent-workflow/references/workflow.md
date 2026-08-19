# Normative workflow

This document is the single normative state-machine description. Scripts are the executable authority when prose and implementation differ.

## State machine

```text
feasibility -- BLOCKED ----------------------------------------------> blocked
     |
     PASS
     v
scouting -> exploration -> planning -> implementation
                                      ^        |
                                      |        v
                                      |  independent Preflight Reviewer
                                      |     | RETURN
                                      |     ` PASS
                                      |        v
                                      |  independent Test Writer
                                      |     | RETURN #1 -> repair
                                      |     | RETURN #2 -> return-review
                                      |     ` PASS
                                      |        v
                                      |  independent Seal Reviewer
                                      |     | RETURN -> repair/new cycle
                                      |     ` PASS
                                      |        v
                                      +---- validation
                                             | RETURN -> next attempt/exploration
                                             ` PASS
                                                v
                                      packaging / status=validated
                                             | PR head or checks fail -> remain validated
                                             ` finalize-pr PASS
                                                v
                                             complete
```

Preflight Reviewer RETURN does not spend a Tester cycle. Test Writer RETURN increments `test_cycle`. Seal Reviewer RETURN stays in implementation; any repair requires a fresh Preflight Reviewer and Test Writer cycle. Validator RETURN increments `attempt`, clears candidate evidence, and returns to formal exploration.

## Frozen specification

New tasks use schema v5. Initialization writes:

```text
issue.json              canonical raw GitHub payload
issue.md                deterministic human-readable rendering
state.toml              hashes, phase, verdicts, candidate bindings, PR binding
```

Every mutating command revalidates `issue.json`, its payload hash, the exact rendered `issue.md`, Issue number and URL, and the normalized checklist hash. Editing scope or prohibitions only in `issue.md` is detected and rejected.

## Candidate identity

`base_revision` is frozen at initialization. The candidate fingerprint hashes every changed or untracked path relative to that base, excluding `.codex/tasks/`. It is content-based, so history-only recommits or squashes may preserve the fingerprint.

State records distinct bindings:

- `preflight_candidate_sha256`
- `test_candidate_sha256`
- `sealed_candidate_sha256`
- `validation_candidate_sha256`
- `packaging_candidate_sha256`

Preflight Reviewer PASS may precede Test Writer changes. Tester PASS therefore binds a new candidate. Seal Reviewer PASS must match the Tester candidate and re-run complete canonical checks. Validation and packaging must match the sealed candidate. A stale artifact or machine-result file is rejected.

## Canonical command authority

`02-planning/checks.json` is the machine-readable command authority. Each check has:

- unique ID;
- exact `argv` array;
- repository-relative `cwd`;
- explicit environment additions;
- authorized stages: `preflight`, `test`, `seal`;
- required/optional status;
- AC authority list.

Run commands only through `manage_issue_task.py run-check`. The helper writes stage-specific JSON results and raw logs, including invocation digest, candidate fingerprint, exit code, and verdict. A changed argv, cwd, environment, candidate, or missing required check makes the stage verdict invalid.

## Artifact tree

```text
.codex/tasks/issue-<number>/
├── issue.json
├── issue.md
├── state.toml
├── 00-feasibility/feasibility.md
├── 01-exploration/exploration.md
├── 02-planning/
│   ├── plan.md
│   └── checks.json
├── 03-implementation/
│   ├── implementation.md
│   ├── preflight.md
│   ├── preflight-checks.json
│   ├── tests.md
│   ├── test-checks.json
│   ├── seal.md
│   └── seal-checks.json
├── 04-validation/validation.md
├── 05-packaging/
│   ├── pr-evidence.json
│   └── packaging.md
└── logs/
```

One logical artifact has one path. Formal artifacts are replaced in place. Raw logs are evidence attachments, not verdicts by themselves.

## User-directed topology

Parallelism is a latency optimization, not an acceptance criterion. The parent records the user-selected topology in `plan.md`. One Implementer may execute all work sequentially. This never weakens scope, canonical checks, candidate sealing, Validator independence, or PR-head binding.

Default Implementers do not write shared workflow artifacts. They return compact handoffs. The parent or one explicitly named implementation integrator writes only `implementation.md`; the independent Preflight Reviewer owns `preflight.md`, and the independent Seal Reviewer owns `seal.md`. In a one-Implementer topology, the parent may explicitly designate that Implementer as the implementation integrator.

## Automatic artifact gates

Mutating commands invoke targeted artifact checks internally:

- feasibility verdict -> feasibility artifact
- transition to planning -> exploration
- transition to implementation -> plan plus check manifest
- preflight verdict -> implementation, preflight, stage results, candidate
- test verdict -> tests, stage results, candidate
- seal verdict -> seal, stage results, Tester-candidate equality
- transition to validation -> all implementation artifacts and sealed candidate
- Validator verdict -> complete artifact set and sealed candidate
- capture-pr -> real PR metadata, complete paginated file list, final head, and status-check rollup
- finalize-pr -> captured evidence digest, packaging, local HEAD, revision content, and remote checks

Calling `artifact-check` manually provides earlier feedback but cannot bypass these checks.

## Context, communication, and retry discipline

- Prefer deterministic inventory scripts over agents for mechanical scans.
- Every `spawn_agent` call uses `fork_turns = "none"` exactly. Numeric or inherited turns invalidate the child handoff; respawn it fresh. For Preflight Reviewers, Test Writers, Seal Reviewers, and Validators this is an independence failure, not a cosmetic deviation.
- Supply child context through frozen artifacts, artifact paths, AC IDs, explicit ownership, and focused failure bundles. Never use parent-turn inheritance as an implicit context channel.
- Every child assignment ends with the exact `Communication mode: terminal-only.` footer in `spawn-contracts.md`. Custom agent instructions enforce the same boundary.
- Child agents do not stream commentary, milestones, percentage updates, or command-in-progress messages. Before their single terminal handoff, they may contact the parent only for missing authority, ownership collision, or an unresolvable in-scope blocker.
- Do independent parent work before waiting. Then call `wait_agent` with `timeout_ms = 3_600_000` when supported, otherwise the maximum accepted timeout.
- The timeout is an upper bound, not a polling interval. `wait_agent` may wake on any child communication. Treat only `FINAL_ANSWER` as completion; silently resume the long wait after any nonterminal message that is not an allowed escalation.
- Do not pair waiting with repeated `list_agents`, short-timeout polling, or `send_message` requests for status.
- Join each completed child once. Keep raw output in child threads or `logs/`; parent handoffs contain terminal status, changed files/evidence, exact command IDs, outcomes, and unresolved risks.
- Aggregate user-visible status at phase boundaries instead of relaying each child event.
- After Validator RETURN, start fresh bounded Explorer and Implementer sessions by default. The artifacts, not a long chat thread, carry state.
- Retry prompts contain only affected AC IDs, the new failure bundle, authorized ownership, artifact paths, and the mandatory terminal-only footer.
