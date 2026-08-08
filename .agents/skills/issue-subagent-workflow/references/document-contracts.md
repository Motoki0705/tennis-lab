# Document and machine-artifact contracts

`.agents/skills/issue-subagent-workflow/scripts/issue_task_schema.py` is the single machine-readable source for paths, exact headings, non-empty sections, allowed `None` sections, cycle metadata, hashes, and candidate metadata. Templates and agent instructions must follow that schema.

Every completed Markdown artifact records the current Issue, attempt, and `Status: COMPLETE`. Implementation-stage artifacts also record the exact test cycle. Candidate-bound artifacts record `Candidate SHA-256`.

## Frozen Issue

- `issue.json` is canonical raw Issue JSON.
- `issue.md` is deterministically rendered from it.
- `state.toml` records the raw payload hash and rendered-document hash.
- The checklist is normalized to ordered `AC-001`, `AC-002`, ... rows.

Manual changes to Issue prose, title, URL, number, scope, prohibitions, or checklist are rejected unless the upstream Issue is refreshed.

## `00-feasibility/feasibility.md`

Required sections cover allowed/prohibited changes, baseline and required checks, breaking-change impact, exact AC feasibility matrix, conflicts, verdict, and required resolution. PASS requires all AC rows `FEASIBLE` and no unresolved conflict. BLOCKED requires a `BLOCKED` or `UNKNOWN` row plus concrete conflict and resolution evidence.

## `01-exploration/exploration.md`

Required exact sections cover Issue interpretation, files/symbols, real execution paths, data/config/interface contracts, tests/fixtures, invariants, impact risks, unresolved questions, and a fact/inference/unknown evidence table. A path list alone is insufficient.

## `02-planning/plan.md` and `checks.json`

`plan.md` preserves every AC row and records planned files, user-selected topology and ownership, independent test unit, canonical command IDs, ordered execution, validation, non-goals, and risks.

`checks.json` contains the exact executable command definitions. Natural-language commands in Markdown are descriptive only; the JSON manifest and stage-result files are authoritative.

## `03-implementation/implementation.md`

Written only by the parent or explicit integrator after all Implementer handoffs. It records ownership, files/symbols, behavior, deviations, focused results, remaining risks, and handoff. A default Implementer never races to overwrite it.

## `03-implementation/preflight.md`

Production preflight records candidate identity, scope, deterministic and focused checks, canonical command results, baseline comparison, exact outcomes, and PASS/RETURN. PASS is bound to `preflight-checks.json` and the current candidate fingerprint.

## `03-implementation/tests.md`

Owned by one independent Test Writer after production preflight PASS. It records the post-test candidate, exact AC-to-evidence rows, test changes, case coverage, canonical command results, outcomes, failures, untested risks, and PASS/RETURN. It may change allowed tests but never production.

## `03-implementation/seal.md`

Owned by the integrator after Tester PASS. No source or test edits occur while sealing. It records the Tester-candidate identity, changed-since-test inspection, complete scope inspection, seal-stage canonical results, exact outcomes, and PASS/RETURN. Seal PASS is required for validation.

## `04-validation/validation.md`

Owned by one Issue-only Validator. It records the sealed candidate, inspection scope, exactly one ordered verdict row per AC with substantive evidence, code/runtime evidence, repository-rule checks, final verdict, and concrete re-exploration questions on RETURN. The ordinary row verdicts are `PASS`, `FAIL`, and `NOT VERIFIED`.

For enforced validation containing AC-079, final verdict `PASS` uses the existing row grammar:
AC-001 through AC-078 remain `PASS`, while AC-079 must be exactly `NOT VERIFIED`. The AC-079
Evidence cell must literally name `post-Validator packaging`,
`capture-pr`, `captured evidence`, `packaging.md`, `finalize-pr`, and `final workflow check`.
Any other `FAIL` or `NOT VERIFIED` blocks PASS, and AC-079 may not be marked `PASS` before
packaging. Validator RETURN retains its ordinary requirement for at least one `FAIL` or
`NOT VERIFIED`; packaging-pending evidence never changes that final verdict.

## `05-packaging/pr-evidence.json` and `packaging.md`

`capture-pr` is run after Validator PASS from the checked-out final PR head. It generates `pr-evidence.json` from real GitHub metadata, every paginated changed-file page, and the status-check rollup, then stores the evidence digest in state.

`packaging.md` records the validated candidate, PR number/head, PR-evidence digest, complete diff-scope conclusion, required remote checks, packaging evidence, and PASS. For a checklist containing AC-079, its Packaging evidence section must literally name `AC-079`, `capture-pr`, `exact PR head`, `complete paginated files`, `candidate equality`, `required remote checks`, `packaging.md`, `captured evidence`, `finalize-pr`, and `final workflow check`. `finalize-pr` verifies the captured evidence, local HEAD, revision content fingerprint, complete revision file inventory, artifact metadata, and prospective complete state before completion.

## Machine-result JSON

`preflight-checks.json`, `test-checks.json`, and `seal-checks.json` are generated only by `run-check`. Each result binds a canonical invocation digest and exit code to one candidate fingerprint. Hand-written result rows cannot substitute for these files.
