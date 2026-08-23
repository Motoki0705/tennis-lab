# Document and machine-artifact contracts

`.agents/skills/issue-subagent-workflow/scripts/issue_task_schema.py` is authoritative for artifact paths, exact headings, required/non-empty sections, allowed `None`, templates, and cycle/hash/candidate metadata. Completed Markdown records Issue, attempt, and `Status: COMPLETE`; implementation artifacts also record test cycle, candidate-bound artifacts `Candidate SHA-256`.

## Frozen Issue

`issue.json` is canonical raw JSON, `issue.md` its deterministic rendering, and `state.toml` stores both hashes. Checklist items become ordered `AC-001`, `AC-002`, ... rows. Until upstream refresh, manual changes to Issue title/URL/number/prose/scope/prohibitions/checklist are rejected.

## Contracts

| Artifact | Owner/timing; required content |
|---|---|
| `00-feasibility/feasibility.md` | Allowed/prohibited changes, baseline/required checks, breaking impact, exact AC feasibility matrix, conflicts, verdict, resolution. PASS: all ACs `FEASIBLE`, no unresolved conflict. BLOCKED: a `BLOCKED`/`UNKNOWN` row plus concrete conflict/resolution evidence. |
| `01-exploration/exploration.md` | Authoritative Explorer; Issue interpretation, files/symbols, real paths, data/config/interface contracts, tests/fixtures, invariants, impact risks, unresolved questions, fact/inference/unknown table. A path list fails. |
| `02-planning/plan.md` | Parent; every AC, planned files, topology/ownership, mandatory minimum test unit, canonical IDs, execution order, validation, non-goals, risks, and any bounded diagnostic categories permitted during the initial Preflight. Its test cases and risks do not cap the independent Test Writer. Repeated/late returns that expose a planning or impact-model gap require the strategy and `checks.json` baseline to be revised before another cycle. |
| `02-planning/checks.json` | Parent; exact mandatory baseline executable definitions. Markdown commands are descriptive; manifest and generated results are authoritative. The Test Writer never edits this file. |
| `03-implementation/implementation.md` | Parent/explicit integrator after all handoffs; ownership, files/symbols, behavior, deviations, focused results, risks, handoff. Default Implementers do not overwrite it. |
| `03-implementation/preflight.md` | Independent Preflight Reviewer after integration; candidate, scope, deterministic/focused checks, canonical results, baseline, exact outcomes, PASS/RETURN. The first review may perform one bounded AC-derived discovery pass. After RETURN, the replacement artifact is a closure record for the frozen findings, canonical checks, and repair-local regressions only. PASS binds `preflight-checks.json` and current fingerprint. No production/test edits; parent mutates state. |
| `03-implementation/tests.md` | Independent adversarial Test Writer after Preflight PASS; post-test candidate, exact AC-evidence rows, an independent risk model, contiguous `AT-*` perspectives with authority/oracle/machine evidence, test changes/cases, canonical and probe outcomes, failures, untested risks, PASS/RETURN. The plan is minimum coverage. Tests/fixtures only, never production or planning authority. |
| `03-implementation/seal.md` | Independent Seal Reviewer after Tester PASS, with no source/test edits; Tester candidate, changed-since-test/full-scope inventory, repository-rule and evidence-completeness inspection, seal results/outcomes, PASS/RETURN. Seal records no new semantic fuzzing or mutation campaign. Parent mutates state; PASS gates validation, RETURN requires `return-review`. |
| `04-validation/validation.md` | Issue-only Validator; sealed candidate/scope, one ordered PASS/FAIL/NOT VERIFIED row per AC with substantive evidence, code/runtime evidence, repository-rule checks, verdict, concrete RETURN questions. |
| `05-packaging/pr-evidence.json` | `capture-pr` at checked-out final PR head after Validator PASS; real metadata, every paginated file page, status rollup; digest stored in state. |
| `05-packaging/packaging.md` | Validated candidate, PR number/head, evidence digest, complete diff-scope conclusion, required remote checks, packaging evidence, PASS. `finalize-pr` verifies evidence, local HEAD, revision fingerprint, metadata, and prospective complete state. |

## Machine results

Only `run-check` creates `preflight-checks.json`, `test-checks.json`, and `seal-checks.json`; each binds one canonical invocation digest/exit code to one candidate fingerprint. Only `run-test-probe` creates `test-probes.json`; each `AT-*` result additionally binds its authority, authority source, exact invocation, outcome, and log. Hand-written rows cannot substitute.
