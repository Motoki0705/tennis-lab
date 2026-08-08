# Completion hardening

These rules are enforced by schema-v5 scripts rather than relying on instruction compliance alone.

## Test authoring and final seal

Production preflight binds the candidate before independent test authoring. The Test Writer may add or update allowed tests, producing a new candidate fingerprint. After Tester PASS, the integrator performs a no-edit final seal and re-runs complete seal-stage canonical checks. Any content change after Tester PASS rejects seal PASS and requires retesting.

## Candidate and PR binding

Preflight, Tester, seal, Validator, and packaging fingerprints are separate state fields. Validation requires the sealed candidate; packaging requires the validated candidate.

After Validator PASS, create or update the delivery PR and let the final-head checks complete. Then check out the exact PR head and run `capture-pr`. It queries the real PR through `gh`, stores all paginated changed files and the complete status-check rollup in `pr-evidence.json`, and binds the evidence digest to state. `finalize-pr` verifies:

1. local HEAD equals the supplied PR head;
2. the revision's content fingerprint equals the validated candidate;
3. the captured complete paginated file list equals the revision diff;
4. required remote checks in captured evidence are PASS;
5. `pr-evidence.json`, its state digest, `packaging.md`, and current content agree.

History-only packaging is allowed when content fingerprint remains identical. Any content change invalidates downstream evidence.

## Two-stage completion

Validator PASS sets `status = "validated"`, `phase = "packaging"`, `verdict = "VALIDATED"`. The task is not complete yet. Only `finalize-pr` can set `status = "complete"` and `verdict = "PASS"`.

When the enforced checklist contains AC-079, Validator PASS requires AC-001 through AC-078 to be
`PASS` and AC-079 to be exactly `NOT VERIFIED` with the mandatory packaging-pending evidence
tokens defined by the validation document contract. Any other `FAIL` or `NOT VERIFIED` blocks
PASS, and legacy binding remains all-PASS. Validator RETURN still requires an ordinary failing or
unverified row; packaging-pending evidence does not promote it. The phase-aware AC-079 rule
authorizes only the validated/packaging transition and is not an early AC-079 PASS.

Before completion, `finalize-pr` also requires the Packaging evidence section to establish AC-079
using its exact contract tokens. It re-reconciles the captured complete paginated file inventory
with the final revision, in addition to checking the evidence digest, exact head, candidate
equality, required remote checks, packaging artifact, and prospective whole-task check.

A packaging failure leaves the task validated so the PR can be corrected without falsifying acceptance evidence. A content-changing correction requires returning to the applicable preflight/Tester/seal gates.

## Automatic artifact checks

Every mutating command validates its input artifact and state before writing. Completion and packaging build a prospective next state, validate it, run whole-task checks, and only then atomically replace `state.toml`. An error leaves prior state bytes unchanged.

## Canonical command mismatch

Agents never reconstruct authoritative commands from prose. `run-check` executes the exact manifest entry and generates the result JSON. A broader diagnostic failure may be reported, but it cannot force RETURN unless it independently proves an AC defect. A stale or altered canonical invocation is rejected mechanically.

## User topology and rollover

An explicit user request for one Implementer is compliant. It changes orchestration topology, not evidence gates. After Validator RETURN or several long cycles, use a fresh bounded session even with one Implementer; artifacts carry state more reliably and cheaply than inherited conversation logs.
