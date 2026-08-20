# Completion hardening

Schema-v5 scripts enforce these invariants.

## Candidate and seal

Preflight PASS binds the pre-test candidate; allowed Test Writer edits may create another fingerprint, which Tester PASS binds. The no-edit Seal Reviewer reruns every seal-stage check. Any post-Tester content change rejects the seal and requires fresh Preflight/Test.

Preflight, test, seal, validation, and packaging have separate state bindings. Validation equals the seal; packaging equals the validated candidate. History-only packaging may preserve the content fingerprint; content changes invalidate downstream evidence.

## PR-bound completion

After Validator PASS, create/update the PR, finish final-head checks, check out that head, then `capture-pr`. Through `gh`, it stores real metadata, all paginated files, and the complete status rollup in `pr-evidence.json`, binding its digest to state.

`finalize-pr` verifies local HEAD=supplied head, revision content=validated candidate, captured files=revision diff, required remote checks=PASS, and agreement among evidence JSON, state digest, `packaging.md`, and current content.

Validator PASS sets `status = "validated"`, `phase = "packaging"`, `verdict = "VALIDATED"`; only `finalize-pr` sets complete/PASS. Packaging failure preserves validated state. Content-changing repair returns to applicable Preflight/Test/Seal gates.

## Atomicity and command authority

Every mutation first validates state/artifacts. Packaging/completion validate a prospective next state and whole task before atomically replacing `state.toml`; errors preserve prior bytes.

Never reconstruct canonical commands from prose: `run-check` executes the manifest entry and writes results. A broader diagnostic forces RETURN only if it independently proves an AC defect; stale/altered invocations fail mechanically.

A user-directed sole Implementer changes topology, not gates. After Validator RETURN or several long cycles, prefer a fresh bounded session; artifacts carry state more reliably/cheaply than inherited chat.
