# Completion hardening

Schema-v5 scripts enforce these invariants.

## Candidate and seal

Preflight PASS binds the pre-test candidate; allowed Test Writer edits may create another fingerprint, which Tester PASS binds. The no-edit Seal Reviewer reruns every seal-stage check. Any post-Tester content change rejects the seal and requires fresh Preflight/Test.

The plan and `checks.json` define mandatory minimum coverage, not the Test Writer's search ceiling. The Test Writer may add impacted tests and candidate-bound `AT-*` probes from Issue/public contracts, repository invariants, or baseline regressions. Every reported `AT-*` row must match `test-probes.json`; an all-AC-PASS candidate still returns when a supported adversarial probe fails. Unsupported or ambiguous oracles do not become product requirements.

The first Preflight is the only discovery pass. It may use only bounded diagnostic categories frozen in `plan.md` from the ACs and planned risks; the Reviewer does not design new categories during review. After one RETURN, the next Preflight is closure-only: verify the frozen findings, canonical checks, and direct repair regressions. A second consecutive Preflight RETURN requires `return-review`; do not start a third ordinary Preflight with another exploratory mutation set.

Seal is narrower than Preflight and Validator. It verifies Tester-candidate equality, no post-test content change, approved diff scope, repository rules, artifact completeness, and canonical seal results. It does not design semantic mutations, fuzz readers, reopen architecture, or search indefinitely for new failure categories. Any Seal RETURN requires `return-review`; after classification, every content repair restarts Preflight/Test before resealing.

Preflight, test, seal, validation, and packaging have separate state bindings. Validation equals the seal; packaging equals the validated candidate. History-only packaging may preserve the content fingerprint; content changes invalidate downstream evidence.

## PR-bound completion

After Validator PASS, create/update the PR, finish final-head checks, check out that head, then `capture-pr`. Through `gh`, it stores the actual remote base/head metadata, all paginated files, and the complete status rollup in `pr-evidence.json`, binding its digest to state. The PR file inventory uses GitHub's three-dot semantics: resolve `merge-base(baseRefOid, headRefOid)`, then compare that merge base to the PR head with an unfiltered `--no-renames` path inventory. A missing merge base or Git failure is an explicit error; it never falls back to an endpoint two-tree diff or an empty inventory.

`finalize-pr` verifies local HEAD=supplied head, revision content=validated candidate, captured files=the same merge-base-to-head PR inventory, required remote checks=PASS, and agreement among evidence JSON, state digest, `packaging.md`, and current content. Candidate and revision fingerprints remain a separate frozen-base identity: they stay anchored to `state.base_revision` and exclude `.codex/tasks/**`, while the PR inventory records the actual remote endpoints and remains unfiltered.

Validator PASS sets `status = "validated"`, `phase = "packaging"`, `verdict = "VALIDATED"`; only `finalize-pr` sets complete/PASS. Packaging failure preserves validated state. Content-changing repair returns to applicable Preflight/Test/Seal gates.

## Atomicity and command authority

Every mutation first validates state/artifacts. Packaging/completion validate a prospective next state and whole task before atomically replacing `state.toml`; errors preserve prior bytes.

Never reconstruct canonical commands from prose: `run-check` executes the manifest entry and writes results. On the first Preflight, a broader diagnostic may force RETURN only when its category was frozen in `plan.md` and it independently proves a frozen-AC defect. On closure Preflight or Seal, do not introduce a new diagnostic category; an incidentally discovered new category is classified through `return-review` and promoted into `plan.md`/`checks.json` before another cycle.

A user-directed sole Implementer changes topology, not gates. After `return-review`, Validator RETURN, or several long cycles, prefer a fresh bounded session; artifacts carry state more reliably/cheaply than inherited chat.
