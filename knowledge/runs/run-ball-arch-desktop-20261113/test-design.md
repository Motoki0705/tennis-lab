# Desktop continuation validation design

Refs #1859; superseding agreement: issuecomment-6840801232. No validator evaluation requested.

- Contract: activation checkpointing is opt-in, adds no model/optimizer state keys, preserves outputs, gradients and RNG; old configs/artifacts restore with checkpointing off. One explicit scope is fixed before T1/T2 production. Recipe/source changes are rejected by the existing strict resume reader; only a documented, byte-audited desktop migration may adapt the original checkpoint.
- Regressions to detect: accidentally detached CNN/MSDA projection gradients, late-bound loop closure recomputing the wrong chunk, dropout consuming a second RNG stream, ignored scope, custom-op graph break inside an incompatible checkpoint span, and silent resume with different memory policy.
- Existing nodes: unit/tasks/ball_detection/models/mdd/test_multiscale_query.py (layout and initialization), test_x3d_csn.py::test_chunked_encoder_matches_full_outputs_and_gradients; integration/tasks/ball_detection/test_multiscale_query.py (official CUDA dtype/chunk/Inductor); unit/tasks/ball_detection/checkpoints/test_temporal.py (strict full resume).
- Add small CPU attention-boundary tests using a declared differentiable test double only at the CUDA boundary, comparing every parameter gradient and RNG. Extend real CUDA tests for checkpoint scopes, BF16, dropout and compilation. Full T64/720p checks run separately on the real GPU through the shared queue with the allocator cap and host RAM guard.
- Run adapter tests cover path-only/state-exact migration, immutable policy, stop-on-failure serial order, hardlinks and write budget. Probe updates are isolated and never counted as scientific updates.
- Cost: two supported checkpoint scopes, each with tiny CPU tests; measure collection and durations. GPU qualification/probes run sequentially with resource=all; record cold compilation separately. Full-size eager/no-checkpoint reference may fail under the allocator cap; record this explicitly and compare independent sequential subgraphs if necessary without changing scientific shapes.
- Numerical contract: eager BF16 full-shape component VJP and compiled FP32 output/all-parameter gradients establish replay equivalence. Inductor fuses checkpointed BF16 graphs differently; compiled BF16 checks require close outputs and every finite gradient, without claiming elementwise gradient equality across those distinct mixed-precision graphs.
- Lifecycle: worker acceptance ends after ten finite, consecutive updates following T1 cold compilation. The parent owns subsequent validation/unfreeze/checkpoint milestones. ExecStopPost must produce a local terminal event even when the service process exits abnormally; a failed scientific job must not be reported as success merely because queue serve returned zero.
- Local tensor eviction is allowed only after an exact matching immutable snapshot and independent state-equal readback, plus a published retirement ledger. Failure to publish the ledger must leave all local tensors intact.

## D: physical-space guard and attempt-3 recovery (2026-11-13)

- Contract: stop before CUDA work and while running when `/mnt/d` statvfs available bytes are below 10 GiB; exactly 10 GiB is allowed. Keep the existing 4 GiB host-available-RAM limit and 80% CUDA allocator. WSL filesystem growth is telemetry only, even when it exceeds 20 GB. Missing/failed capacity measurements stop explicitly.
- Regressions: measuring WSL `/` instead of D:, using `f_bfree` instead of `f_bavail`, retaining the old 19 GB growth cutoff, ignoring a threshold crossed after startup, silently losing the observer thread on an OSError, or failing to restore the SIGTERM handler.
- Layer/reuse: add CPU-only `tests/unit/tasks/ball_detection/training/test_desktop_guard.py`, following the knowledge-script loader pattern in `test_sequential_recovery.py`; retain the 13 adapter state/serial/notification nodes. Synthetic resource readings and an intercepted termination signal exercise the real observer; no GPU allocation or real signal is needed.
- Cost: fewer than 12 new CPU cases, each bounded by a 2-second event wait; no new model/GPU sweep. Run the selected tests with durations and the changed-file pre-commit hooks, then target-HEAD CI Python.
- Recovery: independently restore the uploaded 6,000 generation with its pinned completion-marker digest, compare every file and complete checkpoint state against the stopped local stage, and validate the fixed report/predictions. Adopt it only after all checks pass; otherwise select the last verified 5,500 pointer. The runner's completed-budget resume must perform zero additional T1 updates and preserve the existing validation. Observe only the first ten updates of the next running stage.

## RAM admission and attempt-4 recovery (2026-11-13)

- Contract: before each optimizer update, wait without a deadline while host available is below 6 GiB; exactly 6 GiB admits the update. Keep the last-resort stop strictly below 2 GiB throughout the active stage and the D: stop strictly below 10 GiB. A wait longer than 30 minutes emits one notice. Waits must not consume sampler positions, model/optimizer/RNG state or update timing.
- Telemetry: append start/resume/interruption/30-minute events with epoch time, monotonic duration, available bytes, and the training parent plus recursive DataLoader/other child RSS. Every new train.jsonl row contains process-tree RSS, parent/child RSS, available RAM and that update's admission wait seconds. RSS is a sum, including shared pages; it is not PSS or a unique physical allocation.
- Reuse/extension: extend the existing CPU guard nodes for the new 2/6 GiB boundaries, deterministic injected clocks for unlimited wait and one 30-minute notice, exception paths and process-tree traversal including children started by non-main threads. Extend the existing actual temporal-training integration fixture to prove the optional hook preserves model/AdamW/RNG/schedule/sampler and excludes admission time from seconds. Exercise observer failure before update 2 to ensure no second update or false completion occurs.
- Recovery: retain strict read_resume. Permit only a pinned old/new SHA-256 transition of training/temporal/runner.py for the explicit nonnumerical observer hooks, rejecting every unapproved source change. Extend migration tests and compare the actual independently restored update-1,000 full state before accepting it. Old rows without RSS stay unchanged; only new rows get telemetry.
- Cost: about ten added CPU guard/migration cases plus one small real-model integration case (three-update fixture, no GPU test sweep). Run only the affected contracts with durations, then pre-commit and target-HEAD CI Python. GPU acceptance is 11 real resumed updates (10 after the compile update), followed by no progress polling or recurring follow-up.
## Attempt 5: relocated ext4 data volume and independent capacity guard

Parent instruction 2026-11-14 adds a second 10 GiB minimum for the mounted
`/mnt/wsl/tennis-data` volume. Keep D:, RAM admission/emergency, allocator and
the numerical training recipe fixed. Missing/unreadable E must stop explicitly.

Extend `test_desktop_guard.py` existing statvfs, exact-boundary, pre-CUDA,
background-termination and admission-wait cases for E. Distinguish D/E readings
so reading D twice, ignoring E, using free instead of available blocks, accepting
one byte below the limit, or continuing after an E measurement error fails.
Add one missing-mount case; do not duplicate numerical-model tests for guard-only
changes. Reuse `test_state.py` and `test_serial.py` for strict resume and lifecycle.
Expected guard cases: 24 (8 additional); measure actual collection and durations.
Use injected readings/Event synchronization, without real sleeps in tests.

Full dataset integrity uses the existing hardlink rebuild: all 2,925 payloads,
157,028,755,547 bytes, pinned distribution/SHA and source/destination inode pairs.
Record mount identity, reused environment/native binary and compiler cache.
The GPU acceptance is the actual restored update-1,000 production state, first
compile update plus ten further finite continuous updates. Capture per-update
data-wait and wall seconds, resource peaks/minima, full-state restore and exact
committed log prefix. Do not compare GPU timing with the historical L4 benchmark.
