# CPU cost baseline and test contracts

[日本語](baseline.ja.md) · [简体中文](baseline.zh-CN.md)

This records #1401, supporting #1402–#1405, and the runner scheduling follow-up #1523.
**The full CPU observation completed with failures; it is not a successful baseline.** This document does not
claim that the CI target or a performance improvement has been demonstrated.
The machine-readable run metadata, inventory, JUnit and phase events are the
authority for the exact source and collection; a failed or cancelled run remains
a failed or cancelled observation.

## Completed local observation

The complete CPU request selected **12,764 cases: 12,551 passed, 88 skipped and
125 failed (including 8 setup errors)**. The command finished in **5,790.692 s**,
with no missing inventory, phase or JUnit records. It is a **failed observation**,
not a successful performance baseline. The raw pytest result is 117 failures and
8 errors; the measurement summary groups both under `failed`.

Two workers ran on WSL2, Python 3.11.15, 12 visible logical CPUs (Ryzen 7 9700X),
23.47 GiB RAM, native thread limits of one and hidden CUDA. Other validation was
active. Cache condition is unknown. Collection completed at 78.673 s and final
reporting/shutdown took 25.273 s. Individual process max RSS was 1,531,464 KiB;
this is not the concurrent process-tree peak. A wall-clock correction of −183.694 s
occurred; elapsed measurements use the monotonic clock.

| File | Cases | Setup s | Call s |
|---|---:|---:|---:|
| merge gate | 161 | 1630.339 | 766.868 |
| board messages | 539 | 88.313 | 877.462 |
| parent session | 97 | 194.544 | 380.229 |
| ball refiner coordinate training | 9 | 4.012 | 559.781 |
| annotation screening | 86 | 168.074 | 254.512 |

This prioritizes merge fixture preparation and measured scheduling. Dense/RGB
contracts have independently measured construction reductions; SLCS/MDD smoke
costs do not justify a change. Outcome details are in the child documents. The
failed APIs/fakes are tracked in #1504, #1509 and #1511. The screening case exceeded
its 15-second deadline under load and passed an isolated rerun; it remains a
flakiness observation, not an erased failure.

Three alternating matched 12-test overhead pairs had plain median **5.992 s** and
instrumented median **6.243 s** (+0.251 s). They are startup-dominated and ran with
other local work; do not extrapolate this as full-suite percentage overhead. Event
serialization/write time in the full observation was 4.269 s, only a lower bound.
There is no successful local full-suite median or p95 yet. The n=1 failed run and
its full 5,790.692 s cost remain included. [Machine evidence](baseline-evidence.json)
records provenance, environment, inventory identity, phase/fixture costs and the
three overhead pairs. Same-source Actions comparisons follow integration.


## Reference Actions observations

The saved real retry fixture is
[ci_attempts.json](../../tests/support/development/ci_attempts.json). It retains
attempt-specific REST times, statuses, run/head identity and the source endpoints.

| Run | Attempt | Outcome | Apt | Test | All job runner time | Observed Python completion |
|---|---:|---|---:|---:|---:|---:|
| [39767861734](https://github.com/Motoki0705/tennis-lab/actions/runs/39767861734) | 1 | Cancelled | 1480 s | Skipped (0 s) | 1518 s | 1581 s from initial creation |
| Same run | 2 | Success | 25 s | 2966 s | 3027 s | 3030 s from retry start proxy |

Both attempts consumed **4545 runner seconds** in total. Counting only the retry
would discard 1518 seconds. The exact retry request time and cache condition were
not retained; retry completion uses the API's labelled start proxy. The successful
completion sample has n=1: median and nearest-rank p95 are 3030 s, with p95 reference
only. This is not a before/after comparison.

The parent's saved recent-main sample has 23 push runs: 1 success, 1 failure and
21 cancellations, at different revisions. Its latest saved successful run,
[40047681315](https://github.com/Motoki0705/tennis-lab/actions/runs/40047681315),
had Test 3338 s and Python runner time 3396 s. The saved current-main cancellation,
[40063537455](https://github.com/Motoki0705/tennis-lab/actions/runs/40063537455),
was interrupted during Test. Job/step data are not complete for all 23 runs.
No all-attempt runner total, warm-cache median, or credible p95 is claimed for
this mixed sample.

The historical local partial run described in #1401 ended with exit 2 after
14m28s, 3786 passed / 1 skipped. Its merge-gate testcase time sum, 641.41 s,
identifies a candidate; it is neither a complete baseline nor an estimate for
the current source or Actions runner.

## Candidate contracts and decision rules

The following exact files/functions define candidate groups. The measurement
summary reports their case counts and setup/call/teardown contribution. Fixture
time is already included in setup; model/CLI construction inside a test appears
in call. Summed worker elapsed time can overlap and is not predicted wall savings.

| Candidate / follow-up | Current tests | Repeated contract / possible cost | Boundaries that must survive | Decision and uncertainty |
|---|---|---|---|---|
| Merge-gate preparation / #1402 | `tests/e2e/agents_ops/test_merge_gate.py` (`gate_env`); `test_parent_session.py` (`parent_env`, `make_review_env`) | Per-case disposable git history, three PR worktrees, review/CLI preparation; historical partial sum 641.41 s | Human instruction and one-review consumption, current head/base/diff coverage, independent/dependent merge order, ordinary review, stale/failing merge recovery, dry run and auto-merge authority; current CI is observational | Prioritize fixture reuse/minimal scenario construction after current timing confirmation. Each test must retain private writable state and real git/CLI boundaries; sharing mutable session state is not justified by time alone. |
| Dense-mask matrix / #1403 | `test_block.py::test_dense_boundary_rejects_noncanonical_full_masks` (30 combinations) | Five invalid classifications × MHA/GQA × module/direct/update | Bool dtype, exact rank/shape, batch/query broadcast rejection, device mismatch; each public invocation must reach validation; attention-specific paths and mask/state exclusions | Candidate for shared validator classifications plus representative public-entry connections, subject to defect-injection evidence. A blanket 30→5 reduction is not accepted. |
| Direct/module equivalence / #1403 | `test_block.py::test_direct_forward_matches_module_call_for_valid_arguments` (3); `...values_gradients_and_state` (3) | Repeated value comparison | Exact outputs, all input/parameter gradients and state; MHA/GQA; CSWA all-valid versus partially-invalid state branch | Consolidate only after accounting for the distinct CSWA branch. Similar assertions alone do not prove containment. |
| RGB predictor matrix / #1403 | `tests/unit/tasks/ball_detection/model_io/test_predict.py` (24 combinations) | Two channel configurations × two concrete model classes × six bad RGB inputs; model allocation before forward patch | RGB frame/channel/layout/dtype/spatial/empty rejection; DPT/DeepQuery and channels 2/5 wiring; forward must not be reached; preserve concrete type dispatch | Candidate to move full invalid classes to the common validator and keep concrete entry checks. A generic mock that removes `isinstance` behavior is insufficient. |
| SLCS smoke / #1404 | `tests/integration/tasks/slcs/test_training_smoke.py::test_dataset_model_loss_backward_smoke` (4 combinations) | Frames 6/18 × context off/on; remaining-window training and four evaluation modes repeated | Padded/short/full window data, adapter→model→loss/backward, context gradient connection, optimizer mutation, checkpoint reload, full/no_rgb/detector_gap/rgb_only saved targets/masks/IDs/frames | Separate contracts and minimize repeated steps only if measured material. Keep integration context wiring even where model units check gradients. |
| Normal fit/callback / #1404 | `...test_real_rgb_profile_normal_fit_saves_monitored_checkpoint` | Production trainer/logger/callback initialization | Real monitored metric, regular fit, checkpoint callback and actual TensorBoard events | Retain a real normal-fit smoke. `fast_dev_run` or hand-written optimizer steps do not cover the callback contract. |
| MDD module structure / #1404 | `tests/unit/tasks/ball_detection/models/mdd/test_cnn.py` production-config structure test | Allocating full weights to count Conv2d/Conv3d and SwiGLU dimensions | Actual composed module tree and width/count invariants | Consider allocation-free construction only when material and supported by all initializers. Do not substitute config-value assertions; reject unsupported meta initialization. |
| Repository static scans / later candidate | `tests/e2e/development/test_architecture_boundaries.py` | Repeated repository file discovery/AST/reference parsing | Newly added files, deleted modules, dynamic/CLI entrypoints and transitive forward helpers; negative policy controls | Retain coverage. Consider immutable per-worker scan/index reuse if timing justifies it; do not remove scans because their count is large. |

All `test_block.py` entries above refer to
`tests/unit/utils/models/components/test_block.py`. These are contract families,
not a list of approved deletions. Untimed optional overlay/installer/DSH matrices
remain deferred. No test, marker, production model, or validator changes are made
by #1401.

## Fixed comparison procedure

1. Keep current default CPU command, source, complete selected inventory, dependency
   versions, runner OS/image/CPU/RAM, worker/thread counts, cache condition and
   competing load fixed when comparing scheduling. Compare every attempt, including
   cancellation and preparation failures.
2. When consolidation intentionally changes inventory, compare the surviving
   contract map and representative defect detection, plus case counts, fixture/CLI/
   model initialization counts and phase times. Do not call a layer move a reduction
   in tested cases.
3. Record n and the successful median/nearest-rank p95 alongside failure/cancel
   counts and cumulative all-job runner time. Small p95 samples remain reference.
   Unknown cache or missing preparation/post data prevents a warm-cache claim.
4. Rerun the complete CPU path in [README.md](README.md) at the chosen fixed source.
   Partial candidate measurements rank work; they do not certify whole-suite
   equivalence or the target of 50% / approximately 15 minutes.

Completed observations and limitations are recorded above. Successful same-source Actions comparisons follow integration.

After live job-log forwarding was added, a new set of three alternating matched 12-test pairs measured plain median **4.213 s** and instrumented median **4.467 s** (+0.254 s). All six runs passed with identical inventories. These startup-dominated samples have separate provenance in `overhead_live_log` in the machine evidence; differences from the earlier samples are not a between-version speedup.

## Runner scheduling observations

The initial four branch comparisons use the same source, refreshed profile and 12,795-case CPU inventory, including 13 new CI regression cases. The 10-shard follow-up adds three CI regression cases (12,798 total); production tests are unchanged, and its source is identified separately in the machine evidence. Each configuration has n=1. The earlier 4 × 2 reference uses the old profile and 12,782 cases; it is descriptive context, not a same-source control. Elapsed time starts at Actions `created_at` (including queue) and ends at completion of `Python`, including aggregation and post steps. Runner time sums every job, including lint and aggregation. All attempts and failures remain in [machine evidence](scheduling-evidence.json).

| Configuration / run | Python completion min | Shard min–max min (spread s) | All-job runner min | Test cache hits | Largest process RSS GiB | Failed cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [4 × 2 / 40986313121](https://github.com/Motoki0705/tennis-lab/actions/runs/40986313121) | 26.22 | 14.42–25.80 (683) | 82.60 | 4/4 | 1.70 | 1 |
| [4 × 4 / 40992996490](https://github.com/Motoki0705/tennis-lab/actions/runs/40992996490) | 20.53 | 12.62–20.13 (451) | 66.83 | 4/4 | 1.65 | 2 |
| [6 × 4 / 40995509440](https://github.com/Motoki0705/tennis-lab/actions/runs/40995509440) | 16.22 | 11.73–15.77 (242) | 87.98 | 6/6 | 1.64 | 2 |
| [6 × 2 / 40997143981](https://github.com/Motoki0705/tennis-lab/actions/runs/40997143981) | 18.30 | 9.93–17.88 (477) | 90.33 | 6/6 | 1.60 | 1 |
| [8 × 2 / 40999125159](https://github.com/Motoki0705/tennis-lab/actions/runs/40999125159) | 15.02 | 10.20–14.62 (265) | 104.20 | 8/8 | 1.51 | 1 |
| [10 × 2 / 41001256601](https://github.com/Motoki0705/tennis-lab/actions/runs/41001256601) | 14.33 | 9.57–13.88 (259) | 118.45 | 10/10 | 1.50 | 1 |

The profile refresh procedure and fixture boundaries are owned by [sharding](sharding.md#regenerate-measured-weights). Individual process RSS excludes test subprocesses and is not the simultaneous process-tree peak. All measured runners expose 4 vCPUs and about 16 GB RAM, but CPU generations vary. Every complete shard observation has all expected worker session finishes; no OOM or worker loss was observed. Native-library environment variables remain one; some existing tests override PyTorch intraop threads to 2/4 and leak that state into subsequent cases. Thus four workers can contend for the same CPUs despite the environment limits. Fixing those test bodies belongs to the parent.

All runs retain the known README.zh-CN failure. Both four-worker observations also fail `test_timeout_reaps_group_when_output_is_open_or_closed[True]`: the assertion sees a running child immediately after timeout cleanup. This failure is not ignored or converted to success. No successful warm-cache median or p95 is established from these failed observations. Source/profile and inventory checks still run, and a complete failing run reports `observations_complete=true` with `success=false`.

The optional local full collect-only observation was cancelled under competing host load and is not a performance or passing result. The Actions manifests and coverage artifacts provide the full execution proof. Repeat the chosen configuration and regenerate weights after the parent's fixture optimizations merge; hardware variation, limited sample count and test thread leakage limit extrapolation.

The selected default is 10 × 2: it is the smallest measured two-worker configuration completing Python within 900 s with only the known README failure. Eight shards took 901 s. Observed runner cost rises from 104.20 to 118.45 min (+13.7%), or +43.4% against the earlier 82.60 min reference. CPU generations and source differ, so the full cost difference cannot be attributed to adding jobs. The five new attempts consumed 467.80 runner min in total, including all failures; none was cancelled. Reprofile after the parent’s fixture optimizations before reducing the job count.
